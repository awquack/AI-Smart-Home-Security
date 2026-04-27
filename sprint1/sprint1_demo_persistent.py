"""
AI Home Security - Sprint 1 Demo
Improvements:
  - MOG2 background subtractor (handles lighting changes, far fewer false alarms)
  - Contour merging (groups nearby blobs into single detection box)
  - Audio alert (beep on detection) with cooldown and mute toggle
  - Live sensitivity controls (+/- keys)
  - Boxes persist on screen for 1 second with fade
"""

import cv2
import datetime
import time
import os
import threading
import numpy as np
from collections import deque

# ─── Audio ────────────────────────────────────────────────────────────────────

try:
    import sounddevice as sd

    def _beep(freq=880, duration=0.18, volume=0.6):
        """Generate and play a sine-wave beep in a background thread."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = volume * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Quick fade-out to avoid click
        fade = int(sample_rate * 0.02)
        wave[-fade:] *= np.linspace(1, 0, fade)
        sd.play(wave, sample_rate)

    def play_alert():
        """Play a two-tone alert without blocking the main loop."""
        def _play():
            _beep(880, 0.12)
            time.sleep(0.12)
            _beep(1100, 0.12)
        threading.Thread(target=_play, daemon=True).start()

    AUDIO_AVAILABLE = True

except ImportError:
    # Fallback: macOS system sound
    import subprocess

    def play_alert():
        threading.Thread(
            target=lambda: subprocess.run(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                capture_output=True,
            ),
            daemon=True,
        ).start()

    AUDIO_AVAILABLE = True

# ─── Setup ────────────────────────────────────────────────────────────────────

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

print("[INIT] Starting camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(2)

ret, frame1 = cap.read()
if not ret:
    print("[ERROR] Cannot read from camera.")
    cap.release()
    exit(1)

# ─── MOG2 background subtractor ───────────────────────────────────────────────
# history=300  – how many frames to build the background model
# varThreshold – higher = less sensitive; lower = more sensitive
# detectShadows – mark shadows separately (we ignore them later)
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=40, detectShadows=True
)

# Warm up the background model with the first frame
for _ in range(5):
    bg_sub.apply(frame1)

# ─── Parameters ───────────────────────────────────────────────────────────────
MIN_AREA       = 2500   # px²  – ignore tiny blobs
BLUR_SIZE      = 11     # must be odd
MERGE_DISTANCE = 60     # px   – merge contour boxes closer than this
PERSIST_TIME   = 1.2    # sec  – how long a box stays on screen
ALERT_COOLDOWN = 3.0    # sec  – min seconds between audio alerts

# ─── State ────────────────────────────────────────────────────────────────────
recent_detections: deque = deque(maxlen=20)
motion_count   = 0
frame_count    = 0
start_time     = time.time()
last_alert_time = 0.0
audio_muted    = False

print("=" * 60)
print("  AI HOME SECURITY - SPRINT 1 DEMO")
print("=" * 60)
print("\n  Camera  : READY")
print("  Engine  : MOG2 background subtractor")
print(f"  Min area: {MIN_AREA} px²")
print("\n  Controls:")
print("    q  – Quit")
print("    s  – Save snapshot")
print("    m  – Toggle audio mute")
print("    +  – Increase sensitivity (lower threshold)")
print("    -  – Decrease sensitivity (raise threshold)")
print("=" * 60)
print("\n[STATUS] Monitoring...\n")


# ─── Helper: merge overlapping / nearby bounding boxes ────────────────────────

def merge_boxes(boxes, gap):
    """
    Expand each box by `gap` pixels, merge overlapping results, then shrink back.
    Returns a list of merged (x, y, w, h) tuples.
    """
    if not boxes:
        return []
    rects = []
    for (x, y, w, h) in boxes:
        rects.append((x - gap, y - gap, x + w + gap, y + h + gap))  # x1,y1,x2,y2

    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(rects)
        for i, r1 in enumerate(rects):
            if used[i]:
                continue
            x1, y1, x2, y2 = r1
            for j, r2 in enumerate(rects):
                if i == j or used[j]:
                    continue
                # Check overlap
                if x1 <= r2[2] and x2 >= r2[0] and y1 <= r2[3] and y2 >= r2[1]:
                    x1 = min(x1, r2[0])
                    y1 = min(y1, r2[1])
                    x2 = max(x2, r2[2])
                    y2 = max(y2, r2[3])
                    used[j] = True
                    merged = True
            result.append((x1, y1, x2, y2))
            used[i] = True
        rects = result

    # Shrink back and clip to frame
    out = []
    for (x1, y1, x2, y2) in rects:
        x1 = max(0, x1 + gap)
        y1 = max(0, y1 + gap)
        x2 = max(0, x2 - gap)
        y2 = max(0, y2 - gap)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


# ─── Main loop ────────────────────────────────────────────────────────────────

while True:
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ── Motion mask via MOG2 ──────────────────────────────────────────────────
    mask = bg_sub.apply(frame)

    # Remove shadows (value 127 in shadow mode) – keep only foreground (255)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Denoise
    mask = cv2.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ── Find & filter contours ────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / h if h > 0 else 0
        if ar > 4.5 or ar < 0.22:   # filter extreme aspect ratios (wires, lines)
            continue
        raw_boxes.append((x, y, w, h))

    merged = merge_boxes(raw_boxes, MERGE_DISTANCE)

    # ── Register new detections ───────────────────────────────────────────────
    for box in merged:
        recent_detections.append({"box": box, "time": current_time, "id": motion_count})
        motion_count += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        x, y, w, h = box
        print(f"[{ts}]  MOTION #{motion_count} – area ~{w*h} px  at ({x},{y})")

    # ── Audio alert (with cooldown) ───────────────────────────────────────────
    if merged and not audio_muted:
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            play_alert()
            last_alert_time = current_time

    # ── Build display frame ───────────────────────────────────────────────────
    display_frame = frame.copy()

    active = [d for d in recent_detections if current_time - d["time"] < PERSIST_TIME]

    for det in active:
        age = current_time - det["time"]
        fade = max(0.25, 1.0 - age / PERSIST_TIME)   # 1.0 → 0.25
        x, y, w, h = det["box"]

        # Box colour: red when fresh, fading to green
        g = int(255 * fade)
        b = int(255 * (1 - fade))
        color = (b, g, 0)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

        # Label bar above box
        lx1, ly1 = x, max(0, y - 26)
        lx2, ly2 = min(639, x + 155), y
        cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (0, 0, 180), -1)
        cv2.putText(display_frame, f"MOTION #{det['id']}", (lx1 + 4, ly2 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # ── Status overlay ────────────────────────────────────────────────────────
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 110), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0)

    if active:
        status_text  = f"  INTRUSION DETECTED ({len(active)} zone{'s' if len(active)>1 else ''})"
        status_color = (0, 60, 255)
    else:
        status_text  = "  MONITORING – ALL CLEAR"
        status_color = (0, 220, 0)

    cv2.putText(display_frame, status_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    ts_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(display_frame, ts_str, (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    mute_label = "[MUTED]" if audio_muted else ""
    cv2.putText(display_frame,
                f"  Events: {motion_count}   FPS: {fps:.1f}   {mute_label}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    cv2.imshow("AI Home Security – Sprint 1 Demo", display_frame)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"snapshots/event_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, display_frame)
        print(f"[SNAP] Saved: {fname}")
    elif key == ord('m'):
        audio_muted = not audio_muted
        print(f"[AUDIO] {'Muted' if audio_muted else 'Unmuted'}")
    elif key in (ord('+'), ord('=')):
        # More sensitive → lower varThreshold
        vt = max(10, bg_sub.getVarThreshold() - 5)
        bg_sub.setVarThreshold(vt)
        print(f"[SENS] varThreshold → {vt}  (more sensitive)")
    elif key in (ord('-'), ord('_')):
        vt = min(120, bg_sub.getVarThreshold() + 5)
        bg_sub.setVarThreshold(vt)
        print(f"[SENS] varThreshold → {vt}  (less sensitive)")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print(f"  Total motion events : {motion_count}")
print(f"  Runtime             : {elapsed:.1f}s   Avg FPS: {fps:.1f}")
print("=" * 60)
