"""
AI Home Security – Sprint 2
  - YOLO v8 nano : classifies objects in motion zones (person, cat, dog, car…)
  - MFCC audio   : microphone thread flags anomalous sounds using MFCCs
  - Shake filter  : skips detections when >15% of the frame is moving (camera wobble)
  - Temporal gate : motion must persist for 3+ consecutive frames before alerting
  - Raised thresholds: much less sensitive defaults than Sprint 1
"""

import cv2
import datetime
import time
import os
import queue
import threading
import numpy as np
from collections import deque

# ─── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO_CLS
    _yolo_model = _YOLO_CLS("yolov8n.pt")
    _yolo_model.fuse()
    YOLO_AVAILABLE = True
    print("[YOLO] YOLOv8n ready")
except Exception as _e:
    YOLO_AVAILABLE = False
    print(f"[YOLO] Unavailable – {_e}")

# Security-relevant COCO classes
_YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    15: "cat",   16: "dog",
}

def yolo_detect_frame(frame):
    """Run YOLO on the whole frame; return list of (x,y,w,h,label)."""
    if not YOLO_AVAILABLE:
        return []
    results = _yolo_model(frame, verbose=False, conf=0.35)[0]
    out = []
    for det in results.boxes:
        cls = int(det.cls[0])
        if cls not in _YOLO_CLASSES:
            continue
        conf  = float(det.conf[0])
        bx, by, bx2, by2 = map(int, det.xyxy[0])
        out.append((bx, by, bx2 - bx, by2 - by,
                    f"{_YOLO_CLASSES[cls]} {conf:.0%}"))
    return out


# ─── MFCC Audio Monitoring ────────────────────────────────────────────────────
_audio_q      = queue.Queue(maxsize=20)
_audio_event  = threading.Event()
_audio_label  = ""
_audio_lock   = threading.Lock()
AUDIO_MONITOR = False

try:
    import sounddevice as sd
    import librosa

    _SR          = 22050
    _CHUNK_N     = int(_SR * 0.5)     # 0.5 s chunks
    _N_MFCC      = 13
    _ENERGY_TH   = 0.005              # RMS threshold for MFCC check
    _ENERGY_LOUD = 0.05               # RMS fast-path: always fire if this loud (clap/knock)
    _MFCC_DIST   = 8.0                # L2 distance from baseline
    _BASELINE_N  = 10                 # calibrate over first 10 chunks (~5 s) – stay quiet!
    _mfcc_base   = None
    _base_buf    = []
    _post_cal    = 0                  # chunks processed after calibration (for debug)

    def _sd_cb(indata, frames, t, status):
        try:
            _audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _mfcc_worker():
        global _mfcc_base, _audio_label, _post_cal
        last_event = 0.0

        while True:
            chunk = _audio_q.get()
            mfcc  = librosa.feature.mfcc(y=chunk, sr=_SR, n_mfcc=_N_MFCC)
            mmean = np.mean(mfcc, axis=1)
            rms   = float(np.sqrt(np.mean(chunk ** 2)))

            # Build baseline – keep quiet for the first ~5 seconds
            if _mfcc_base is None:
                if len(_base_buf) == 0:
                    print("[AUDIO] Calibrating baseline – STAY QUIET for 5 seconds…")
                _base_buf.append(mmean)
                print(f"[AUDIO] Calibrating… {len(_base_buf)}/{_BASELINE_N}  rms={rms:.4f}")
                if len(_base_buf) >= _BASELINE_N:
                    _mfcc_base = np.mean(_base_buf, axis=0)
                    print("[AUDIO] Baseline ready – now listening for anomalies")
                continue

            # Always print first 20 chunks after calibration so user can see live values
            dist = float(np.linalg.norm(mmean - _mfcc_base))
            _post_cal += 1
            if _post_cal <= 20 or _post_cal % 10 == 0:
                print(f"[AUDIO] rms={rms:.4f}  mfcc_dist={dist:.1f}  "
                      f"(fire if rms>{_ENERGY_TH} & dist>{_MFCC_DIST}, "
                      f"or rms>{_ENERGY_LOUD})")

            now = time.time()
            if now - last_event > 2.0:
                # Fast-path: very loud sound fires immediately regardless of MFCC
                if rms > _ENERGY_LOUD:
                    label = f"LOUD SOUND  rms={rms:.3f}"
                    fired = True
                # Normal path: energy + MFCC deviation from baseline
                elif rms > _ENERGY_TH and dist > _MFCC_DIST:
                    label = f"SOUND  rms={rms:.3f}  dist={dist:.0f}"
                    fired = True
                else:
                    fired = False

                if fired:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    with _audio_lock:
                        _audio_label = label
                    _audio_event.set()
                    print(f"[{ts}] AUDIO ANOMALY – {label}")
                    last_event = now
                else:
                    _audio_event.clear()

    _mic_stream = sd.InputStream(
        samplerate=_SR, channels=1, blocksize=_CHUNK_N, callback=_sd_cb
    )
    _mic_stream.start()
    threading.Thread(target=_mfcc_worker, daemon=True).start()
    AUDIO_MONITOR = True
    print("[AUDIO] MFCC monitoring active")

except Exception as _ae:
    print(f"[AUDIO] MFCC unavailable – {_ae}")


# ─── Alert beep ───────────────────────────────────────────────────────────────
def _make_play_alert():
    try:
        import sounddevice as _sd

        def play_alert():
            def _do():
                sr = 44100
                for freq, dur in [(880, 0.12), (1100, 0.12)]:
                    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                    w = (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                    w[-int(sr * 0.02):] *= np.linspace(1, 0, int(sr * 0.02))
                    _sd.play(w, sr)
                    time.sleep(dur + 0.01)
            threading.Thread(target=_do, daemon=True).start()
        return play_alert

    except ImportError:
        import subprocess

        def play_alert():
            threading.Thread(
                target=lambda: subprocess.run(
                    ["afplay", "/System/Library/Sounds/Ping.aiff"],
                    capture_output=True,
                ),
                daemon=True,
            ).start()
        return play_alert

play_alert = _make_play_alert()


# ─── Camera setup ─────────────────────────────────────────────────────────────
os.makedirs("snapshots", exist_ok=True)

# Scan available cameras and auto-select the built-in Mac camera.
# The built-in FaceTime camera is usually the lowest index that isn't a phone.
# Change CAMERA_INDEX manually if the wrong camera opens.
CAMERA_INDEX = None
print("[INIT] Scanning cameras...")
_available = []
for _i in range(6):
    _t = cv2.VideoCapture(_i)
    if _t.isOpened():
        _w = int(_t.get(cv2.CAP_PROP_FRAME_WIDTH))
        _h = int(_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"         [{_i}] available  ({_w}x{_h})")
        _available.append(_i)
    _t.release()

if not _available:
    print("[ERROR] No cameras found.")
    exit(1)

# Index 1 = built-in FaceTime HD camera (phone Continuity Camera takes index 0).
# Change back to _available[0] if the wrong camera opens.
CAMERA_INDEX = 1 if len(_available) > 1 else _available[0]
print(f"[INIT] Using camera index {CAMERA_INDEX}  "
      f"(change CAMERA_INDEX at top of file if wrong)\n")

print("[INIT] Starting camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
time.sleep(2)

ret, frame1 = cap.read()
if not ret:
    print("[ERROR] Cannot open camera.")
    cap.release()
    exit(1)

FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INIT] Camera resolution: {FRAME_W}x{FRAME_H}")

# ─── Background subtractor ────────────────────────────────────────────────────
# varThreshold=70 (Sprint 1 used 40): much less reactive to small pixel changes
# history=500: slower to adapt = more stable background model
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=700, varThreshold=120, detectShadows=True
)
for _ in range(15):
    bg_sub.apply(frame1)

# ─── Parameters ───────────────────────────────────────────────────────────────
MIN_AREA        = 20000   # px²  – only large blobs (a person = ~100k+ px at 720p)
BLUR_SIZE       = 21      # heavier blur = fewer speckles
MERGE_DISTANCE  = 150     # px   – aggressively merge nearby boxes into one
MAX_BOXES       = 2       # keep only the 2 largest boxes per frame
PERSIST_TIME    = 1.5     # sec  – how long boxes stay on screen
ALERT_COOLDOWN  = 3.0     # sec  – min gap between audio alerts
SHAKE_RATIO_MAX = 0.20    # if >20% of pixels are "moving" = camera shake → skip
CONFIRM_FRAMES  = 5       # motion must appear in N consecutive frames to count
YOLO_EVERY      = 5       # run YOLO every N frames (not every frame)
FUSION_WINDOW   = 2.0     # sec  – motion + sound within this window = HIGH CONFIDENCE

# ─── State ────────────────────────────────────────────────────────────────────
recent_detections = deque(maxlen=30)
yolo_detections   = deque(maxlen=15)   # {box, label, time}
motion_count      = 0
frame_count       = 0
start_time        = time.time()
last_alert_time   = 0.0
audio_muted       = False
consec_motion     = 0
frames_since_yolo = YOLO_EVERY        # run on first confirmed detection
last_motion_time  = 0.0               # timestamp of last confirmed motion
last_sound_time   = 0.0               # timestamp of last audio anomaly
high_conf_until   = 0.0               # show HIGH CONFIDENCE banner until this time

print("=" * 62)
print("  AI HOME SECURITY – SPRINT 2")
print("=" * 62)
print(f"  YOLO object detection  : {'ON (YOLOv8n)' if YOLO_AVAILABLE else 'OFF'}")
print(f"  MFCC audio monitoring  : {'ON' if AUDIO_MONITOR else 'OFF'}")
print(f"  Camera-shake filter    : ON  (>{SHAKE_RATIO_MAX*100:.0f}% frame pixels = shake)")
print(f"  Temporal confirmation  : {CONFIRM_FRAMES} consecutive frames")
print(f"  Min motion area        : {MIN_AREA} px²")
print()
print("  Controls:")
print("    q  – Quit")
print("    s  – Save snapshot")
print("    m  – Toggle audio mute")
print("    +  – Increase sensitivity (lower threshold)")
print("    -  – Decrease sensitivity (raise threshold)")
print("=" * 62)
print("[STATUS] Monitoring...\n")

WIN_NAME = "AI Home Security – Sprint 2"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1280, 720)


# ─── Helper: merge nearby bounding boxes ──────────────────────────────────────
def merge_boxes(boxes, gap):
    if not boxes:
        return []
    rects = [(x-gap, y-gap, x+w+gap, y+h+gap) for (x, y, w, h) in boxes]
    merged = True
    while merged:
        merged = False
        result, used = [], [False] * len(rects)
        for i, r1 in enumerate(rects):
            if used[i]:
                continue
            x1, y1, x2, y2 = r1
            for j, r2 in enumerate(rects):
                if i == j or used[j]:
                    continue
                if x1 <= r2[2] and x2 >= r2[0] and y1 <= r2[3] and y2 >= r2[1]:
                    x1, y1 = min(x1, r2[0]), min(y1, r2[1])
                    x2, y2 = max(x2, r2[2]), max(y2, r2[3])
                    used[j] = True
                    merged = True
            result.append((x1, y1, x2, y2))
            used[i] = True
        rects = result
    out = []
    for (x1, y1, x2, y2) in rects:
        x1, y1 = max(0, x1 + gap), max(0, y1 + gap)
        x2, y2 = max(0, x2 - gap), max(0, y2 - gap)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


# ─── Main loop ────────────────────────────────────────────────────────────────
while True:
    frame_count      += 1
    frames_since_yolo += 1
    elapsed = time.time() - start_time
    fps     = frame_count / elapsed if elapsed > 0 else 0

    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ── Motion mask via MOG2 ──────────────────────────────────────────────────
    mask = bg_sub.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ── Camera-shake filter ───────────────────────────────────────────────────
    # When you're holding the camera, the whole frame shifts → huge mask coverage.
    # If >SHAKE_RATIO_MAX of pixels are flagged, assume it's camera movement.
    motion_ratio = np.count_nonzero(mask) / mask.size
    is_shaking   = motion_ratio > SHAKE_RATIO_MAX

    if is_shaking:
        consec_motion = 0  # reset temporal counter

    # ── Contour detection (only when camera is stable) ────────────────────────
    merged = []
    if not is_shaking:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            if ar > 4.5 or ar < 0.22:   # filter extreme aspect ratios
                continue
            raw_boxes.append((x, y, w, h))
        merged = merge_boxes(raw_boxes, MERGE_DISTANCE)
        # Keep only the largest MAX_BOXES to avoid flooding the screen
        merged = sorted(merged, key=lambda b: b[2]*b[3], reverse=True)[:MAX_BOXES]

    # ── Temporal confirmation ─────────────────────────────────────────────────
    # Require motion in N consecutive frames – eliminates single-frame glitches
    if merged:
        consec_motion += 1
    elif not is_shaking:
        consec_motion = 0

    confirmed = merged if consec_motion >= CONFIRM_FRAMES else []

    # ── YOLO (run every YOLO_EVERY frames when motion is confirmed) ───────────
    if confirmed and frames_since_yolo >= YOLO_EVERY:
        new_yolo = yolo_detect_frame(frame)
        for (x, y, w, h, lbl) in new_yolo:
            yolo_detections.append({"box": (x, y, w, h), "label": lbl,
                                    "time": current_time})
        frames_since_yolo = 0

    # ── Register confirmed motion events ──────────────────────────────────────
    if confirmed:
        last_motion_time = current_time
    for box in confirmed:
        recent_detections.append(
            {"box": box, "time": current_time, "id": motion_count}
        )
        motion_count += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        x, y, w, h = box
        print(f"[{ts}]  MOTION #{motion_count}  area~{w*h}px  at ({x},{y})")

    # ── Track audio event timestamps for fusion ────────────────────────────────
    if AUDIO_MONITOR and _audio_event.is_set():
        last_sound_time = current_time

    # ── Fusion: motion + sound within FUSION_WINDOW → HIGH CONFIDENCE ─────────
    motion_recent = (current_time - last_motion_time) < FUSION_WINDOW
    sound_recent  = (current_time - last_sound_time)  < FUSION_WINDOW
    if motion_recent and sound_recent and last_motion_time > 0 and last_sound_time > 0:
        if current_time > high_conf_until:   # first time entering high-conf state
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] *** HIGH CONFIDENCE ALERT – motion + sound within "
                  f"{FUSION_WINDOW:.0f}s ***")
        high_conf_until = current_time + 3.0   # keep banner for 3 s after last trigger

    # ── Audio alert ───────────────────────────────────────────────────────────
    if confirmed and not audio_muted:
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            play_alert()
            last_alert_time = current_time

    # ── Build display ──────────────────────────────────────────────────────────
    display_frame = frame.copy()
    active_motion = [d for d in recent_detections
                     if current_time - d["time"] < PERSIST_TIME]
    active_yolo   = [d for d in yolo_detections
                     if current_time - d["time"] < PERSIST_TIME]

    # Draw motion boxes (red→green fade)
    for det in active_motion:
        age  = current_time - det["time"]
        fade = max(0.25, 1.0 - age / PERSIST_TIME)
        x, y, w, h = det["box"]
        g = int(255 * fade)
        b = int(255 * (1 - fade))
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (b, g, 0), 3)
        lx1, ly1 = x, max(0, y - 26)
        lx2, ly2 = min(FRAME_W - 1, x + 120), y
        cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (0, 0, 160), -1)
        cv2.putText(display_frame, f"MOTION #{det['id']}", (lx1+4, ly2-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

    # Draw YOLO boxes (cyan / blue)
    for yd in active_yolo:
        x, y, w, h = yd["box"]
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
        cv2.putText(display_frame, yd["label"], (x+4, max(10, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    # ── Status bar ────────────────────────────────────────────────────────────
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_W, 135), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0)

    is_high_conf = current_time < high_conf_until

    if is_shaking:
        status_text  = "  CAMERA MOVING – stabilising…"
        status_color = (60, 180, 255)
    elif is_high_conf:
        status_text  = "  !! HIGH CONFIDENCE ALERT – motion + sound !!"
        status_color = (0, 0, 255)
        # Flash the entire status bar red when in high-conf mode
        flash = int((current_time * 4) % 2)   # toggles 0/1 at ~4 Hz
        if flash:
            cv2.rectangle(display_frame, (0, 0), (FRAME_W, 45), (0, 0, 180), -1)
    elif active_motion:
        status_text  = f"  INTRUSION DETECTED ({len(active_motion)} zone{'s' if len(active_motion)>1 else ''})"
        status_color = (0, 60, 255)
    else:
        status_text  = "  MONITORING – ALL CLEAR"
        status_color = (0, 220, 0)

    cv2.putText(display_frame, status_text, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(display_frame,
                datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    mute_label = "[MUTED]" if audio_muted else ""
    cv2.putText(
        display_frame,
        f"  Events:{motion_count}  FPS:{fps:.1f}  "
        f"consec:{consec_motion}/{CONFIRM_FRAMES}  {mute_label}",
        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1,
    )

    # Audio event indicator
    audio_is_active = _audio_event.is_set() if AUDIO_MONITOR else False
    if audio_is_active:
        with _audio_lock:
            cur_audio_lbl = _audio_label
        fuse_tag = "  [FUSED]" if is_high_conf else ""
        cv2.putText(display_frame, f"  [MIC] {cur_audio_lbl}{fuse_tag}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    cv2.imshow(WIN_NAME, display_frame)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = (f"snapshots/event_"
                 f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(fname, display_frame)
        print(f"[SNAP] Saved: {fname}")
    elif key == ord('m'):
        audio_muted = not audio_muted
        print(f"[AUDIO] {'Muted' if audio_muted else 'Unmuted'}")
    elif key in (ord('+'), ord('=')):
        vt = max(10, bg_sub.getVarThreshold() - 5)
        bg_sub.setVarThreshold(vt)
        print(f"[SENS] varThreshold → {vt}  (more sensitive)")
    elif key in (ord('-'), ord('_')):
        vt = min(150, bg_sub.getVarThreshold() + 5)
        bg_sub.setVarThreshold(vt)
        print(f"[SENS] varThreshold → {vt}  (less sensitive)")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
if AUDIO_MONITOR:
    _mic_stream.stop()

cap.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
fps     = frame_count / elapsed if elapsed > 0 else 0
print("\n" + "=" * 62)
print("  SPRINT 2 COMPLETE")
print(f"  Total events : {motion_count}")
print(f"  Runtime      : {elapsed:.1f}s   Avg FPS: {fps:.1f}")
print("=" * 62)
