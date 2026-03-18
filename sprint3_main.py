"""
AI Home Security – Sprint 3  (detection engine)

Can run in two modes:
  Standalone : python sprint3_main.py          → opens OpenCV window
  Integrated : imported by app.py              → writes frames to shared buffer,
                                                 no OpenCV window needed
"""

import cv2
import datetime
import time
import os
import queue
import threading
import numpy as np
from collections import deque

import config
import database
import alerts

# ─── DB + snapshot dir ────────────────────────────────────────────────────────
os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
database.init_db()


# ─── YOLO (loaded once at module level) ───────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO_CLS
    _yolo_model = _YOLO_CLS("yolov8n.pt")
    _yolo_model.fuse()
    YOLO_AVAILABLE = True
    print("[YOLO] YOLOv8n ready")
except Exception as _e:
    YOLO_AVAILABLE = False
    print(f"[YOLO] Unavailable – {_e}")

_YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    15: "cat",   16: "dog",
}

def yolo_detect_frame(frame):
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
                    f"{_YOLO_CLASSES[cls]} {conf:.0%}", conf))
    return out


# ─── MFCC Audio (started once at module level) ────────────────────────────────
_audio_q      = queue.Queue(maxsize=20)
_audio_event  = threading.Event()
_audio_label  = ""
_audio_lock   = threading.Lock()
AUDIO_MONITOR = False
_mic_stream   = None

try:
    import sounddevice as sd
    import librosa

    _SR          = 22050
    _CHUNK_N     = int(_SR * 0.5)
    _N_MFCC      = 13
    _ENERGY_TH   = 0.005
    _ENERGY_LOUD = 0.05
    _MFCC_DIST   = 25.0   # raised from 8.0 — silence baseline sits at ~22, so threshold must be above that
    _BASELINE_N  = 10
    _mfcc_base   = None
    _base_buf    = []
    _post_cal    = 0

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

            if _mfcc_base is None:
                if len(_base_buf) == 0:
                    print("[AUDIO] Calibrating baseline – stay quiet for 5 s…")
                _base_buf.append(mmean)
                print(f"[AUDIO] Calibrating… {len(_base_buf)}/{_BASELINE_N}  rms={rms:.4f}")
                if len(_base_buf) >= _BASELINE_N:
                    _mfcc_base = np.mean(_base_buf, axis=0)
                    print("[AUDIO] Baseline ready – listening for anomalies")
                continue

            dist = float(np.linalg.norm(mmean - _mfcc_base))
            _post_cal += 1
            if _post_cal <= 20 or _post_cal % 10 == 0:
                print(f"[AUDIO] rms={rms:.4f}  mfcc_dist={dist:.1f}")

            now = time.time()
            if now - last_event > 2.0:
                if rms > _ENERGY_LOUD:
                    label, fired = f"LOUD SOUND  rms={rms:.3f}", True
                elif rms > _ENERGY_TH and dist > _MFCC_DIST:
                    label, fired = f"SOUND  rms={rms:.3f}  dist={dist:.0f}", True
                else:
                    label, fired = "", False

                if fired:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    with _audio_lock:
                        _audio_label = label
                    _audio_event.set()
                    print(f"[{ts}] AUDIO ANOMALY – {label}")
                    database.log_event("audio", label=label)
                    alerts.send_alert("audio", label=label)
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
        return lambda: None

play_alert = _make_play_alert()


# ─── Helpers ──────────────────────────────────────────────────────────────────
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
                    merged  = True
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


def save_snapshot(frame, prefix="event"):
    fname = (f"{config.SNAPSHOT_DIR}/{prefix}_"
             f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(fname, frame)
    return fname


# ─── Main detection loop ──────────────────────────────────────────────────────
def run_detection(shared_state=None, show_window=True):
    """
    Run the full detection pipeline.

    shared_state : SharedState instance from shared.py
                   If provided, annotated frames are pushed into it for the
                   Flask dashboard MJPEG stream.
    show_window  : If True, open an OpenCV window (standalone mode).
                   Set False when called from app.py.
    """

    # ── Camera scan ───────────────────────────────────────────────────────────
    print("[INIT] Scanning cameras…")
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
        print("[ERROR] No cameras found – detection stopped.")
        if shared_state:
            shared_state.status_text = "No camera found"
            shared_state.running     = False
        return

    camera_index = 1 if len(_available) > 1 else _available[0]
    print(f"[INIT] Using camera index {camera_index}\n")

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(2)

    ret, frame1 = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera.")
        cap.release()
        return

    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Camera resolution: {FRAME_W}x{FRAME_H}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=700, varThreshold=120, detectShadows=True
    )
    for _ in range(15):
        bg_sub.apply(frame1)

    # ── Parameters ────────────────────────────────────────────────────────────
    MIN_AREA        = config.MIN_AREA
    BLUR_SIZE       = config.BLUR_SIZE
    MERGE_DISTANCE  = config.MERGE_DISTANCE
    MAX_BOXES       = config.MAX_BOXES
    PERSIST_TIME    = config.PERSIST_TIME
    ALERT_COOLDOWN  = config.ALERT_COOLDOWN
    SHAKE_RATIO_MAX = config.SHAKE_RATIO_MAX
    CONFIRM_FRAMES  = config.CONFIRM_FRAMES
    YOLO_EVERY      = config.YOLO_EVERY
    FUSION_WINDOW   = config.FUSION_WINDOW

    # ── State ─────────────────────────────────────────────────────────────────
    recent_detections = deque(maxlen=30)
    yolo_detections   = deque(maxlen=15)
    motion_count      = 0
    frame_count       = 0
    start_time        = time.time()
    last_alert_time   = 0.0
    audio_muted       = False
    consec_motion     = 0
    frames_since_yolo = YOLO_EVERY
    last_motion_time  = 0.0
    last_sound_time   = 0.0
    high_conf_until   = 0.0
    high_conf_logged  = False

    print("=" * 62)
    print("  AI HOME SECURITY – SPRINT 3")
    print("=" * 62)
    print(f"  YOLO   : {'ON (YOLOv8n)' if YOLO_AVAILABLE else 'OFF'}")
    print(f"  Audio  : {'ON' if AUDIO_MONITOR else 'OFF'}")
    print(f"  Mode   : {'Standalone (OpenCV window)' if show_window else 'Integrated (Flask dashboard)'}")
    print(f"  DB     : {config.DB_PATH}")
    if show_window:
        print("  Keys   : q=Quit  s=Snapshot  m=Mute  +/-=Sensitivity")
    print("=" * 62)

    if shared_state:
        shared_state.running     = True
        shared_state.status_text = "Monitoring"

    if show_window:
        WIN_NAME = "AI Home Security – Sprint 3"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        frame_count       += 1
        frames_since_yolo += 1
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0

        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Motion mask
        mask = bg_sub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Camera-shake filter
        motion_ratio = np.count_nonzero(mask) / mask.size
        is_shaking   = motion_ratio > SHAKE_RATIO_MAX
        if is_shaking:
            consec_motion = 0

        # Contour detection
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
                if ar > 4.5 or ar < 0.22:
                    continue
                raw_boxes.append((x, y, w, h))
            merged = merge_boxes(raw_boxes, MERGE_DISTANCE)
            merged = sorted(merged, key=lambda b: b[2]*b[3], reverse=True)[:MAX_BOXES]

        # Temporal confirmation
        if merged:
            consec_motion += 1
        elif not is_shaking:
            consec_motion = 0
        confirmed = merged if consec_motion >= CONFIRM_FRAMES else []

        # YOLO
        if confirmed and frames_since_yolo >= YOLO_EVERY:
            new_yolo = yolo_detect_frame(frame)
            for det in new_yolo:
                x, y, w, h, lbl, conf = det
                yolo_detections.append({"box": (x, y, w, h), "label": lbl,
                                        "time": current_time})
                snap = save_snapshot(frame, "yolo") if config.AUTO_SNAPSHOT else None
                database.log_event("yolo", label=lbl, confidence=conf,
                                    snapshot_path=snap, x=x, y=y, w=w, h=h)
                alerts.send_alert("yolo", label=lbl, snapshot_path=snap)
                print(f"[YOLO] {lbl}  conf={conf:.0%}")
            frames_since_yolo = 0

        # Motion events
        if confirmed:
            last_motion_time = current_time
        for box in confirmed:
            recent_detections.append({"box": box, "time": current_time, "id": motion_count})
            x, y, w, h = box
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}]  MOTION #{motion_count}  area~{w*h}px  at ({x},{y})")
            snap = save_snapshot(frame, "motion") if config.AUTO_SNAPSHOT else None
            database.log_event("motion", snapshot_path=snap,
                                area=w*h, x=x, y=y, w=w, h=h)
            alerts.send_alert("motion",
                              label=f"Motion at ({x},{y}) area={w*h}px",
                              snapshot_path=snap)
            motion_count += 1

        # Audio tracking
        if AUDIO_MONITOR and _audio_event.is_set():
            last_sound_time = current_time

        # Fusion
        motion_recent = (current_time - last_motion_time) < FUSION_WINDOW
        sound_recent  = (current_time - last_sound_time)  < FUSION_WINDOW
        is_high_conf  = (motion_recent and sound_recent
                         and last_motion_time > 0 and last_sound_time > 0)

        if is_high_conf:
            if current_time > high_conf_until:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] *** HIGH CONFIDENCE ALERT – motion + sound ***")
                if not high_conf_logged:
                    snap = save_snapshot(frame, "highconf") if config.AUTO_SNAPSHOT else None
                    database.log_event("high_confidence",
                                       label="Motion + audio fusion",
                                       snapshot_path=snap)
                    alerts.send_alert("high_confidence",
                                      label="Motion AND sound detected simultaneously",
                                      snapshot_path=snap)
                    high_conf_logged = True
            high_conf_until = current_time + 3.0
        else:
            high_conf_logged = False

        # Local beep
        if confirmed and not audio_muted:
            if current_time - last_alert_time >= ALERT_COOLDOWN:
                play_alert()
                last_alert_time = current_time

        # ── Build annotated display frame ──────────────────────────────────
        display_frame = frame.copy()
        active_motion = [d for d in recent_detections
                         if current_time - d["time"] < PERSIST_TIME]
        active_yolo   = [d for d in yolo_detections
                         if current_time - d["time"] < PERSIST_TIME]

        for det in active_motion:
            age  = current_time - det["time"]
            fade = max(0.25, 1.0 - age / PERSIST_TIME)
            x, y, w, h = det["box"]
            g = int(255 * fade);  b = int(255 * (1 - fade))
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (b, g, 0), 3)
            lx1, ly1 = x, max(0, y - 26)
            lx2, ly2 = min(FRAME_W - 1, x + 120), y
            cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (0, 0, 160), -1)
            cv2.putText(display_frame, f"MOTION #{det['id']}", (lx1+4, ly2-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        for yd in active_yolo:
            x, y, w, h = yd["box"]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
            cv2.putText(display_frame, yd["label"], (x+4, max(10, y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Status bar overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, 140), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0)

        hc_active = current_time < high_conf_until
        if is_shaking:
            status_text  = "  CAMERA MOVING – stabilising…"
            status_color = (60, 180, 255)
        elif hc_active:
            status_text  = "  !! HIGH CONFIDENCE ALERT – motion + sound !!"
            status_color = (0, 0, 255)
            if int(current_time * 4) % 2:
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
        cv2.putText(display_frame,
                    f"  Events:{motion_count}  FPS:{fps:.1f}  "
                    f"consec:{consec_motion}/{CONFIRM_FRAMES}  {mute_label}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        audio_is_active = _audio_event.is_set() if AUDIO_MONITOR else False
        if audio_is_active:
            with _audio_lock:
                cur_audio_lbl = _audio_label
            fuse_tag = "  [FUSED]" if hc_active else ""
            cv2.putText(display_frame, f"  [MIC] {cur_audio_lbl}{fuse_tag}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

        counts = database.get_event_counts()
        cv2.putText(display_frame, f"  DB events: {sum(counts.values())}",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 120), 1)

        # ── Push annotated frame to shared buffer (integrated mode) ────────
        if shared_state:
            shared_state.set_frame(display_frame)
            shared_state.motion_count = motion_count
            shared_state.audio_active = audio_is_active
            shared_state.high_conf    = hc_active
            shared_state.status_text  = status_text.strip()
            shared_state.fps          = round(fps, 1)
            shared_state.consec       = consec_motion

        # ── OpenCV window (standalone mode only) ───────────────────────────
        if show_window:
            cv2.imshow(WIN_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = save_snapshot(display_frame, "manual")
                database.log_event("manual_snapshot", snapshot_path=fname)
                print(f"[SNAP] Saved: {fname}")
            elif key == ord('m'):
                audio_muted = not audio_muted
                print(f"[AUDIO] {'Muted' if audio_muted else 'Unmuted'}")
            elif key in (ord('+'), ord('=')):
                vt = max(10, bg_sub.getVarThreshold() - 5)
                bg_sub.setVarThreshold(vt)
                print(f"[SENS] varThreshold → {vt}")
            elif key in (ord('-'), ord('_')):
                vt = min(150, bg_sub.getVarThreshold() + 5)
                bg_sub.setVarThreshold(vt)
                print(f"[SENS] varThreshold → {vt}")
        else:
            # Integrated mode — yield CPU to Flask thread
            time.sleep(0.001)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if AUDIO_MONITOR and _mic_stream:
        _mic_stream.stop()
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    if shared_state:
        shared_state.running     = False
        shared_state.status_text = "Stopped"

    elapsed = time.time() - start_time
    fps     = frame_count / elapsed if elapsed > 0 else 0
    counts  = database.get_event_counts()
    print("\n" + "=" * 62)
    print("  SPRINT 3 SESSION COMPLETE")
    print(f"  Motion events : {motion_count}  |  Runtime: {elapsed:.1f}s  |  FPS: {fps:.1f}")
    print(f"  DB breakdown  : {counts}")
    print("=" * 62)


# ─── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_detection(shared_state=None, show_window=True)
