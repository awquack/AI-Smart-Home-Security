# fix_clips.py – Re-encode existing mp4v clips to H.264 for browser playback
# Run once: python fix_clips.py

import os
import cv2

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "clips")

if not os.path.isdir(CLIPS_DIR):
    print("[FIX] No clips folder found.")
    exit()

files = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")]
if not files:
    print("[FIX] No clips to convert.")
    exit()

print(f"[FIX] Found {len(files)} clip(s) to re-encode…")

for fname in files:
    src  = os.path.join(CLIPS_DIR, fname)
    tmp  = src.replace(".mp4", "_h264.mp4")

    cap  = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[FIX] Cannot open {fname} — skipping")
        continue

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(tmp, fourcc, fps, (w, h))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
        print(f"[FIX] H.264 unavailable for {fname} — keeping original codec")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1

    cap.release()
    out.release()

    if count > 0:
        os.remove(src)
        os.rename(tmp, src)
        print(f"[FIX] Re-encoded: {fname}  ({count} frames)")
    else:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[FIX] Empty clip — removed: {fname}")
        os.remove(src)

print("[FIX] Done — refresh the Clips page in your browser.")
