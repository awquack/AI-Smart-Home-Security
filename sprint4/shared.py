# shared.py – Thread-safe state shared between detection and Flask dashboard
#
# sprint3_main.py  →  writes frames + status into SharedState
# app.py           →  reads frames + status out of SharedState
#
# One global instance `shared` is imported by both modules.

import threading
import cv2


class SharedState:
    def __init__(self):
        self._lock        = threading.Lock()
        self._frame       = None      # latest annotated BGR frame (numpy array)

        # ── Status fields (written by detection, read by dashboard) ──────────
        self.running      = False     # is detection loop active?
        self.status_text  = "Starting…"
        self.motion_count = 0
        self.audio_active = False
        self.high_conf    = False
        self.fps          = 0.0
        self.consec       = 0         # consecutive motion frames

    # ── Frame buffer ──────────────────────────────────────────────────────────

    def set_frame(self, frame):
        """Called by detection thread to push the latest annotated frame."""
        with self._lock:
            self._frame = frame.copy()

    def get_jpeg(self, quality=70):
        """Called by Flask MJPEG route to pull the latest frame as JPEG bytes."""
        with self._lock:
            if self._frame is None:
                return None
            _, buf = cv2.imencode(
                ".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            return buf.tobytes()

    def has_frame(self):
        with self._lock:
            return self._frame is not None


# Single global instance — import this in both app.py and sprint3_main.py
shared = SharedState()
