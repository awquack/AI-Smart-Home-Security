"""
AI Home Security – Sprint 4  (standalone entry point)

Sprint 4 adds identity-aware detection on top of Sprint 3:
  • Face recognition — known vs intruder classification
  • Distance-from-camera estimation for every detected person
  • Face enrollment through the web dashboard (/faces)
  • Intruder alerts via Telegram / Email with snapshot
  • Extended database schema (face_recognized, intruder events)

Run modes:
  Standalone : python sprint4_main.py          → opens OpenCV window
  Integrated : python app.py                   → Flask dashboard + MJPEG stream

Usage:
  python sprint4_main.py
"""

from sprint3_main import run_detection

if __name__ == "__main__":
    run_detection(shared_state=None, show_window=True)
