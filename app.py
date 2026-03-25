# app.py – Flask dashboard  (integrated with detection engine)
#
# How it works:
#   1. Imports `shared` (SharedState instance) from shared.py
#   2. Imports `run_detection` from sprint3_main (YOLO + audio load at import time)
#   3. Starts run_detection(shared, show_window=False) in a background thread
#      → detection writes annotated frames into shared.set_frame()
#   4. Flask /video_feed reads shared.get_jpeg() and streams it as MJPEG
#   5. Flask routes read events from SQLite via database.py
#
# Run: python app.py
# Open: http://localhost:5000

import os
import csv
import cv2
import threading
import time
import numpy as np
from functools import wraps
from flask import (Flask, Response, render_template, redirect,
                   url_for, request, session, jsonify, send_from_directory)

import config
import database
from shared import shared                          # shared frame buffer + status
from sprint3_main import run_detection             # detection engine

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

database.init_db()

# ─── Start detection in background thread ─────────────────────────────────────
# show_window=False → no OpenCV popup, frames go to shared buffer only
_detection_thread = threading.Thread(
    target=run_detection,
    args=(shared, False),
    daemon=True
)
_detection_thread.start()
print("[APP] Detection thread started — waiting for first frame…")


# ─── Login required decorator ─────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─── MJPEG stream generator ───────────────────────────────────────────────────
# Reads the latest annotated frame from shared buffer (written by detection thread)
# and yields it as a JPEG byte chunk.

def _placeholder_jpeg():
    """Return a 'Starting…' placeholder JPEG while detection warms up."""
    img = np.zeros((360, 640, 3), dtype="uint8")
    cv2.putText(img, "Starting detection…", (140, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 100), 2)
    cv2.putText(img, "Please wait", (220, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


_placeholder = _placeholder_jpeg()


def generate_mjpeg():
    """Generator — yields MJPEG frames consumed by <img src='/video_feed'>."""
    while True:
        jpeg = shared.get_jpeg(quality=75)
        if jpeg is None:
            jpeg = _placeholder
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.04)   # ~25 fps


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if (username == config.DASHBOARD_USERNAME and
                password == config.DASHBOARD_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    counts = database.get_event_counts()
    total  = sum(counts.values())
    recent = database.get_recent_events(limit=5)
    return render_template("dashboard.html",
                           counts=counts, total=total, recent=recent)


@app.route("/events")
@login_required
def events():
    filter_type = request.args.get("type", "all")
    rows   = (database.get_recent_events(limit=100) if filter_type == "all"
              else database.get_events_by_type(filter_type, limit=100))
    counts = database.get_event_counts()
    return render_template("events.html",
                           events=rows, filter_type=filter_type, counts=counts)


@app.route("/video_feed")
@login_required
def video_feed():
    """MJPEG stream — consumed by <img src='/video_feed'> in the browser."""
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot/<path:filename>")
@login_required
def snapshot(filename):
    # Use script directory so the path is correct regardless of where app.py is run from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    snap_dir = os.path.join(base_dir, config.SNAPSHOT_DIR)
    return send_from_directory(snap_dir, filename)


@app.route("/api/events")
@login_required
def api_events():
    filter_type = request.args.get("type", "all")
    limit       = int(request.args.get("limit", 50))
    rows = (database.get_recent_events(limit=limit) if filter_type == "all"
            else database.get_events_by_type(filter_type, limit=limit))
    return jsonify(rows)


@app.route("/api/stats")
@login_required
def api_stats():
    counts = database.get_event_counts()
    return jsonify({
        "counts":       counts,
        "total":        sum(counts.values()),
        "fps":          shared.fps,
        "status":       shared.status_text,
        "running":      shared.running,
        "motion_count": shared.motion_count,
        "audio_active": shared.audio_active,
        "high_conf":    shared.high_conf,
    })


@app.route("/status")
@login_required
def status():
    """Live system health page — reads health_log.csv for the recent entries table."""
    health_rows = []
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "health_log.csv")
    if os.path.isfile(log_path):
        try:
            with open(log_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for r in reversed(rows[-10:]):
                health_rows.append({
                    "timestamp": r.get("timestamp", ""),
                    "cpu":       r.get("cpu_%", "N/A"),
                    "mem_pct":   r.get("memory_%", "N/A"),
                    "mem_mb":    r.get("memory_used_mb", "N/A"),
                    "fps":       r.get("fps", "N/A"),
                    "total":     r.get("total_events", "N/A"),
                    "status":    r.get("status", "N/A"),
                })
        except Exception:
            pass
    return render_template("status.html", health_rows=health_rows)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[DASH] Dashboard → http://localhost:{config.DASHBOARD_PORT}")
    print(f"[DASH] Login: {config.DASHBOARD_USERNAME} / {config.DASHBOARD_PASSWORD}")
    app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False, threaded=True)
