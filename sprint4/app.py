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
import datetime
import numpy as np
from functools import wraps
from flask import (Flask, Response, render_template, redirect,
                   url_for, request, session, jsonify, send_from_directory)

import config
import database
import alerts
from shared import shared                          # shared frame buffer + status
from sprint3_main import run_detection             # detection engine

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

database.init_db()

# ─── Daily summary + disk cleanup scheduler ───────────────────────────────────
def _cleanup_old_files():
    """Delete clips and snapshots older than CLEANUP_MAX_DAYS."""
    if not config.CLEANUP_ENABLED:
        return
    cutoff   = time.time() - config.CLEANUP_MAX_DAYS * 86400
    base_dir = os.path.dirname(os.path.abspath(__file__))
    removed  = 0
    for folder in [config.CLIP_DIR, config.SNAPSHOT_DIR]:
        dirpath = os.path.join(base_dir, folder)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                removed += 1
    if removed:
        print(f"[CLEANUP] Removed {removed} old file(s) older than {config.CLEANUP_MAX_DAYS} days")


def _daily_summary_scheduler():
    """Sleeps until midnight, sends a daily summary, runs cleanup, then repeats."""
    while True:
        now       = datetime.datetime.now()
        midnight  = (now + datetime.timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
        time.sleep((midnight - now).total_seconds())
        counts = database.get_today_counts()
        alerts.send_daily_summary(counts)
        _cleanup_old_files()

threading.Thread(target=_daily_summary_scheduler, daemon=True).start()


# ─── Start detection in background thread (with auto-restart) ─────────────────
def _detection_watchdog():
    """Runs detection in a loop — restarts automatically if it crashes."""
    while True:
        print("[APP] Detection thread started — waiting for first frame…")
        try:
            run_detection(shared, False)
        except Exception as e:
            print(f"[APP] Detection crashed: {e}")
        if not shared.running:
            break   # clean stop requested — don't restart
        print("[APP] Detection thread ended unexpectedly — restarting in 3s…")
        shared.status_text = "Restarting…"
        time.sleep(3)

threading.Thread(target=_detection_watchdog, daemon=True).start()


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
        time.sleep(0.016)  # ~60 fps


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


@app.route("/api/chart_data")
@login_required
def api_chart_data():
    return jsonify({
        "hourly": database.get_hourly_counts(),
        "daily":  database.get_daily_counts(),
    })


@app.route("/clips")
@login_required
def clips():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    clips_dir = os.path.join(base_dir, "clips")
    files = []
    if os.path.isdir(clips_dir):
        for f in sorted(os.listdir(clips_dir), reverse=True):
            if f.endswith(".mp4"):
                size = os.path.getsize(os.path.join(clips_dir, f))
                files.append({"name": f, "size_mb": round(size / 1024 / 1024, 1)})
    return render_template("clips.html", clips=files)


@app.route("/clip/<path:filename>")
@login_required
def serve_clip(filename):
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    clips_dir = os.path.join(base_dir, "clips")
    return send_from_directory(clips_dir, filename)


@app.route("/faces", methods=["GET"])
@login_required
def faces():
    known = database.get_known_faces()
    return render_template("faces.html", faces=known)


@app.route("/faces/register", methods=["POST"])
@login_required
def register_face():
    import face_recognition_module as face_mod
    name  = request.form.get("name", "").strip()
    photo = request.files.get("photo")
    if not name or not photo:
        return redirect(url_for("faces"))

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(base_dir, config.FACE_PHOTOS_DIR)
    os.makedirs(photos_dir, exist_ok=True)
    photo_path = os.path.join(photos_dir, f"{name}_{photo.filename}")
    photo.save(photo_path)

    encodings = face_mod.encode_face_from_path(photo_path)
    if not encodings:
        return render_template("faces.html",
                               faces=database.get_known_faces(),
                               error=f"No face detected in photo for '{name}'. Try a clearer image.")
    database.add_known_face(name, encodings[0].tolist())
    print(f"[FACE] Registered: {name}")
    return redirect(url_for("faces"))


@app.route("/faces/delete/<int:face_id>", methods=["POST"])
@login_required
def delete_face(face_id):
    database.delete_known_face(face_id)
    return redirect(url_for("faces"))


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
    if config.HTTPS_ENABLED and os.path.isfile(config.CERT_FILE) and os.path.isfile(config.KEY_FILE):
        protocol = "https"
        ssl_ctx  = (config.CERT_FILE, config.KEY_FILE)
    else:
        protocol = "http"
        ssl_ctx  = None
        if config.HTTPS_ENABLED:
            print("[DASH] WARNING: cert.pem/key.pem not found — run: python generate_cert.py")

    print(f"[DASH] Dashboard → {protocol}://localhost:{config.DASHBOARD_PORT}")
    print(f"[DASH] Login: {config.DASHBOARD_USERNAME} / {config.DASHBOARD_PASSWORD}")
    app.run(host="0.0.0.0", port=config.DASHBOARD_PORT,
            debug=False, threaded=True, ssl_context=ssl_ctx)
