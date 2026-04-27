# health_monitor.py – System health logger for 48-hour stability test
#
# Logs every 60 seconds:
#   - CPU usage %
#   - Memory usage %
#   - FPS from shared state
#   - Total DB events
#   - Detection status
#
# Output: health_log.csv  (open in Excel to review after test)
#
# Usage (runs alongside app.py automatically, or standalone):
#   python health_monitor.py

import csv
import os
import time
import datetime
import threading

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    print("[HEALTH] psutil not found — install with: pip install psutil")

LOG_FILE     = "health_log.csv"
LOG_INTERVAL = 60   # seconds between each log entry


def _write_header(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "cpu_%", "memory_%", "memory_used_mb",
            "fps", "status", "motion_events", "audio_events",
            "yolo_events", "high_conf_events", "total_events"
        ])


def _log_entry(shared_state=None):
    """Collect one snapshot of system health and write to CSV."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CPU + memory
    if PSUTIL_OK:
        cpu     = psutil.cpu_percent(interval=1)
        mem     = psutil.virtual_memory()
        mem_pct = mem.percent
        mem_mb  = round(mem.used / 1024 / 1024, 1)
    else:
        cpu, mem_pct, mem_mb = "N/A", "N/A", "N/A"

    # FPS + status from shared state
    fps    = shared_state.fps    if shared_state else "N/A"
    status = shared_state.status_text if shared_state else "N/A"

    # DB event counts
    try:
        from database import get_event_counts
        counts = get_event_counts()
        motion    = counts.get("motion", 0)
        audio     = counts.get("audio", 0)
        yolo      = counts.get("yolo", 0)
        high_conf = counts.get("high_confidence", 0)
        total     = sum(counts.values())
    except Exception:
        motion = audio = yolo = high_conf = total = "N/A"

    row = [ts, cpu, mem_pct, mem_mb, fps, status,
           motion, audio, yolo, high_conf, total]

    # Append to CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            _write_header(LOG_FILE)
        writer.writerow(row)

    print(f"[HEALTH] {ts}  CPU:{cpu}%  MEM:{mem_pct}%  FPS:{fps}  "
          f"Events:{total}  Status:{status}")


def start_monitor(shared_state=None, interval=LOG_INTERVAL):
    """
    Start health logging in a background daemon thread.
    Called automatically by app.py.
    """
    if not os.path.isfile(LOG_FILE):
        _write_header(LOG_FILE)
        print(f"[HEALTH] Logging to {LOG_FILE} every {interval}s")

    def _loop():
        while True:
            try:
                _log_entry(shared_state)
            except Exception as e:
                print(f"[HEALTH] Log error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[HEALTH] Running standalone health monitor (no shared state)")
    print(f"[HEALTH] Logging to {LOG_FILE} every {LOG_INTERVAL}s")
    print("[HEALTH] Press Ctrl+C to stop\n")

    if not os.path.isfile(LOG_FILE):
        _write_header(LOG_FILE)

    try:
        while True:
            _log_entry(shared_state=None)
            time.sleep(LOG_INTERVAL)
    except KeyboardInterrupt:
        print("\n[HEALTH] Monitor stopped.")
