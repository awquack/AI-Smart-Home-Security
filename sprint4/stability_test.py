# stability_test.py – 48-hour continuous stability test
#
# Runs the full system (detection + dashboard) and:
#   - Monitors for crashes every 30 seconds
#   - Auto-restarts detection thread if it dies
#   - Logs all events (start, crash, restart) to stability_log.txt
#   - Prints a summary report at the end
#
# Usage:
#   python stability_test.py              ← runs for 48 hours
#   python stability_test.py --hours 1    ← runs for 1 hour (quick test)

import sys
import time
import threading
import datetime
import argparse
import os

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=float, default=48,
                    help="Duration in hours (default: 48)")
args = parser.parse_args()

TEST_DURATION_SEC = args.hours * 3600
LOG_FILE          = "stability_log.txt"
CHECK_INTERVAL    = 30   # check thread health every 30 seconds


# ── Logger ────────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Import system modules ─────────────────────────────────────────────────────
log("=" * 60)
log(f"STABILITY TEST START — duration: {args.hours:.1f} hours")
log("=" * 60)

from shared import shared
from sprint3_main import run_detection
from health_monitor import start_monitor
import database

# Start health monitor (logs CPU/mem/FPS every 60s)
start_monitor(shared_state=shared, interval=60)
log("[INIT] Health monitor started")


# ── Detection thread manager ──────────────────────────────────────────────────
_detection_thread = None
restart_count     = 0


def start_detection_thread():
    global _detection_thread
    t = threading.Thread(
        target=run_detection,
        args=(shared, False),
        daemon=True,
        name="DetectionThread"
    )
    t.start()
    _detection_thread = t
    log("[DETECTION] Thread started")
    return t


def is_detection_alive():
    return _detection_thread is not None and _detection_thread.is_alive()


# ── Start Flask in background ─────────────────────────────────────────────────
import config
from app import app

def start_flask():
    app.run(host="0.0.0.0", port=config.DASHBOARD_PORT,
            debug=False, threaded=True, use_reloader=False)

flask_thread = threading.Thread(target=start_flask, daemon=True, name="FlaskThread")
flask_thread.start()
log(f"[FLASK] Dashboard started on http://localhost:{config.DASHBOARD_PORT}")

# Start detection
start_detection_thread()

# ── Main stability loop ───────────────────────────────────────────────────────
start_time = time.time()
last_event_count = 0

log("[TEST] Monitoring started — press Ctrl+C to stop early\n")

try:
    while True:
        elapsed   = time.time() - start_time
        remaining = TEST_DURATION_SEC - elapsed

        if remaining <= 0:
            log("=" * 60)
            log("TEST DURATION REACHED — stopping normally")
            break

        time.sleep(CHECK_INTERVAL)

        # ── Check detection thread health ────────────────────────────────────
        if not is_detection_alive():
            restart_count += 1
            log(f"[CRASH] Detection thread died! Restart #{restart_count}")
            time.sleep(2)
            start_detection_thread()

        # ── Log progress every 10 minutes ────────────────────────────────────
        elapsed_min = int(elapsed / 60)
        if elapsed_min % 10 == 0 and elapsed_min > 0:
            counts       = database.get_event_counts()
            total        = sum(counts.values())
            hours_done   = elapsed / 3600
            hours_left   = remaining / 3600
            log(f"[PROGRESS] {hours_done:.1f}h done | {hours_left:.1f}h left | "
                f"Events: {total} | FPS: {shared.fps} | "
                f"Restarts: {restart_count}")

except KeyboardInterrupt:
    log("\n[TEST] Stopped manually by user")

# ── Final report ──────────────────────────────────────────────────────────────
elapsed     = time.time() - start_time
counts      = database.get_event_counts()
total       = sum(counts.values())

log("\n" + "=" * 60)
log("  STABILITY TEST REPORT")
log("=" * 60)
log(f"  Duration run     : {elapsed/3600:.2f} hours  ({elapsed/60:.1f} minutes)")
log(f"  Target duration  : {args.hours} hours")
log(f"  Thread restarts  : {restart_count}  {'✓ STABLE' if restart_count == 0 else '✗ UNSTABLE'}")
log(f"  Total events     : {total}")
log(f"  Event breakdown  : {counts}")
log(f"  Final FPS        : {shared.fps}")
log(f"  Log file         : {LOG_FILE}")
log(f"  Health log       : health_log.csv")
log("=" * 60)

if restart_count == 0:
    log("  RESULT: PASSED — system ran stably for full duration")
else:
    log(f"  RESULT: WARNING — {restart_count} restart(s) occurred")
log("=" * 60)
