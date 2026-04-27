# Changes After Sprint 2
## AI-Based Smart Home Security Monitoring System

---

## What Was in Sprint 2

Sprint 2 delivered the core detection engine (`sprint3_main.py`) with:
- Motion detection using OpenCV MOG2 background subtraction
- YOLO v8n object detection (person, car, bicycle, cat, dog)
- Audio anomaly detection using MFCC (librosa + sounddevice)
- Camera shake filter and temporal gate (5 consecutive frames)
- Motion + audio fusion logic (HIGH CONFIDENCE within 2 seconds)
- OpenCV popup window showing the live annotated feed

**Limitation after Sprint 2:** Everything ran in a single terminal window with no storage, no web interface, and no notifications.

---

## Sprint 3 — Database, Dashboard & Alerts

### 1. New file: `database.py`
Added SQLite event logging. Every detection is now saved permanently.

| Function | Purpose |
|---|---|
| `init_db()` | Creates the `events` table on first run |
| `log_event(...)` | Saves event type, label, confidence, snapshot path, bounding box |
| `get_recent_events(limit)` | Returns latest N events |
| `get_events_by_type(type, limit)` | Filter events by type |
| `get_event_counts()` | Returns count per event type (motion, audio, yolo, etc.) |

**Database schema:**
```
id, timestamp, event_type, label, confidence, snapshot_path, area, x, y, w, h
```

---

### 2. New file: `alerts.py`
Sends notifications when an event is detected.

- **Email alerts** — Gmail SMTP with snapshot image attached
- **Telegram alerts** — Bot API with snapshot photo
- **Cooldown system** — minimum 30 seconds between alerts to prevent spam
- Runs in a background thread so it never slows down detection

---

### 3. New file: `shared.py`
Thread-safe frame buffer that connects the detection engine to the Flask dashboard.

```
Detection thread  →  shared.set_frame()  →  Flask reads shared.get_jpeg()
```

Without this, both `app.py` and `sprint3_main.py` would try to open the camera at the same time and crash.

---

### 4. New file: `app.py` — Flask Web Dashboard
Replaced the OpenCV popup window with a browser-based dashboard.

**Pages:**
| Route | Page |
|---|---|
| `/` | Dashboard — live video feed + stat cards + recent events |
| `/events` | Full event log with type filter buttons + snapshot viewer |
| `/login` | Login page (session-based authentication) |
| `/video_feed` | MJPEG stream consumed by the browser |
| `/snapshot/<filename>` | Serves snapshot images |
| `/api/events` | JSON API for events |
| `/api/stats` | JSON API for live stats (FPS, flags, counts) |

**Features:**
- Login required for all pages (username/password set in `config.py`)
- Live MJPEG video stream at ~25 FPS
- Placeholder frame shown while YOLO is loading
- Bootstrap 5 dark theme
- Auto-refresh every 5 seconds
- Snapshot modal — click any event row to view the captured image

---

### 5. New templates
| File | Purpose |
|---|---|
| `templates/base.html` | Shared navbar, Bootstrap 5 dark theme, Font Awesome icons |
| `templates/login.html` | Login page with error message |
| `templates/dashboard.html` | Live feed + stat cards + recent events table |
| `templates/events.html` | Full event log + filter buttons + snapshot modal |

---

### 6. Changes to `sprint3_main.py`
| What changed | Why |
|---|---|
| All camera code moved inside `run_detection()` function | Previously ran at import time — crashed `app.py` on import |
| Added `shared_state` parameter | When `show_window=False`, writes frames to shared buffer instead of OpenCV window |
| Audio MFCC threshold raised from 8.0 → 25.0 | Silence baseline was ~22, causing constant false audio alerts |
| `save_snapshot()` uses `__file__`-relative path | Fixed snapshots saving to wrong folder depending on where script was run |
| Microseconds added to snapshot filenames (`%f`) | Fixed identical filenames when two events occurred in the same second |
| Calls `database.log_event()` on every confirmed event | Persists all detections to SQLite |
| Calls `alerts.send_alert()` on confirmed events | Triggers email/Telegram notifications |

---

### 7. New file: `config.py`
Central configuration for the entire system. Previously settings were hardcoded across files.

| Setting | Purpose |
|---|---|
| `DASHBOARD_USERNAME/PASSWORD` | Flask login credentials |
| `EMAIL_ENABLED`, `EMAIL_SENDER`, etc. | Gmail alert settings |
| `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram alert settings |
| `ALERT_COOLDOWN_SEC` | Minimum gap between alerts |
| `MIN_AREA`, `CONFIRM_FRAMES`, `YOLO_CONF`, etc. | Detection sensitivity |

---

## Sprint 4 — Performance, Monitoring & Testing

### 1. New file: `health_monitor.py`
Logs system health to `health_log.csv` every 60 seconds while the system runs.

**Columns logged:**
```
timestamp, cpu_%, memory_%, memory_used_mb, fps, status, motion_events,
audio_events, yolo_events, high_conf_events, total_events
```

- Starts automatically as a background thread when `app.py` runs
- Can also run standalone: `python health_monitor.py`
- Open `health_log.csv` in Excel to review performance over time

---

### 2. New file: `stability_test.py`
Runs the full system for an extended period and monitors for crashes.

- Default: 48 hours (`--hours 48`), quick test: `--hours 1`
- Checks detection thread health every 30 seconds
- Auto-restarts the detection thread if it dies
- Logs all events (start, crash, restart, progress) to `stability_log.txt`
- Prints PASS (0 restarts) or WARNING (N restarts) at the end

---

### 3. New file: `testing_report.py`
Reads all events from the database and generates `testing_report.txt`.

**Report sections:**
1. Event detection summary (counts per type)
2. YOLO object class breakdown (person, car, etc. with percentages)
3. YOLO confidence statistics (avg, min, max)
4. Snapshot coverage (% of events with image saved)
5. Session timeline (first event, last event, duration, event rate)
6. Last 10 events sample
7. Success criteria evaluation (target: ≥85% detections at ≥85% confidence)

---

### 4. New page: `templates/status.html` — Health Page
Added a live **Health** tab to the dashboard navbar.

**Shows in real time (auto-refreshes every 3 seconds):**
- Detection status text
- Current FPS
- CPU usage %
- Memory usage %
- Motion / Audio / High Confidence active flags
- Event breakdown by type
- Component status (camera, YOLO, audio, database, Flask)
- Table of last 10 entries from `health_log.csv`

---

### 5. Changes to `app.py`
| What changed | Why |
|---|---|
| Added `import csv` | Needed to read `health_log.csv` for the status page |
| Added `/status` route | Serves the new Health page |
| `/status` reads `health_log.csv` | Passes last 10 rows to the template |

---

### 6. Changes to `templates/base.html`
Added **Health** link to the navbar between Events and Logout.

---

### 7. Security fix: `.env` file
Telegram credentials moved out of `config.py` into a `.env` file.

- `.env` is listed in `.gitignore` — never pushed to GitHub
- `config.py` reads credentials with `os.getenv()` via `python-dotenv`
- Prevents bot token from being exposed in the public repository

---

### 8. New files: `USER_MANUAL.md` and `Software_Architecture.md`
- `USER_MANUAL.md` — full setup guide, configuration, usage, troubleshooting
- `Software_Architecture.md` — system architecture with diagrams

---

## Summary of All New Files

| File | Sprint | Purpose |
|---|---|---|
| `database.py` | 3 | SQLite event logging |
| `alerts.py` | 3 | Email + Telegram notifications |
| `shared.py` | 3 | Thread-safe frame buffer |
| `app.py` | 3 | Flask web dashboard |
| `config.py` | 3 | Central configuration |
| `templates/base.html` | 3 | Shared layout |
| `templates/login.html` | 3 | Login page |
| `templates/dashboard.html` | 3 | Main dashboard |
| `templates/events.html` | 3 | Event log + snapshot viewer |
| `health_monitor.py` | 4 | CPU/memory/FPS logger |
| `stability_test.py` | 4 | 48-hour stability test runner |
| `testing_report.py` | 4 | Accuracy & performance report |
| `templates/status.html` | 4 | Live health page |
| `USER_MANUAL.md` | 4 | Setup & usage guide |
| `Software_Architecture.md` | 4 | Architecture document |
| `.env` | 4 | Secret credentials (not in GitHub) |

---

*AI-Based Smart Home Security Monitoring System*
*Sprint 3 & 4 Changes Document*
