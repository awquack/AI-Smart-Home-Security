# Software Architecture Document
## AI-Based Smart Home Security Monitoring System

**Version:** Sprint 3
**Date:** March 2026
**Team:** 2 Students — AI Engineer + Backend & Integration Engineer

---

## 1. System Overview

The AI-Based Smart Home Security Monitoring System is a **locally deployed**, real-time security solution that combines computer vision, audio anomaly detection, and a web-based dashboard. The system detects intrusions using a camera and microphone, logs all events to a local database, sends alerts via Email/Telegram, and streams a live annotated feed to a browser-based dashboard.

**Key design principle:** Everything runs on a single local machine — no cloud, no external servers, no paid services.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL MACHINE                            │
│                                                                 │
│  ┌──────────┐    ┌──────────┐                                   │
│  │  Camera  │    │   Mic    │                                   │
│  └────┬─────┘    └────┬─────┘                                   │
│       │               │                                         │
│       ▼               ▼                                         │
│  ┌─────────────────────────────────────┐                        │
│  │         sprint3_main.py             │                        │
│  │         (Detection Engine)          │                        │
│  │                                     │                        │
│  │  ┌─────────┐  ┌──────┐  ┌───────┐  │                        │
│  │  │  MOG2   │  │ YOLO │  │ MFCC  │  │                        │
│  │  │ Motion  │  │  v8n │  │ Audio │  │                        │
│  │  └────┬────┘  └──┬───┘  └───┬───┘  │                        │
│  │       └──────────┴──────────┘       │                        │
│  │                  │                  │                        │
│  │                  ▼                  │                        │
│  │         Fusion Logic                │                        │
│  │    (motion + audio = HIGH CONF)     │                        │
│  └──────┬──────────────────┬───────────┘                        │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │  shared.py  │    │ database.py │                             │
│  │ (Frame buf) │    │  (SQLite)   │                             │
│  └──────┬──────┘    └──────┬──────┘                             │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────────────────────────────┐    ┌───────────────┐   │
│  │           app.py (Flask)            │    │  alerts.py    │   │
│  │                                     │───►│  Email        │   │
│  │  /video_feed  /events  /api/stats   │    │  Telegram     │   │
│  └─────────────────────────────────────┘    └───────────────┘   │
│                    │                                             │
└────────────────────┼─────────────────────────────────────────────┘
                     │ HTTP
                     ▼
              ┌─────────────┐
              │   Browser   │
              │ localhost:  │
              │    5000     │
              └─────────────┘
```

---

## 3. Component Architecture

### 3.1 Detection Engine — `sprint3_main.py`

The core AI processing module. Runs as a **background thread** when launched via `app.py`.

| Sub-Component | Technology | Description |
|---|---|---|
| Motion Detection | OpenCV MOG2 | Background subtraction to detect moving regions |
| Object Detection | YOLOv8n (Ultralytics) | Classifies detected objects (person, car, bicycle, cat, dog) |
| Audio Detection | MFCC + sounddevice + librosa | Detects anomalous sounds via microphone |
| Shake Filter | OpenCV | Ignores detections when >20% of pixels move (camera wobble) |
| Temporal Gate | Counter logic | Motion must appear in 5+ consecutive frames before alerting |
| Fusion Logic | Time-window check | Motion + audio within 2s = HIGH CONFIDENCE alert |
| Snapshot Saver | OpenCV imwrite | Auto-saves JPEG on every confirmed event |

**Detection flow:**
```
Camera Frame
    │
    ▼
MOG2 Background Subtraction
    │
    ▼
Shake Filter (skip if >20% pixels moving)
    │
    ▼
Contour Detection (min area: 20,000 px²)
    │
    ▼
Temporal Confirmation (5 consecutive frames)
    │
    ▼
YOLO Object Classification (every 5 frames)
    │
    ▼
Fusion Check (motion + audio within 2s?)
    │
    ▼
Log to DB + Send Alert + Save Snapshot + Push to shared buffer
```

---

### 3.2 Shared State — `shared.py`

Thread-safe bridge between the detection engine and the Flask dashboard.

```
Detection Thread          SharedState            Flask Thread
─────────────────         ───────────            ────────────
set_frame(frame)  ──────► _frame (lock)  ──────► get_jpeg()
motion_count      ──────► motion_count   ──────► api_stats()
status_text       ──────► status_text    ──────► api_stats()
audio_active      ──────► audio_active   ──────► api_stats()
high_conf         ──────► high_conf      ──────► api_stats()
fps               ──────► fps            ──────► api_stats()
```

Uses `threading.Lock()` to prevent race conditions between threads.

---

### 3.3 Web Dashboard — `app.py` + `templates/`

Flask-based local web server. Reads from `shared.py` and `database.py`.

| Route | Method | Description |
|---|---|---|
| `/login` | GET/POST | Session-based authentication |
| `/logout` | GET | Clears session, redirects to login |
| `/` | GET | Dashboard — live feed + stat cards |
| `/events` | GET | Event log table with type filter |
| `/video_feed` | GET | MJPEG stream from shared frame buffer |
| `/snapshot/<file>` | GET | Serves saved snapshot images |
| `/api/events` | GET | JSON — latest events (auto-refresh) |
| `/api/stats` | GET | JSON — live stats (motion count, FPS, status) |

**MJPEG Streaming:**
```
Detection Thread                    Browser
─────────────────                   ───────
Annotated frame                     <img src="/video_feed">
    │                                        │
    ▼                                        │
shared.set_frame()                           │
    │                                        │
    ▼                                        │
generate_mjpeg()  ──── HTTP chunks ─────────►│
(multipart/x-mixed-replace)          updates ~25fps
```

**Frontend pages:**

| Template | Technology | Description |
|---|---|---|
| `base.html` | Bootstrap 5 + Font Awesome | Shared navbar, layout, dark theme |
| `login.html` | Bootstrap 5 forms | Login with error feedback |
| `dashboard.html` | Bootstrap 5 + JS fetch() | Live feed + stat cards, auto-refresh 5s |
| `events.html` | Bootstrap 5 modal + JS | Event table, type filter, snapshot viewer |

---

### 3.4 Database — `database.py`

SQLite database for persistent event storage.

**Table: `events`**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | TEXT | `YYYY-MM-DD HH:MM:SS` |
| `event_type` | TEXT | `motion` / `audio` / `yolo` / `high_confidence` |
| `label` | TEXT | YOLO class name or audio description |
| `confidence` | REAL | YOLO confidence score (0.0–1.0) |
| `snapshot_path` | TEXT | Path to saved JPEG file |
| `area` | INTEGER | Motion area in pixels² |
| `x, y, w, h` | INTEGER | Bounding box coordinates |

**Functions:**

| Function | Description |
|---|---|
| `init_db()` | Creates table if not exists |
| `log_event(...)` | Inserts one event row |
| `get_recent_events(limit)` | Returns latest N events |
| `get_events_by_type(type, limit)` | Returns events filtered by type |
| `get_event_counts()` | Returns count per event type |

---

### 3.5 Alert System — `alerts.py`

Non-blocking alert dispatcher. Runs in a background thread to avoid slowing detection.

```
Detection Event
      │
      ▼
alerts.send_alert(event_type, label, snapshot_path)
      │
      ├── Cooldown check (30s minimum between alerts)
      │
      ├── HIGH_CONF_ONLY filter (optional)
      │
      └── Background Thread
               │
               ├── Email (smtplib + Gmail SMTP)
               │     └── Attaches snapshot image
               │
               └── Telegram (requests + Bot API)
                     └── Sends photo with caption
```

---

### 3.6 Configuration — `config.py`

Single file for all system settings.

| Category | Settings |
|---|---|
| Dashboard | `SECRET_KEY`, `DASHBOARD_USERNAME`, `DASHBOARD_PASSWORD`, `DASHBOARD_PORT` |
| Email | `EMAIL_ENABLED`, `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECEIVER` |
| Telegram | `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Alerts | `ALERT_COOLDOWN_SEC`, `HIGH_CONF_ONLY` |
| Detection | `MIN_AREA`, `CONFIRM_FRAMES`, `SHAKE_RATIO_MAX`, `YOLO_EVERY`, `FUSION_WINDOW` |
| Storage | `DB_PATH`, `SNAPSHOT_DIR`, `AUTO_SNAPSHOT` |

---

## 4. Threading Model

```
python app.py (main process)
│
├── Thread 1: Detection Engine  (sprint3_main.run_detection)
│     ├── Camera capture loop
│     ├── MOG2 + YOLO + Fusion logic
│     └── Writes to: shared, database, snapshots, alerts
│
├── Thread 2: Flask Server  (app.run threaded=True)
│     ├── Serves HTTP requests
│     ├── Reads from: shared, database
│     └── Streams MJPEG to browser
│
└── Thread 3: MFCC Audio Worker  (daemon)
      ├── Reads microphone via sounddevice
      └── Writes to: database, alerts, shared.audio_active
```

---

## 5. Data Flow

```
Input Sources          Processing              Output
─────────────          ──────────              ──────
Camera       ────────► MOG2 Motion ──────────► SQLite DB
                  │    YOLO Object ──────────► snapshots/
                  │    Fusion      ──────────► shared buffer ──► Browser
                  │                                          (MJPEG stream)
Microphone   ────────► MFCC Audio  ──────────► SQLite DB
                                   ──────────► Email / Telegram alerts
```

---

## 6. File Structure

```
AI-Smart-Home-Security/
│
├── app.py                  ← Flask server + MJPEG stream + routes
├── sprint3_main.py         ← Detection engine (MOG2 + YOLO + MFCC)
├── shared.py               ← Thread-safe frame buffer + status
├── database.py             ← SQLite event logging
├── alerts.py               ← Email + Telegram notifications
├── config.py               ← All credentials and thresholds
│
├── templates/
│   ├── base.html           ← Shared layout (Bootstrap 5 dark theme)
│   ├── login.html          ← Login page
│   ├── dashboard.html      ← Live feed + stat cards
│   └── events.html         ← Event log table + snapshot viewer
│
├── requirements.txt        ← Python dependencies
├── yolov8n.pt              ← YOLOv8 nano model weights
├── security_events.db      ← SQLite database (auto-created at runtime)
└── snapshots/              ← Saved event images (auto-created at runtime)
```

---

## 7. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.13 | Core runtime |
| Computer Vision | OpenCV | ≥4.8 | Camera, motion detection, frame drawing |
| Object Detection | Ultralytics YOLOv8n | ≥8.0 | Person/vehicle detection |
| Audio | sounddevice + librosa | ≥0.4 / ≥0.10 | Mic input + MFCC features |
| Web Framework | Flask | ≥3.0 | Dashboard server + routing |
| Templating | Jinja2 | Built-in | HTML template rendering |
| Frontend | Bootstrap 5 + Font Awesome | CDN | Responsive dark UI |
| Database | SQLite3 | Built-in | Local event storage |
| Alerts | smtplib + requests | Built-in / ≥2.31 | Email + Telegram |
| Math | NumPy | ≥1.24 | Array operations |
| Concurrency | threading | Built-in | Multi-thread architecture |

---

## 8. Security Design

| Feature | Implementation |
|---|---|
| Dashboard access | Session-based login (Flask sessions + SECRET_KEY) |
| Local deployment | No cloud — all data stays on the machine |
| Snapshot storage | Only event-triggered snapshots saved (not continuous recording) |
| Credentials | Stored in `config.py` — not hardcoded in logic files |
| Alert spam prevention | 30-second cooldown between external alerts |

---

## 9. Performance Targets

| Metric | Target | Implementation |
|---|---|---|
| Detection accuracy | ≥85% | YOLO confidence threshold: 35%, temporal gate: 5 frames |
| Alert latency | <5 seconds | Background thread dispatch, no blocking |
| Live feed frame rate | ~25 FPS | MJPEG stream with 40ms sleep |
| False alarm reduction | Camera shake filter | Skip if >20% pixels moving |
| System stability | 48 hours continuous | Daemon threads, graceful error handling |

---

## 10. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the system
python app.py

# 3. Open dashboard
http://localhost:5000
# Login: admin / admin123
```

**Stop:** `Ctrl + C`
