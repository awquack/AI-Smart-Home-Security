# USER MANUAL
## AI-Based Smart Home Security Monitoring System

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running the System](#5-running-the-system)
6. [Using the Dashboard](#6-using-the-dashboard)
7. [Setting Up Alerts](#7-setting-up-alerts)
8. [Sprint 4 Tools](#8-sprint-4-tools)
9. [Stopping the System](#9-stopping-the-system)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

This system monitors your home using a webcam and microphone. It detects:

| Detection Type | How it works |
|---|---|
| **Motion** | OpenCV MOG2 background subtraction |
| **Objects** | YOLOv8n — recognises person, car, bicycle, cat, dog |
| **Audio anomalies** | MFCC audio fingerprinting via librosa |
| **High confidence** | Motion + audio detected within 2 seconds |

All events are logged to a local SQLite database and viewable on a web dashboard at `http://localhost:5000`.

---

## 2. Requirements

### Hardware
- PC or laptop running Windows 10/11
- Webcam (built-in or USB)
- Microphone (built-in or USB)

### Software
- Python 3.9 or newer
- pip (comes with Python)

---

## 3. Installation

### Step 1 — Clone or download the project

```
git clone https://github.com/awquack/AI-Smart-Home-Security
cd AI-Smart-Home-Security
```

### Step 2 — Install dependencies

```
pip install -r requirements.txt
```

This installs:
- `flask` — web dashboard
- `opencv-python` — camera and motion detection
- `ultralytics` — YOLOv8 object detection
- `torch torchvision` — deep learning backend
- `librosa sounddevice soundfile` — audio processing
- `psutil` — CPU/memory monitoring (Sprint 4)
- `requests` — Telegram alerts

> **Note:** The first run will download the YOLOv8n model weights (~6 MB) automatically.

---

## 4. Configuration

Open `config.py` and adjust the settings:

### Camera
```python
CAMERA_INDEX = 0        # 0 = default webcam, change to 1 for external camera
```

### Dashboard login
```python
DASHBOARD_USERNAME = "admin"
DASHBOARD_PASSWORD = "admin123"
DASHBOARD_PORT     = 5000
```

### Detection sensitivity
```python
MIN_AREA       = 20000   # minimum pixel area to count as motion (lower = more sensitive)
CONFIRM_FRAMES = 5       # consecutive frames needed before alert fires
YOLO_CONF      = 0.45    # YOLO confidence threshold (0.0–1.0)
```

### Alerts (optional — see Section 7)
```python
EMAIL_ENABLED    = False
TELEGRAM_ENABLED = False
ALERT_COOLDOWN_SEC = 30  # minimum seconds between repeated alerts
```

---

## 5. Running the System

### Normal operation (dashboard + detection)

```
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

Login with the credentials set in `config.py` (default: `admin` / `admin123`).

### Detection only (no browser, shows OpenCV window)

```
python sprint3_main.py
```

Press `Q` in the OpenCV window to quit.

---

## 6. Using the Dashboard

### Dashboard page (`/`)
- Live video feed from your camera with bounding boxes drawn around detected objects
- 4 stat cards: Motion, Audio, YOLO, and High Confidence event counts
- Table of the 5 most recent events

### Events page (`/events`)
- Full event log (up to 100 events)
- Filter buttons: All / Motion / Audio / YOLO / High Confidence / Manual Snapshot
- Click any row that has a snapshot icon (📷) to view the captured image

### Health page (`/health`)
- Live system metrics: FPS, CPU %, Memory %
- Detection flags: whether motion/audio/high-confidence is currently active
- Event breakdown by type
- Table of recent health log entries (updated every 60 seconds)

### API endpoints (for developers)
| Endpoint | Description |
|---|---|
| `GET /api/stats` | JSON: FPS, status, event counts, detection flags |
| `GET /api/events` | JSON: recent events (use `?type=motion&limit=20`) |

---

## 7. Setting Up Alerts

### Email alerts (Gmail)

1. Enable 2-Factor Authentication on your Gmail account
2. Go to Google Account → Security → App Passwords → create one
3. Edit `config.py`:

```python
EMAIL_ENABLED  = True
EMAIL_SENDER   = "your.email@gmail.com"
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"   # 16-character App Password
EMAIL_RECEIVER = "receiver@example.com"
```

### Telegram alerts

1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Message `@userinfobot` on Telegram → copy your Chat ID
3. Edit `config.py`:

```python
TELEGRAM_ENABLED     = True
TELEGRAM_BOT_TOKEN   = "123456789:ABCdefGHI..."
TELEGRAM_CHAT_ID     = "987654321"
```

Alerts fire at most once every `ALERT_COOLDOWN_SEC` seconds (default 30).

---

## 8. Sprint 4 Tools

### Health monitor

Logs CPU, memory, FPS, and event counts to `health_log.csv` every 60 seconds.
It starts automatically when you run `app.py`. To run standalone:

```
python health_monitor.py
```

Open `health_log.csv` in Excel to review system performance over time.

### Stability test

Runs the full system for 48 hours, auto-restarts the detection thread if it crashes, and saves a log to `stability_log.txt`.

```
python stability_test.py              # 48 hours
python stability_test.py --hours 1   # 1 hour (quick test)
```

### Performance / testing report

Reads all events from the database and saves a detailed accuracy report to `testing_report.txt`.

```
python testing_report.py
```

The report includes:
- Event detection summary
- YOLO class breakdown with confidence statistics
- Snapshot coverage
- Session timeline and event rate
- Success criteria evaluation (target: ≥85% detections at ≥85% confidence)

---

## 9. Stopping the System

In the terminal where `app.py` is running, press:

```
Ctrl + C
```

The system will stop cleanly. All events already logged to the database are preserved.

---

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| **Camera not found** | Change `CAMERA_INDEX` in `config.py` to `1` or `2` |
| **Too many false motion alerts** | Increase `MIN_AREA` in `config.py` (e.g. 30000) |
| **Too many audio alerts** | Increase `_MFCC_DIST` in `sprint3_main.py` (currently 25.0) |
| **Dashboard shows "Starting detection…"** | Wait 5–10 seconds for YOLO to load |
| **Email alerts not sending** | Check App Password is correct; check `EMAIL_ENABLED = True` |
| **`ModuleNotFoundError`** | Run `pip install -r requirements.txt` again |
| **`No module named psutil`** | Run `pip install psutil` |
| **Port 5000 already in use** | Change `DASHBOARD_PORT` in `config.py` to e.g. `5001` |
| **Snapshot images not loading** | Make sure you are running `app.py` from inside the project folder |

---

*AI-Based Smart Home Security Monitoring System — Sprint 4*
