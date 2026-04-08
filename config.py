# config.py – Central configuration for AI Smart Home Security

import os
from dotenv import load_dotenv
load_dotenv()   # reads .env file into environment variables

# ─── Dashboard (Flask) ────────────────────────────────────────────────────────
SECRET_KEY           = "change_this_to_a_random_string"  # used to sign Flask sessions
DASHBOARD_USERNAME   = "admin"
DASHBOARD_PASSWORD   = "admin123"   # change before demo
DASHBOARD_PORT       = 5001

# Fill in your credentials before running sprint3_main.py

# ─── Email Alerts ──────────────────────────────────────────────────────────────
EMAIL_ENABLED   = False               # Set True after filling credentials
EMAIL_SENDER    = "your_email@gmail.com"
EMAIL_PASSWORD  = "your_app_password" # Use a Gmail App Password (not your main password)
EMAIL_RECEIVER  = "receiver@email.com"
EMAIL_SMTP_HOST = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# ─── Telegram Alerts ──────────────────────────────────────────────────────────
TELEGRAM_ENABLED   = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Alert Behavior ───────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC  = 30   # Minimum seconds between alerts (avoids spam)
HIGH_CONF_ONLY      = False # If True, only send alerts on HIGH CONFIDENCE events

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = "security_events.db"

# ─── Snapshots ────────────────────────────────────────────────────────────────
SNAPSHOT_DIR        = "snapshots"
AUTO_SNAPSHOT       = True   # Automatically save snapshot on every confirmed event

# ─── Face Recognition ─────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"   # folder for uploaded enrollment photos
FACE_TOLERANCE  = 0.50            # lower = stricter matching (0.4–0.6 recommended)

# ─── Detection Thresholds (override sprint2 defaults here) ────────────────────
MIN_AREA         = 20000
BLUR_SIZE        = 21
MERGE_DISTANCE   = 150
MAX_BOXES        = 2
PERSIST_TIME     = 1.5
ALERT_COOLDOWN   = 3.0
SHAKE_RATIO_MAX  = 0.20
CONFIRM_FRAMES   = 5
YOLO_EVERY       = 5
FUSION_WINDOW    = 2.0
