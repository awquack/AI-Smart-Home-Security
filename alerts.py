# alerts.py – Email and Telegram alert notifications
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import config

_last_alert_time = 0.0
_lock = threading.Lock()


def _can_fire() -> bool:
    """Enforce global alert cooldown to prevent spam."""
    global _last_alert_time
    with _lock:
        now = time.time()
        if now - _last_alert_time >= config.ALERT_COOLDOWN_SEC:
            _last_alert_time = now
            return True
        return False


# ─── Email ────────────────────────────────────────────────────────────────────

def _send_email(subject: str, body: str, snapshot_path: str | None):
    try:
        msg = MIMEMultipart()
        msg["From"]    = config.EMAIL_SENDER
        msg["To"]      = config.EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if snapshot_path:
            try:
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header(
                        "Content-Disposition", "attachment",
                        filename=snapshot_path.split("/")[-1]
                    )
                    msg.attach(img)
            except Exception as e:
                print(f"[ALERT] Could not attach snapshot: {e}")

        with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            s.sendmail(config.EMAIL_SENDER, config.EMAIL_RECEIVER, msg.as_string())

        print(f"[ALERT] Email sent → {config.EMAIL_RECEIVER}")

    except Exception as e:
        print(f"[ALERT] Email failed: {e}")


# ─── Telegram ─────────────────────────────────────────────────────────────────

def _send_telegram(message: str, snapshot_path: str | None):
    try:
        import requests
        base = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

        if snapshot_path:
            try:
                with open(snapshot_path, "rb") as f:
                    resp = requests.post(
                        f"{base}/sendPhoto",
                        data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": message},
                        files={"photo": f},
                        timeout=10,
                    )
                if resp.status_code == 200:
                    print("[ALERT] Telegram photo sent")
                    return
            except Exception as e:
                print(f"[ALERT] Telegram photo failed ({e}), sending text only")

        resp = requests.post(
            f"{base}/sendMessage",
            data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
        if resp.status_code == 200:
            print("[ALERT] Telegram message sent")
        else:
            print(f"[ALERT] Telegram error: {resp.text}")

    except Exception as e:
        print(f"[ALERT] Telegram failed: {e}")


# ─── Public API ───────────────────────────────────────────────────────────────

def send_alert(event_type: str, label: str = "", snapshot_path: str | None = None):
    """
    Send alert via all enabled channels (non-blocking, runs in background thread).
    Respects ALERT_COOLDOWN_SEC and HIGH_CONF_ONLY settings from config.

    event_type: 'motion' | 'audio' | 'yolo' | 'high_confidence'
    """
    if config.HIGH_CONF_ONLY and event_type != "high_confidence":
        return

    if not _can_fire():
        return

    subject = f"[SECURITY ALERT] {event_type.upper()} detected"
    body    = (
        f"Event type : {event_type}\n"
        f"Detail     : {label}\n"
        f"Snapshot   : {snapshot_path or 'none'}\n\n"
        f"-- AI Smart Home Security System"
    )

    def _dispatch():
        if config.EMAIL_ENABLED:
            _send_email(subject, body, snapshot_path)
        if config.TELEGRAM_ENABLED:
            _send_telegram(f"{subject}\n{label}", snapshot_path)

    threading.Thread(target=_dispatch, daemon=True).start()
