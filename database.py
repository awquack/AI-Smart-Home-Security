# database.py – SQLite event logging for AI Smart Home Security
import sqlite3
import datetime
import os
from config import DB_PATH


def init_db():
    """Create the events table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            event_type    TEXT    NOT NULL,
            label         TEXT,
            confidence    REAL,
            snapshot_path TEXT,
            area          INTEGER,
            x             INTEGER,
            y             INTEGER,
            w             INTEGER,
            h             INTEGER
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialised  →  {DB_PATH}")


def log_event(event_type, label=None, confidence=None,
              snapshot_path=None, area=None, x=None, y=None, w=None, h=None):
    """
    Insert one security event row.

    event_type: 'motion' | 'audio' | 'yolo' | 'high_confidence'
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO events
               (timestamp, event_type, label, confidence,
                snapshot_path, area, x, y, w, h)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, event_type, label, confidence, snapshot_path, area, x, y, w, h)
    )
    conn.commit()
    conn.close()


def get_recent_events(limit=50):
    """Return the most recent `limit` events as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_events_by_type(event_type, limit=100):
    """Return events filtered by event_type."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM events WHERE event_type=? ORDER BY id DESC LIMIT ?",
        (event_type, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event_counts():
    """Return a dict with counts per event_type."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}
