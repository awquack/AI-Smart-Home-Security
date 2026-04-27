# database.py – SQLite event logging for AI Smart Home Security
import sqlite3
import datetime
import os
from config import DB_PATH


def init_db():
    """Create the events and known_faces tables if they don't exist."""
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            encoding   TEXT NOT NULL,
            added_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialised  →  {DB_PATH}")


def add_known_face(name, encoding_list):
    """Store a face encoding (128-d list) for a named person."""
    import json
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO known_faces (name, encoding, added_at) VALUES (?, ?, ?)",
        (name, json.dumps(encoding_list), ts)
    )
    conn.commit()
    conn.close()


def get_known_faces():
    """Return all known face rows as list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM known_faces").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_known_face(face_id):
    """Delete a known face by id."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM known_faces WHERE id=?", (face_id,))
    conn.commit()
    conn.close()


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


def get_today_counts():
    """Return event counts for today only."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT event_type, COUNT(*) as cnt FROM events "
        "WHERE timestamp LIKE ? GROUP BY event_type",
        (f"{today}%",)
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_hourly_counts():
    """Return event counts per hour for today (list of 24 ints)."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hr, COUNT(*) "
        "FROM events WHERE timestamp LIKE ? GROUP BY hr",
        (f"{today}%",)
    ).fetchall()
    conn.close()
    counts = [0] * 24
    for hr, cnt in rows:
        counts[hr] = cnt
    return counts


def get_daily_counts():
    """Return event counts per day for the last 7 days."""
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT date(timestamp) as day, COUNT(*) FROM events "
        "WHERE date(timestamp) >= date('now', '-6 days') "
        "GROUP BY day ORDER BY day"
    ).fetchall()
    conn.close()
    # Build a full 7-day list with zeros for missing days
    today  = datetime.date.today()
    labels = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    data   = {r[0]: r[1] for r in rows}
    return {"labels": labels, "values": [data.get(d, 0) for d in labels]}
