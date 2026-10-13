"""SQLite storage for users, screening cases, reports, settings and the audit trail."""
import json
import os
import sqlite3
from datetime import datetime

from flask import current_app, g, session
from werkzeug.security import generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doctor', 'staff', 'admin')),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_code TEXT UNIQUE NOT NULL,
    patient_code TEXT NOT NULL,
    screened_at TEXT NOT NULL,
    image_path TEXT NOT NULL,
    heatmap_path TEXT,
    overlay_path TEXT,
    gradcam_error TEXT,
    predicted_grade INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    top_score REAL NOT NULL,
    score_margin REAL NOT NULL,
    quality_status TEXT NOT NULL,
    quality_json TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    priority TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    clinician_assessment TEXT,
    clinician_grade INTEGER,
    notes TEXT,
    created_by INTEGER REFERENCES users(id),
    reviewed_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    filename TEXT NOT NULL,
    generated_by INTEGER REFERENCES users(id),
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_id INTEGER,
    username TEXT,
    event_type TEXT NOT NULL,
    case_code TEXT,
    details TEXT
);

-- The audit trail is append-only: block edits and deletions at the database level.
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contact_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    organisation TEXT,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(review_status);
CREATE INDEX IF NOT EXISTS idx_cases_screened ON cases(screened_at);
"""

# Clinic-configurable review rules (editable by admins on the Settings page).
DEFAULT_SETTINGS = {
    "uncertainty_margin": "0.15",     # top-1 minus top-2 score below this -> specialist review
    "min_top_score": "0.50",          # highest class score below this -> specialist review
    "priority_grades": "3,4",         # predicted grades that raise review priority
    "dark_threshold": "25",
    "bright_threshold": "230",
    "min_content_fraction": "0.15",
    "blur_threshold": "15",
}

DEMO_USERS = [
    ("admin", "Demo Administrator", "admin@retinaassist.local", "admin"),
    ("doctor", "Dr. Demo Ophthalmologist", "doctor@retinaassist.local", "doctor"),
    ("staff", "Demo Screening Staff", "staff@retinaassist.local", "staff"),
]


def now_iso():
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """Create tables, default settings and demo accounts (only if they don't exist)."""
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    conn = sqlite3.connect(app.config["DATABASE"])
    try:
        conn.executescript(SCHEMA)
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
        demo_password = app.config["DEMO_PASSWORD"]
        for username, full_name, email, role in DEMO_USERS:
            conn.execute(
                "INSERT OR IGNORE INTO users (username, full_name, email, password_hash, role, is_active, created_at)"
                " VALUES (?, ?, ?, ?, ?, 1, ?)",
                (username, full_name, email, generate_password_hash(demo_password), role, now_iso()),
            )
        conn.commit()
    finally:
        conn.close()
    app.teardown_appcontext(close_db)


def get_settings():
    rows = get_db().execute("SELECT key, value FROM settings").fetchall()
    settings = dict(DEFAULT_SETTINGS)
    settings.update({r["key"]: r["value"] for r in rows})
    return settings


def log_event(event_type, case_code=None, details=None, user_id=None, username=None):
    """Append an audit event. Uses the logged-in user unless one is given explicitly."""
    if user_id is None and username is None:
        user_id = session.get("user_id")
        username = session.get("username")
    get_db().execute(
        "INSERT INTO audit_log (created_at, user_id, username, event_type, case_code, details) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), user_id, username, event_type, case_code,
         json.dumps(details) if isinstance(details, (dict, list)) else details),
    )
    get_db().commit()


def case_dict(row):
    """Convert a cases row into a dict with decoded JSON fields."""
    if row is None:
        return None
    c = dict(row)
    c["scores"] = json.loads(c["scores_json"])
    c["quality"] = json.loads(c["quality_json"])
    c["flags"] = json.loads(c["flags_json"])
    return c
