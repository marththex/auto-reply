"""Feedback-loop store: every incoming email, generated draft, and (later)
sent reply, so future training rounds can learn from which drafts were
actually used. Skipped messages are logged too, for auditing the
automated-sender filter.

Default path data/autoreply.db (gitignored); on the NAS it lives on a mounted
volume outside the container (AUTOREPLY_DB env var).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(os.environ.get("AUTOREPLY_DB", "data/autoreply.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id            TEXT PRIMARY KEY,   -- Gmail message id
    thread_id     TEXT,
    incoming_body TEXT,
    incoming_date TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id             TEXT REFERENCES emails(id),
    model_generated_text TEXT,
    model_version        TEXT,
    created_at           TEXT
);
CREATE TABLE IF NOT EXISTS sent_replies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id        TEXT REFERENCES emails(id),
    final_text      TEXT,
    draft_id_if_any INTEGER REFERENCES drafts(id),
    sent_at         TEXT
);
CREATE TABLE IF NOT EXISTS skipped (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    gmail_id   TEXT,
    sender     TEXT,
    reason     TEXT,
    skipped_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    finished_at       TEXT,
    backend           TEXT,
    query             TEXT,
    matched           INTEGER,
    filtered          INTEGER,
    already_processed INTEGER,
    drafted           INTEGER,
    capped            INTEGER,
    reconciled        TEXT  -- JSON outcome counts
);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


# Draft outcome states:
#   dry_run       - generated but never saved to Gmail (excluded from everything)
#   pending       - saved as a Gmail draft, outcome unknown
#   sent_unedited - sent as generated (training pair; sent text = ground truth)
#   sent_edited   - edited before sending (best training pair; edits teach)
#   deleted       - discarded (excluded from training, kept for quality review)
_DRAFT_COLUMNS = {
    "gmail_draft_id": "TEXT",
    "status": "TEXT DEFAULT 'pending'",
    "resolved_at": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(drafts)")}
    for column, decl in _DRAFT_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE drafts ADD COLUMN {column} {decl}")
    conn.commit()


def record_email(conn, *, gmail_id: str, thread_id: str,
                 incoming_body: str, incoming_date: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO emails (id, thread_id, incoming_body, incoming_date) "
        "VALUES (?, ?, ?, ?)",
        (gmail_id, thread_id, incoming_body, incoming_date),
    )
    conn.commit()


def record_draft(conn, *, email_id: str, text: str, model_version: str,
                 gmail_draft_id: str | None = None, status: str = "pending") -> int:
    cursor = conn.execute(
        "INSERT INTO drafts (email_id, model_generated_text, model_version, "
        "created_at, gmail_draft_id, status) VALUES (?, ?, ?, ?, ?, ?)",
        (email_id, text, model_version, _now(), gmail_draft_id, status),
    )
    conn.commit()
    return cursor.lastrowid


def processed_email_ids(conn) -> set[str]:
    """Emails that already got a real (non-dry-run) draft - never redraft."""
    rows = conn.execute("SELECT DISTINCT email_id FROM drafts WHERE status != 'dry_run'")
    return {r[0] for r in rows}


def record_skip(conn, *, gmail_id: str, sender: str, reason: str) -> None:
    conn.execute(
        "INSERT INTO skipped (gmail_id, sender, reason, skipped_at) VALUES (?, ?, ?, ?)",
        (gmail_id, sender, reason, _now()),
    )
    conn.commit()


def record_run(conn, *, backend: str, query: str, matched: int, filtered: int,
               already_processed: int, drafted: int, capped: bool,
               reconciled: dict) -> None:
    """One row per bridge invocation - the reviewable audit trail for
    unattended scheduled runs."""
    conn.execute(
        "INSERT INTO runs (finished_at, backend, query, matched, filtered, "
        "already_processed, drafted, capped, reconciled) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), backend, query, matched, filtered, already_processed,
         drafted, int(capped), json.dumps(reconciled)),
    )
    conn.commit()


# TODO(feedback loop): detect sent replies that match a stored draft — poll
# the Sent folder, pair by thread_id, and populate sent_replies with
# draft_id_if_any so training data can distinguish used vs discarded drafts.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
