"""Reconcile pending drafts with their real-world outcome.

Runs at the start of every bridge invocation, before new mail is touched.
For each draft still marked 'pending': one drafts.get - if it's gone, one
thread fetch decides whether it was sent (unedited vs edited, by comparing
normalized texts) or deleted. Cheap and idempotent: a handful of API calls
against known ids, never a mailbox scan.
"""

import logging
import re
from datetime import datetime, timezone

from googleapiclient.errors import HttpError

from autoreply.gmail.inbox import extract_message

log = logging.getLogger("autoreply.reconcile")

_WS = re.compile(r"\s+")


def texts_match(draft_text: str, sent_text: str) -> bool:
    """Whitespace-insensitive equality: Gmail rewraps drafts when sending."""
    return _WS.sub(" ", draft_text).strip() == _WS.sub(" ", sent_text).strip()


def classify_outcome(*, draft_exists: bool, sent_text: str | None,
                     draft_text: str) -> str:
    if draft_exists:
        return "pending"
    if sent_text is None:
        return "deleted"
    if texts_match(draft_text, sent_text):
        return "sent_unedited"
    return "sent_edited"


def reconcile(service, conn) -> dict:
    """Resolve pending drafts; returns outcome counts for the run summary."""
    counts = {"pending": 0, "sent_unedited": 0, "sent_edited": 0, "deleted": 0}
    rows = conn.execute(
        "SELECT d.id, d.email_id, d.gmail_draft_id, d.model_generated_text, "
        "d.created_at, e.thread_id FROM drafts d JOIN emails e ON e.id = d.email_id "
        "WHERE d.status = 'pending' AND d.gmail_draft_id IS NOT NULL"
    ).fetchall()

    for draft_id, email_id, gmail_draft_id, draft_text, created_at, thread_id in rows:
        draft_exists = _draft_exists(service, gmail_draft_id)
        sent_text = None
        if not draft_exists:
            sent_text = _sent_text_after(service, thread_id, created_at)
        outcome = classify_outcome(draft_exists=draft_exists,
                                   sent_text=sent_text, draft_text=draft_text)
        counts[outcome] += 1
        if outcome == "pending":
            continue
        conn.execute("UPDATE drafts SET status = ?, resolved_at = ? WHERE id = ?",
                     (outcome, _now(), draft_id))
        if outcome in ("sent_unedited", "sent_edited"):
            conn.execute(
                "INSERT INTO sent_replies (email_id, final_text, draft_id_if_any, sent_at) "
                "VALUES (?, ?, ?, ?)", (email_id, sent_text, draft_id, _now()),
            )
        conn.commit()
        log.info("reconciled draft %s (email %s): %s", gmail_draft_id, email_id, outcome)
    return counts


def draft_is_live(draft_resource: dict) -> bool:
    """A trashed draft counts as deleted: discarding a threaded reply draft
    moves it to TRASH, and drafts.get keeps returning it anyway."""
    return "TRASH" not in draft_resource.get("message", {}).get("labelIds", [])


def _draft_exists(service, gmail_draft_id: str) -> bool:
    try:
        resource = service.users().drafts().get(
            userId="me", id=gmail_draft_id).execute()
        return draft_is_live(resource)
    except HttpError as e:
        if e.resp.status == 404:
            return False
        raise


def _sent_text_after(service, thread_id: str, created_at: str) -> str | None:
    """Body of the newest SENT message in the thread after the draft was made."""
    try:
        thread = service.users().threads().get(
            userId="me", id=thread_id, format="full").execute()
    except HttpError as e:
        if e.resp.status == 404:
            return None
        raise
    created_ms = _iso_to_epoch_ms(created_at)
    candidates = [
        m for m in thread.get("messages", [])
        if "SENT" in m.get("labelIds", []) and int(m.get("internalDate", 0)) >= created_ms
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda m: int(m.get("internalDate", 0)))
    return extract_message(newest)["body"]


def _iso_to_epoch_ms(iso: str) -> int:
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1000)
    except ValueError:
        return 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
