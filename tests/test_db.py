"""Behavior of the feedback-loop SQLite store."""

from autoreply.db import connect, record_draft, record_email, record_skip


def test_connect_creates_schema(tmp_path):
    with connect(tmp_path / "test.db") as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"emails", "drafts", "sent_replies", "skipped"} <= tables


def test_record_email_is_idempotent_by_gmail_id(tmp_path):
    with connect(tmp_path / "test.db") as conn:
        for _ in range(2):
            record_email(conn, gmail_id="abc123", thread_id="t1",
                         incoming_body="Hello?", incoming_date="2026-07-07T10:00:00")
        count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    assert count == 1


def test_record_draft_links_email_and_stores_version(tmp_path):
    with connect(tmp_path / "test.db") as conn:
        record_email(conn, gmail_id="abc123", thread_id="t1",
                     incoming_body="Hello?", incoming_date="2026-07-07T10:00:00")
        draft_id = record_draft(conn, email_id="abc123",
                                text="Hi back!", model_version="lora-v2")
        row = conn.execute(
            "SELECT email_id, model_generated_text, model_version, created_at "
            "FROM drafts WHERE id=?", (draft_id,)).fetchone()
    assert row[0] == "abc123"
    assert row[1] == "Hi back!"
    assert row[2] == "lora-v2"
    assert row[3]  # created_at populated


def test_record_skip_logs_reason(tmp_path):
    with connect(tmp_path / "test.db") as conn:
        record_skip(conn, gmail_id="xyz", sender="noreply@shop.example.com",
                    reason="sender pattern 'noreply@'")
        row = conn.execute("SELECT sender, reason FROM skipped").fetchone()
    assert row == ("noreply@shop.example.com", "sender pattern 'noreply@'")


def test_record_run_stores_summary(tmp_path):
    from autoreply.db import record_run

    with connect(tmp_path / "test.db") as conn:
        record_run(conn, backend="remote", query="is:unread in:inbox",
                   matched=5, filtered=3, already_processed=1, drafted=1,
                   capped=False, reconciled={"deleted": 1, "pending": 0})
        row = conn.execute(
            "SELECT backend, matched, filtered, already_processed, drafted, capped, "
            "reconciled, finished_at FROM runs").fetchone()
    assert row[0] == "remote"
    assert row[1:6] == (5, 3, 1, 1, 0)
    assert '"deleted": 1' in row[6]
    assert row[7]  # finished_at set
