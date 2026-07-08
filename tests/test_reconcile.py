"""Behavior of draft-outcome reconciliation (pure logic; Gmail calls are thin)."""

from autoreply.db import connect, record_draft, record_email
from autoreply.gmail.reconcile import classify_outcome, texts_match


class TestTextsMatch:
    def test_identical_text_matches(self):
        assert texts_match("Hi Alice,\n\nSounds good.", "Hi Alice,\n\nSounds good.")

    def test_whitespace_and_wrapping_differences_still_match(self):
        # Gmail rewraps lines and pads whitespace when sending a draft.
        draft = "Hi Alice,\n\nSounds good - see you Thursday at noon.\n"
        sent = "Hi Alice,\n\nSounds good - see you\nThursday at noon."
        assert texts_match(draft, sent)

    def test_real_edits_do_not_match(self):
        draft = "Sounds good - see you Thursday."
        sent = "Sounds good - see you Friday instead, Thursday got busy."
        assert not texts_match(draft, sent)


class TestClassifyOutcome:
    def test_draft_still_exists_stays_pending(self):
        assert classify_outcome(draft_exists=True, sent_text=None, draft_text="x") == "pending"

    def test_gone_with_matching_sent_text_is_sent_unedited(self):
        outcome = classify_outcome(draft_exists=False, sent_text="Hi  Bob", draft_text="Hi Bob")
        assert outcome == "sent_unedited"

    def test_gone_with_changed_sent_text_is_sent_edited(self):
        outcome = classify_outcome(draft_exists=False,
                                   sent_text="Hi Bob, actually no.", draft_text="Hi Bob, yes!")
        assert outcome == "sent_edited"

    def test_gone_with_no_sent_message_is_deleted(self):
        assert classify_outcome(draft_exists=False, sent_text=None, draft_text="x") == "deleted"


class TestSchemaMigration:
    def test_drafts_table_gains_outcome_columns(self, tmp_path):
        with connect(tmp_path / "t.db") as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(drafts)")}
        assert {"gmail_draft_id", "status", "resolved_at"} <= cols

    def test_record_draft_stores_id_and_status(self, tmp_path):
        with connect(tmp_path / "t.db") as conn:
            record_email(conn, gmail_id="e1", thread_id="t1",
                         incoming_body="q", incoming_date="2026-07-07")
            record_draft(conn, email_id="e1", text="reply", model_version="v",
                         gmail_draft_id="r123", status="pending")
            row = conn.execute(
                "SELECT gmail_draft_id, status FROM drafts").fetchone()
        assert row == ("r123", "pending")

    def test_migration_is_idempotent_on_existing_db(self, tmp_path):
        path = tmp_path / "t.db"
        with connect(path):
            pass
        with connect(path) as conn:  # second connect must not fail on ALTERs
            cols = {r[1] for r in conn.execute("PRAGMA table_info(drafts)")}
        assert "status" in cols


class TestDraftLiveness:
    def test_normal_draft_is_live(self):
        from autoreply.gmail.reconcile import draft_is_live
        assert draft_is_live({"message": {"labelIds": ["DRAFT"]}})

    def test_trashed_draft_is_not_live(self):
        # Real case: discarding a threaded reply draft moves it to TRASH,
        # but drafts.get still returns it.
        from autoreply.gmail.reconcile import draft_is_live
        assert not draft_is_live({"message": {"labelIds": ["DRAFT", "TRASH"]}})
