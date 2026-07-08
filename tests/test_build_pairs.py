"""Behavior of the build-pairs CLI: cleaning, filtering, JSONL output, stats."""

import json

from autoreply.pipeline.build_pairs import build_dataset, main
from autoreply.pipeline.mbox import pair_replies, parse_mbox

from tests.test_pairing import ALICE, ME, make_message, write_mbox

QUOTED_TAIL = (
    "\n\nOn Mon, Jun 1, 2026 at 10:00 AM Alice Smith <alice@example.com> wrote:\n"
    "> Are you free on Thursday for lunch downtown?\n"
)
LONG_REPLY = (
    "Thursday works great for me. Let's meet at the usual place around noon "
    "so we can catch up properly before the afternoon rush starts."
)


def two_thread_mbox(tmp_path):
    """One thread with a long reply, one with a 'thanks!' one-liner."""
    return write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="Lunch?", body="Are you free on Thursday for lunch downtown?",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="111", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Lunch?", body=LONG_REPLY + QUOTED_TAIL,
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="111", labels="Sent", in_reply_to="<a1@example.com>",
        ),
        make_message(
            from_=ALICE, to=ME, subject="Doc", body="Here is the document you asked for.",
            msg_id="<a2@example.com>", date="Tue, 02 Jun 2026 10:00:00 +0000",
            thread_id="222", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Doc", body="Thanks!" + QUOTED_TAIL,
            msg_id="<m2@example.com>", date="Tue, 02 Jun 2026 11:00:00 +0000",
            thread_id="222", labels="Sent", in_reply_to="<a2@example.com>",
        ),
    ])


def test_short_replies_are_filtered_and_counted(tmp_path):
    pairs = pair_replies(parse_mbox(two_thread_mbox(tmp_path)))
    records, stats = build_dataset(pairs, min_words=15)
    assert len(records) == 1
    assert stats["pairs_found"] == 2
    assert stats["kept"] == 1
    assert stats["filtered_short"] == 1


def test_records_contain_cleaned_bodies_and_metadata(tmp_path):
    pairs = pair_replies(parse_mbox(two_thread_mbox(tmp_path)))
    records, _ = build_dataset(pairs, min_words=15)
    record = records[0]
    assert record["thread_id"] == "111"
    assert record["incoming"]["from"] == "alice@example.com"
    assert record["incoming"]["subject"] == "Lunch?"
    assert record["incoming"]["body"] == "Are you free on Thursday for lunch downtown?"
    assert record["reply"]["body"] == LONG_REPLY
    assert "wrote:" not in record["reply"]["body"]
    assert record["incoming"]["date"].startswith("2026-06-01")
    assert record["reply"]["date"].startswith("2026-06-01")


def test_pair_with_empty_incoming_after_cleaning_is_filtered(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="(attachment only)", body="",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="333", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: (attachment only)", body=LONG_REPLY,
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="333", labels="Sent", in_reply_to="<a1@example.com>",
        ),
    ])
    records, stats = build_dataset(pair_replies(parse_mbox(path)), min_words=15)
    assert records == []
    assert stats["filtered_empty_incoming"] == 1


def test_cli_writes_jsonl_and_prints_stats(tmp_path, capsys):
    mbox_path = two_thread_mbox(tmp_path)
    out_path = tmp_path / "pairs.jsonl"
    main([str(mbox_path), "--out", str(out_path)])

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reply"]["body"] == LONG_REPLY

    printed = capsys.readouterr().out
    assert "4" in printed        # total messages
    assert "kept" in printed.lower()
    assert str(out_path) in printed
