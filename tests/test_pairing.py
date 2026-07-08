"""Behavior of mbox parsing, thread grouping, and (incoming, reply) pairing."""

import mailbox
from email.message import EmailMessage

from autoreply.pipeline.mbox import infer_my_email, pair_replies, parse_mbox

ME = "sam@example.com"
ALICE = "Alice Smith <alice@example.com>"
BOB = "Bob Jones <bob@example.com>"


def make_message(
    *,
    from_,
    to,
    subject,
    body,
    msg_id,
    date,
    thread_id=None,
    labels=None,
    in_reply_to=None,
    html=False,
):
    msg = EmailMessage()
    msg["From"] = from_
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = msg_id
    msg["Date"] = date
    if thread_id:
        msg["X-GM-THRID"] = thread_id
    if labels:
        msg["X-Gmail-Labels"] = labels
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    if html:
        msg.set_content(body, subtype="html")
    else:
        msg.set_content(body)
    return msg


def write_mbox(tmp_path, messages):
    path = tmp_path / "export.mbox"
    box = mailbox.mbox(str(path))
    for msg in messages:
        box.add(msg)
    box.flush()
    box.close()
    return path


def test_pairs_sent_reply_with_preceding_incoming(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="Lunch?", body="Free on Thursday?",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="111", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Lunch?", body="Thursday works great for me.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="111", labels="Sent", in_reply_to="<a1@example.com>",
        ),
    ])
    pairs = pair_replies(parse_mbox(path))
    assert len(pairs) == 1
    assert "Free on Thursday?" in pairs[0].incoming.body
    assert "Thursday works great" in pairs[0].reply.body
    assert pairs[0].incoming.sender == "alice@example.com"


def test_sent_thread_starter_yields_no_pair(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ME, to=ALICE, subject="Kickoff", body="Starting the thread myself.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="222", labels="Sent",
        ),
    ])
    assert pair_replies(parse_mbox(path)) == []


def test_in_reply_to_beats_recency(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="Plans", body="Original question from Alice.",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="333", labels="Inbox",
        ),
        make_message(
            from_=BOB, to=ME, subject="Re: Plans", body="Bob chiming in later.",
            msg_id="<b1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="333", labels="Inbox", in_reply_to="<a1@example.com>",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Plans", body="Answering Alice directly here.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 12:00:00 +0000",
            thread_id="333", labels="Sent", in_reply_to="<a1@example.com>",
        ),
    ])
    pairs = pair_replies(parse_mbox(path))
    assert len(pairs) == 1
    assert "Original question from Alice." in pairs[0].incoming.body


def test_each_sent_reply_pairs_with_nearest_preceding_incoming(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="Q1", body="First question.",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="444", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Q1", body="First answer from me.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="444", labels="Sent",
        ),
        make_message(
            from_=ALICE, to=ME, subject="Re: Q1", body="Follow-up question.",
            msg_id="<a2@example.com>", date="Mon, 01 Jun 2026 12:00:00 +0000",
            thread_id="444", labels="Inbox",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: Q1", body="Second answer from me.",
            msg_id="<m2@example.com>", date="Mon, 01 Jun 2026 13:00:00 +0000",
            thread_id="444", labels="Sent",
        ),
    ])
    pairs = pair_replies(parse_mbox(path))
    assert len(pairs) == 2
    assert "First question." in pairs[0].incoming.body
    assert "First answer" in pairs[0].reply.body
    assert "Follow-up question." in pairs[1].incoming.body
    assert "Second answer" in pairs[1].reply.body


def test_groups_by_references_when_no_thread_id(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="No THRID", body="Question without gmail headers.",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: No THRID", body="Reply linked only by references.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            in_reply_to="<a1@example.com>",
        ),
        make_message(
            from_=BOB, to=ME, subject="Unrelated", body="Different conversation entirely.",
            msg_id="<b1@example.com>", date="Mon, 01 Jun 2026 09:00:00 +0000",
        ),
    ])
    pairs = pair_replies(parse_mbox(path, my_email=ME))
    assert len(pairs) == 1
    assert "Question without gmail headers." in pairs[0].incoming.body


def test_html_only_body_is_extracted_as_text(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="HTML", body="<p>Rich <b>formatted</b> question?</p>",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="555", labels="Inbox", html=True,
        ),
        make_message(
            from_=ME, to=ALICE, subject="Re: HTML", body="Plain answer.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="555", labels="Sent", in_reply_to="<a1@example.com>",
        ),
    ])
    pairs = pair_replies(parse_mbox(path))
    assert len(pairs) == 1
    assert "Rich formatted question?" in pairs[0].incoming.body
    assert "<" not in pairs[0].incoming.body


def test_sent_detected_by_from_address_without_labels(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=ALICE, to=ME, subject="Hi", body="Question here.",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000",
            thread_id="666",
        ),
        make_message(
            from_=f"Sam <{ME}>", to=ALICE, subject="Re: Hi", body="Answer here.",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000",
            thread_id="666", in_reply_to="<a1@example.com>",
        ),
    ])
    pairs = pair_replies(parse_mbox(path, my_email=ME))
    assert len(pairs) == 1
    assert "Answer here." in pairs[0].reply.body


def test_infer_my_email_from_sent_labels(tmp_path):
    path = write_mbox(tmp_path, [
        make_message(
            from_=f"Sam <{ME}>", to=ALICE, subject="One", body="x",
            msg_id="<m1@example.com>", date="Mon, 01 Jun 2026 10:00:00 +0000", labels="Sent",
        ),
        make_message(
            from_=f"Sam <{ME}>", to=BOB, subject="Two", body="y",
            msg_id="<m2@example.com>", date="Mon, 01 Jun 2026 11:00:00 +0000", labels="Sent",
        ),
        make_message(
            from_=ALICE, to=ME, subject="Three", body="z",
            msg_id="<a1@example.com>", date="Mon, 01 Jun 2026 12:00:00 +0000", labels="Inbox",
        ),
    ])
    assert infer_my_email(parse_mbox(path)) == ME
