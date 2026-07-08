"""Behavior of Gmail API message payload parsing."""

import base64

from autoreply.gmail.inbox import extract_message


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def gmail_message(*, plain=None, html=None, headers=None):
    """Realistic users().messages().get(format='full') response shape."""
    default_headers = [
        {"name": "From", "value": "Alice Smith <alice@example.com>"},
        {"name": "Subject", "value": "Lunch?"},
        {"name": "Date", "value": "Mon, 06 Jul 2026 10:00:00 -0700"},
        {"name": "Message-ID", "value": "<a1@mail.example.com>"},
    ]
    parts = []
    if plain is not None:
        parts.append({"mimeType": "text/plain", "body": {"data": b64(plain)}})
    if html is not None:
        parts.append({"mimeType": "text/html", "body": {"data": b64(html)}})
    return {
        "id": "18abc",
        "threadId": "17def",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": headers or default_headers,
            "parts": parts,
        },
    }


def test_extracts_ids_headers_and_plain_body():
    msg = extract_message(gmail_message(plain="Are you free Thursday?\n",
                                        html="<p>Are you free Thursday?</p>"))
    assert msg["gmail_id"] == "18abc"
    assert msg["thread_id"] == "17def"
    assert msg["sender"] == "alice@example.com"
    assert msg["subject"] == "Lunch?"
    assert msg["message_id_header"] == "<a1@mail.example.com>"
    assert msg["body"].strip() == "Are you free Thursday?"
    assert msg["headers"]["From"] == "Alice Smith <alice@example.com>"


def test_html_only_message_is_converted_to_text():
    msg = extract_message(gmail_message(html="<p>Rich <b>question</b>?</p>"))
    assert "Rich question?" in msg["body"]
    assert "<" not in msg["body"]


def test_nested_multipart_is_walked():
    inner = gmail_message(plain="Nested body here.")["payload"]
    outer = {
        "id": "18abc", "threadId": "17def",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": inner["headers"],
            "parts": [
                {"mimeType": "multipart/alternative", "parts": inner["parts"]},
                {"mimeType": "application/pdf", "filename": "doc.pdf",
                 "body": {"attachmentId": "att1"}},
            ],
        },
    }
    assert extract_message(outer)["body"].strip() == "Nested body here."


def test_body_on_payload_itself_for_simple_messages():
    simple = {
        "id": "18abc", "threadId": "17def",
        "payload": {
            "mimeType": "text/plain",
            "headers": gmail_message()["payload"]["headers"],
            "body": {"data": b64("Short and simple.")},
        },
    }
    assert extract_message(simple)["body"] == "Short and simple."
