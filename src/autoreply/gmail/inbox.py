"""Parse Gmail API message payloads into plain dicts for the draft bridge."""

import base64
from email.utils import parseaddr

from autoreply.pipeline.cleaning import html_to_text


def extract_message(msg: dict) -> dict:
    """Flatten a users().messages().get(format='full') response."""
    payload = msg.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    lookup = {k.lower(): v for k, v in headers.items()}
    return {
        "gmail_id": msg.get("id", ""),
        "thread_id": msg.get("threadId", ""),
        "sender": parseaddr(lookup.get("from", ""))[1].lower(),
        "subject": lookup.get("subject", ""),
        "date": lookup.get("date", ""),
        "message_id_header": lookup.get("message-id", ""),
        "headers": headers,
        "body": _extract_body(payload),
    }


def _extract_body(part: dict) -> str:
    plain, html = _collect_bodies(part, {})
    if plain and plain.strip():
        return plain
    if html:
        return html_to_text(html)
    return plain or ""


def _collect_bodies(part: dict, found: dict) -> tuple[str | None, str | None]:
    if part.get("filename"):
        pass  # attachment: skip
    elif part.get("mimeType") == "text/plain" and "plain" not in found:
        found["plain"] = _decode(part)
    elif part.get("mimeType") == "text/html" and "html" not in found:
        found["html"] = _decode(part)
    for child in part.get("parts", []):
        _collect_bodies(child, found)
    return found.get("plain"), found.get("html")


def _decode(part: dict) -> str:
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
