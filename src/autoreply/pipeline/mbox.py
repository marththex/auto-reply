"""Parse a Google Takeout Gmail mbox export and pair sent replies with the
incoming message they answered."""

import mailbox
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from autoreply.pipeline.cleaning import html_to_text

_FALLBACK_DATE = datetime.max.replace(tzinfo=timezone.utc)


@dataclass
class ParsedMessage:
    message_id: str
    thread_key: str
    in_reply_to: str | None
    sender: str
    subject: str
    date: datetime | None
    body: str
    is_sent: bool


@dataclass
class Pair:
    incoming: ParsedMessage
    reply: ParsedMessage


def parse_mbox(path: str | Path, my_email: str | None = None) -> list[ParsedMessage]:
    box = mailbox.mbox(str(path))
    try:
        return [_parse_message(msg, my_email) for msg in box]
    finally:
        box.close()


def infer_my_email(messages: list[ParsedMessage]) -> str | None:
    """Most common From address among sent-labeled messages."""
    senders = Counter(m.sender for m in messages if m.is_sent and m.sender)
    return senders.most_common(1)[0][0] if senders else None


def pair_replies(messages: list[ParsedMessage]) -> list[Pair]:
    """For each sent message, find the incoming message it replied to.

    Prefers an exact In-Reply-To match; falls back to the nearest preceding
    non-sent message in the same thread. Sent thread-starters yield no pair.
    """
    by_id = {m.message_id: m for m in messages if m.message_id}
    threads: dict[str, list[ParsedMessage]] = defaultdict(list)
    for m in messages:
        threads[m.thread_key].append(m)

    pairs: list[Pair] = []
    for thread in threads.values():
        thread.sort(key=_sort_key)
        for i, msg in enumerate(thread):
            if not msg.is_sent:
                continue
            incoming = None
            if msg.in_reply_to:
                candidate = by_id.get(msg.in_reply_to)
                if candidate is not None and not candidate.is_sent:
                    incoming = candidate
            if incoming is None:
                preceding = [m for m in thread[:i] if not m.is_sent]
                incoming = preceding[-1] if preceding else None
            if incoming is not None:
                pairs.append(Pair(incoming=incoming, reply=msg))
    pairs.sort(key=lambda p: _sort_key(p.reply))
    return pairs


def _parse_message(msg: Message, my_email: str | None) -> ParsedMessage:
    message_id = (msg.get("Message-ID") or "").strip()
    in_reply_to = (msg.get("In-Reply-To") or "").strip() or None
    references = (msg.get("References") or "").split()
    sender = parseaddr(msg.get("From") or "")[1].lower()
    labels = {label.strip().lower() for label in (msg.get("X-Gmail-Labels") or "").split(",")}
    is_sent = "sent" in labels or (my_email is not None and sender == my_email.lower())

    thread_key = (msg.get("X-GM-THRID") or "").strip()
    if not thread_key:
        # No Gmail thread id: root the thread at the start of the References
        # chain so replies land in the same group as their original.
        thread_key = references[0] if references else (in_reply_to or message_id)

    return ParsedMessage(
        message_id=message_id,
        thread_key=thread_key,
        in_reply_to=in_reply_to,
        sender=sender,
        subject=str(msg.get("Subject") or "").strip(),
        date=_parse_date(msg),
        body=_extract_body(msg),
        is_sent=is_sent,
    )


def _parse_date(msg: Message) -> datetime | None:
    raw = msg.get("Date")
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sort_key(m: ParsedMessage) -> datetime:
    return m.date or _FALLBACK_DATE


def _extract_body(msg: Message) -> str:
    plain = html = None
    for part in msg.walk():
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain" and plain is None:
            plain = _decode_part(part)
        elif ctype == "text/html" and html is None:
            html = _decode_part(part)
    if plain and plain.strip():
        return plain
    if html:
        return html_to_text(html)
    return plain or ""


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")
