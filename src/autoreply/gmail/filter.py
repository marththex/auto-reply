"""Automated-sender filter: reject no-reply/bulk/notification mail before any
model call. Layers run cheapest-first; the allowlist overrides everything."""

import re
from dataclasses import dataclass

AUTOMATED_LOCAL_PREFIXES = (
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "notifications", "notification", "alerts", "mailer", "automated",
)
AUTOMATED_PHRASES = (
    "this is an automated message",
    "do not reply to this email",
    "this mailbox is not monitored",
)
SKIP_PRECEDENCE = ("bulk", "auto_reply")

# Layer 4: scored weak signals for marketing mail that ships without bulk
# headers (a real energy-brand newsletter slipped through layers 1-3).
# One point per signal class; skip at >= 2.
CONTENT_SCORE_THRESHOLD = 2
_MARKETING_SUBDOMAINS = re.compile(
    r"^(mp\d*|mkt|mktg|e|em|email|mail|links?|click|go|news(letter)?|info|rewards|send|events?|member)$"
)
_SALE_SUBJECT = re.compile(
    r"\$\d+[^.]*\boff\b|%\s*off|\bsale\b|limited time|ends soon|exclusive offer|free shipping",
    re.IGNORECASE,
)
_UNSUBSCRIBE_TEXT = re.compile(
    r"unsubscribe|view in browser|(manage|update) your preferences|email preferences|opt[ -]?out",
    re.IGNORECASE,
)
_TRACKER_LINK = re.compile(
    r"https?://(links?|click|email|e|go|mp\d*)\.[^\s\]]+|/ls/click|[?&]utm_",
    re.IGNORECASE,
)
_TRACKER_LINK_MIN = 3
_SOFT_DO_NOT_REPLY = re.compile(
    r"do not reply|not (actively )?monitored", re.IGNORECASE
)


@dataclass
class FilterResult:
    skip: bool
    reason: str | None = None


def should_skip(
    *,
    sender: str,
    headers: dict,
    subject: str,
    body: str,
    allowlist: list[str],
) -> FilterResult:
    sender = sender.lower().strip()

    for entry in allowlist:
        entry = entry.lower().strip()
        if sender == entry or (entry.startswith("@") and sender.endswith(entry)):
            return FilterResult(skip=False, reason=f"allowlisted ({entry})")

    # Match as a prefix ("no-reply@") or as a delimited segment
    # ("googleplay-noreply@"), but never as a bare substring, so a name like
    # valerts.person@ doesn't trip on "alerts".
    local_part = sender.split("@", 1)[0]
    segments = re.split(r"[.\-_+]", local_part)
    for prefix in AUTOMATED_LOCAL_PREFIXES:
        if local_part.startswith(prefix) or prefix in segments:
            return FilterResult(skip=True, reason=f"sender pattern '{prefix}@'")

    lowered_headers = {k.lower(): str(v).lower() for k, v in headers.items()}
    if "list-unsubscribe" in lowered_headers:
        return FilterResult(skip=True, reason="List-Unsubscribe header")
    if lowered_headers.get("precedence") in SKIP_PRECEDENCE:
        return FilterResult(skip=True, reason=f"Precedence: {lowered_headers['precedence']}")
    auto_submitted = lowered_headers.get("auto-submitted")
    if auto_submitted and auto_submitted != "no":
        return FilterResult(skip=True, reason=f"Auto-Submitted: {auto_submitted}")

    haystack = f"{subject}\n{body}".lower()
    for phrase in AUTOMATED_PHRASES:
        if phrase in haystack:
            return FilterResult(skip=True, reason=f"content phrase '{phrase}'")

    signals = _content_signals(sender=sender, subject=subject, body=body)
    if len(signals) >= CONTENT_SCORE_THRESHOLD:
        return FilterResult(
            skip=True, reason=f"content score {len(signals)}: " + " + ".join(signals)
        )

    return FilterResult(skip=False)


def _content_signals(*, sender: str, subject: str, body: str) -> list[str]:
    signals = []
    domain = sender.split("@", 1)[-1]
    if _MARKETING_SUBDOMAINS.match(domain.split(".", 1)[0]):
        signals.append("marketing-subdomain")
    if _SALE_SUBJECT.search(subject):
        signals.append("sale-subject")
    if _UNSUBSCRIBE_TEXT.search(body):
        signals.append("unsubscribe-text")
    if len(_TRACKER_LINK.findall(body)) >= _TRACKER_LINK_MIN:
        signals.append("tracker-links")
    if _SOFT_DO_NOT_REPLY.search(body):
        signals.append("do-not-reply-text")
    return signals
