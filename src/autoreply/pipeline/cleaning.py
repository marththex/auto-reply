"""Strip quoted reply chains, signatures, and boilerplate from email bodies."""

import re
from html.parser import HTMLParser

_QUOTE_LINE = re.compile(r"^\s*>")
_ORIGINAL_MESSAGE = re.compile(r"^-{2,}\s*Original Message\s*-{2,}$", re.IGNORECASE)
_FORWARDED = re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}$", re.IGNORECASE)
# Gmail rich-text quotes can leave a trailing '*' after 'wrote:'.
_WROTE_ENDING = re.compile(r"wrote:\s*\*?\s*$")
# Attribution residue that survives wrapping: a bare 'wrote:' line, or a
# fragment carrying the sender's <email> and ending in 'wrote:'.
_ORPHAN_ATTRIBUTION = re.compile(
    r"^\s*(?:wrote:|,?\s*.*<[^<>\s]+@[^<>\s]+>>?\s+wrote:)\s*\*?\s*$"
)
# Some clients fuse the attribution onto the reply's last content line.
# Require '<...@...>' so prose mentioning 'On ...' is not clipped. [^<>] not
# \S inside the address: addresses in the wild carry U+200A hair spaces.
_FUSED_ATTRIBUTION = re.compile(r"\s+On\s.{0,300}?<[^<>]{1,100}@[^<>]{1,100}>>?\s*wrote:\s*\*?")
# Invisible junk seen after fused 'wrote:' - plain/no-break/hair/zero-width
# spaces and BOM.
_INVISIBLE_WS = " \t\u00a0\u200a\u200b\ufeff"
_SIG_DELIMITER = re.compile(r"^-- ?$")
_DEVICE_BOILERPLATE = re.compile(
    r"^(Sent from my \S+|Get Outlook for \S+).*$", re.IGNORECASE
)


def clean_reply_body(text: str) -> str:
    return _normalize_whitespace(_strip_signature(_strip_quoted(_truncate_at_bom_echo(text))))


def _truncate_at_bom_echo(text: str) -> str:
    """Cut at a BOM separator: some clients append the entire original
    message after U+FEFF with no quote markers at all (29% of the real
    corpus). A leading BOM is a mere encoding artifact, not a separator."""
    text = text.lstrip("﻿ \t\n")
    if "﻿" in text:
        text = text.split("﻿", 1)[0]
    return text


def word_count(text: str) -> int:
    return len(text.split())


def _strip_quoted(text: str) -> str:
    """Drop quoted lines and attribution headers; truncate at embedded copies."""
    lines = text.splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if _ORIGINAL_MESSAGE.match(stripped) or _FORWARDED.match(stripped):
            break  # everything below is an unquoted copy of the original message
        if _QUOTE_LINE.match(lines[i]):
            i += 1
            continue
        span = _attribution_span(lines, i)
        if span:
            i += span
            continue
        fused = _FUSED_ATTRIBUTION.search(lines[i])
        if fused:
            prefix = lines[i][: fused.start()].rstrip()
            if prefix:
                kept.append(prefix)
            if lines[i][fused.end():].strip(_INVISIBLE_WS):
                break  # the original message continues unquoted on this line
            i += 1
            continue
        if not _ORPHAN_ATTRIBUTION.match(lines[i]):
            kept.append(lines[i])
        i += 1
    return "\n".join(kept)


def _attribution_span(lines: list[str], i: int) -> int:
    """Lines consumed by an 'On <date>, <sender> wrote:' header at i (0 if none).

    Gmail wraps long attribution headers, so try joining up to two more lines.
    """
    line = lines[i].strip()
    if not line.startswith("On "):
        return 0
    joined = line
    for span in (1, 2, 3):
        if _WROTE_ENDING.search(joined):
            return span
        if i + span >= len(lines) or len(joined) >= 300:
            return 0
        joined = f"{joined} {lines[i + span].strip()}"
    return 0


def _strip_signature(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if _SIG_DELIMITER.match(line):
            break
        # Device boilerplate can land anywhere (top, when replying inline
        # from a phone), not just at the end.
        if _DEVICE_BOILERPLATE.match(line.strip()):
            continue
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table",
    "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("style", "script"):
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS and tag != "br":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    return "".join(extractor.parts)
