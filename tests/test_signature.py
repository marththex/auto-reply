"""Behavior of Gmail signature fetching and appending."""

from autoreply.gmail.drafts import append_signature, pick_default_signature


class TestPickDefaultSignature:
    def test_picks_default_sendas_signature(self):
        sendas = [
            {"sendAsEmail": "other@example.com", "signature": "<p>Wrong</p>"},
            {"sendAsEmail": "me@example.com", "isDefault": True,
             "signature": "<p>Sam Doe<br>555-0100</p>"},
        ]
        assert "Sam Doe" in pick_default_signature(sendas)
        assert "<" not in pick_default_signature(sendas)  # rendered to text

    def test_no_signature_returns_empty(self):
        assert pick_default_signature([{"sendAsEmail": "me@x.com", "isDefault": True}]) == ""
        assert pick_default_signature([]) == ""


class TestAppendSignature:
    def test_appends_with_blank_line_separator(self):
        result = append_signature("Sounds good, see you then.", "Sam Doe\n555-0100")
        assert result == "Sounds good, see you then.\n\nSam Doe\n555-0100"

    def test_empty_signature_leaves_body_untouched(self):
        assert append_signature("Hi there.", "") == "Hi there."

    def test_trailing_whitespace_normalized(self):
        result = append_signature("Reply text.\n\n\n", "Sig")
        assert result == "Reply text.\n\nSig"
