"""Behavior of reply-body cleaning: strip quoted chains, signatures, boilerplate."""

from autoreply.pipeline.cleaning import clean_reply_body, html_to_text, word_count


class TestCleanReplyBody:
    def test_plain_text_passes_through(self):
        assert clean_reply_body("Sounds good, see you Tuesday.") == "Sounds good, see you Tuesday."

    def test_strips_gmail_attribution_and_quoted_lines(self):
        text = (
            "Happy to help - attached the report.\n"
            "\n"
            "On Mon, Jun 1, 2026 at 3:14 PM Alice Smith <alice@example.com> wrote:\n"
            "> Could you send the Q2 report?\n"
            "> Thanks\n"
        )
        assert clean_reply_body(text) == "Happy to help - attached the report."

    def test_strips_attribution_wrapped_across_two_lines(self):
        text = (
            "Works for me.\n"
            "\n"
            "On Mon, Jun 1, 2026 at 3:14 PM Alice Smith\n"
            "<alice@example.com> wrote:\n"
            "> How about Tuesday?\n"
        )
        assert clean_reply_body(text) == "Works for me."

    def test_strips_quoted_lines_without_attribution(self):
        text = "Agreed on all points.\n\n> The proposal\n> looks solid.\n"
        assert clean_reply_body(text) == "Agreed on all points."

    def test_keeps_inline_reply_text_below_quotes(self):
        text = (
            "On Mon, Jun 1, 2026 at 3:14 PM Alice <a@example.com> wrote:\n"
            "> Can you make Thursday?\n"
            "Thursday works, noon?\n"
            "> And bring the deck?\n"
            "Yes, will do.\n"
        )
        assert clean_reply_body(text) == "Thursday works, noon?\nYes, will do."

    def test_reply_that_is_only_quotes_becomes_empty(self):
        text = "On Mon, Jun 1, 2026 at 3:14 PM Alice <a@example.com> wrote:\n> ping\n"
        assert clean_reply_body(text) == ""

    def test_strips_outlook_original_message_block(self):
        text = (
            "Let me check and get back to you.\n"
            "\n"
            "-----Original Message-----\n"
            "From: Bob <bob@example.com>\n"
            "Sent: Monday\n"
            "The full original text here, unquoted.\n"
        )
        assert clean_reply_body(text) == "Let me check and get back to you."

    def test_strips_forwarded_message_block(self):
        text = "FYI see below.\n\n---------- Forwarded message ---------\nFrom: Carol <c@example.com>\n"
        assert clean_reply_body(text) == "FYI see below."

    def test_strips_signature_after_dash_dash_delimiter(self):
        text = "I'll call you tomorrow.\n\n-- \nSam Doe\nProduct | 555-0134\n"
        assert clean_reply_body(text) == "I'll call you tomorrow."

    def test_strips_sent_from_device_boilerplate(self):
        text = "Running late, start without me.\n\nSent from my iPhone\n"
        assert clean_reply_body(text) == "Running late, start without me."

    def test_strips_device_boilerplate_above_inline_reply(self):
        # iPhone inline-reply style: auto-signature lands at the top,
        # the real reply text follows below.
        text = "Sent from my iPhone\n\nHere is my actual reply, typed below the auto signature.\n"
        assert clean_reply_body(text) == "Here is my actual reply, typed below the auto signature."

    def test_strips_attribution_wrapped_across_three_lines(self):
        text = (
            "Reply above a badly wrapped header.\n"
            "\n"
            "On Mon, Jun 1, 2026 at 3:14 PM Alice Smith\n"
            "<alice@example.com>\n"
            "wrote:\n"
            "> Original question?\n"
        )
        assert clean_reply_body(text) == "Reply above a badly wrapped header."

    def test_strips_attribution_with_gmail_bold_asterisk(self):
        # Gmail rich-text quotes can render as 'wrote:*' in the plain part.
        text = (
            "Thanks all, confirmed.\n"
            "\n"
            "On Mon, Jun 1, 2026 at 3:14 PM Events Team\n"
            "<events@example.com>> wrote:*\n"
            "> Details below.\n"
        )
        assert clean_reply_body(text) == "Thanks all, confirmed."

    def test_strips_orphaned_attribution_fragment(self):
        # Leftover fragment: no 'On' prefix, but sender-in-angle-brackets + wrote:
        text = "Got it, thanks.\n\n, Sam <sam@example.com>  wrote:\n> Earlier note.\n"
        assert clean_reply_body(text) == "Got it, thanks."

    def test_strips_attribution_fused_onto_content_line(self):
        # Some clients join the attribution to the reply text with no newline.
        text = (
            "Count me in for the 14th - Sam On Dec 30, 2025, at 3:12 PM, "
            "Riley D <riley@example.com> wrote:\n"
            "> Are you joining us?\n"
        )
        assert clean_reply_body(text) == "Count me in for the 14th - Sam"

    def test_truncates_at_fused_attribution_followed_by_unquoted_original(self):
        # Worst case seen in real data: attribution glued mid-line, the
        # original message following unquoted, with hair spaces (U+200A)
        # inside the email address and a BOM after 'wrote:'.
        text = (
            "\U0001f601 - Sam On Dec 30, 2025, at 3: 12 PM, "
            "Doe, Riley <riley. doe@ example.com> wrote: ﻿ Good morning Sam,\n"
            "the original message continues here with no quote markers\n"
            "for several more lines.\n"
        )
        assert clean_reply_body(text) == "\U0001f601 - Sam"

    def test_truncates_at_bom_followed_by_echoed_original(self):
        # 112/383 real replies had this: the client appends the entire
        # incoming message after a BOM, with no quote markers at all.
        text = "Happy to help again, send the details.\n﻿\nHi; My name is Dana and I run the volunteer program..."
        assert clean_reply_body(text) == "Happy to help again, send the details."

    def test_leading_bom_alone_is_stripped_not_truncated(self):
        text = "﻿Thanks, that works for me and the team as well."
        assert clean_reply_body(text) == "Thanks, that works for me and the team as well."

    def test_leading_bom_then_reply_then_bom_echo(self):
        text = "﻿Sounds good, see you at noon.\n﻿\nOriginal message text here..."
        assert clean_reply_body(text) == "Sounds good, see you at noon."

    def test_keeps_sentence_mentioning_wrote(self):
        # A real sentence containing 'wrote' must survive.
        text = "She wrote: the plan is fine, so I think we should proceed."
        assert clean_reply_body(text) == text

    def test_keeps_handwritten_signoff(self):
        # A personal sign-off is part of writing voice - must survive cleaning.
        text = "See you there.\n\nBest,\nSam\n"
        assert clean_reply_body(text) == "See you there.\n\nBest,\nSam"

    def test_collapses_excess_blank_lines(self):
        text = "First thought.\n\n\n\nSecond thought.\n"
        assert clean_reply_body(text) == "First thought.\n\nSecond thought."


class TestHtmlToText:
    def test_extracts_text_and_entities_from_tags(self):
        html = "<div><p>Hello there,</p><p>See you at 5 &amp; bring notes.</p></div>"
        text = html_to_text(html)
        assert "Hello there," in text
        assert "See you at 5 & bring notes." in text
        assert "<" not in text

    def test_block_tags_separate_lines(self):
        # Adjacent block elements must not concatenate words together.
        text = html_to_text("<p>first</p><p>second</p>")
        assert "firstsecond" not in text
        assert "first" in text and "second" in text

    def test_br_becomes_newline(self):
        assert html_to_text("line one<br>line two").splitlines() == ["line one", "line two"]

    def test_ignores_style_and_script_content(self):
        html = "<style>p{color:red}</style><p>Visible</p><script>alert(1)</script>"
        assert html_to_text(html).strip() == "Visible"


class TestWordCount:
    def test_counts_whitespace_separated_words(self):
        assert word_count("Sounds good, see you Tuesday.") == 5

    def test_empty_and_blank_are_zero(self):
        assert word_count("") == 0
        assert word_count("  \n ") == 0
