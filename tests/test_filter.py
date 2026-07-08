"""Behavior of the automated-sender filter (runs before any model call)."""

from autoreply.gmail.filter import should_skip


def check(sender="alice@example.com", headers=None, subject="Hi", body="Real question?",
          allowlist=()):
    return should_skip(
        sender=sender, headers=headers or {}, subject=subject, body=body,
        allowlist=list(allowlist),
    )


class TestSenderPatterns:
    def test_normal_sender_passes(self):
        result = check(sender="alice@example.com")
        assert not result.skip

    def test_noreply_variants_are_skipped(self):
        for local in ("noreply", "no-reply", "donotreply", "do-not-reply",
                      "notifications", "notification", "alerts", "mailer", "automated"):
            result = check(sender=f"{local}@shop.example.com")
            assert result.skip, local
            assert "sender" in result.reason

    def test_pattern_matches_local_part_prefix_not_substring_elsewhere(self):
        # 'alerts' inside a real name's address must not trigger
        result = check(sender="valerts.person@example.com")
        assert not result.skip

    def test_pattern_matches_delimited_segment_of_local_part(self):
        # Real leak from first inbox run: googleplay-noreply@google.com
        assert check(sender="googleplay-noreply@google.com").skip
        assert check(sender="store.notifications@example.com").skip
        assert check(sender="mailer_daemon@example.com").skip


class TestHeaderDetection:
    def test_list_unsubscribe_skips(self):
        result = check(headers={"List-Unsubscribe": "<mailto:u@x.com>"})
        assert result.skip and "List-Unsubscribe" in result.reason

    def test_precedence_bulk_and_auto_reply_skip(self):
        assert check(headers={"Precedence": "bulk"}).skip
        assert check(headers={"Precedence": "auto_reply"}).skip
        assert not check(headers={"Precedence": "first-class"}).skip

    def test_auto_submitted_skips_unless_no(self):
        assert check(headers={"Auto-Submitted": "auto-generated"}).skip
        assert not check(headers={"Auto-Submitted": "no"}).skip

    def test_header_names_case_insensitive(self):
        assert check(headers={"list-unsubscribe": "<mailto:u@x.com>"}).skip


class TestContentHeuristics:
    def test_automated_message_phrases_skip(self):
        for phrase in ("This is an automated message", "do not reply to this email",
                       "This mailbox is not monitored"):
            assert check(body=f"Hello,\n{phrase}.\nBye").skip, phrase

    def test_phrase_in_subject_also_skips(self):
        assert check(subject="Do not reply to this email").skip


class TestAllowlist:
    def test_allowlisted_address_overrides_all_layers(self):
        result = check(
            sender="noreply@friendlyco.example.com",
            headers={"List-Unsubscribe": "<mailto:u@x.com>"},
            body="This is an automated message",
            allowlist=["noreply@friendlyco.example.com"],
        )
        assert not result.skip

    def test_allowlisted_domain_overrides(self):
        result = check(sender="alerts@friendlyco.example.com",
                       allowlist=["@friendlyco.example.com"])
        assert not result.skip

    def test_allowlist_is_case_insensitive(self):
        result = check(sender="NoReply@FriendlyCo.example.com",
                       allowlist=["noreply@friendlyco.example.com"])
        assert not result.skip


class TestContentScoring:
    """Layer 4: multiple weak marketing signals => skip (threshold 2)."""

    MARKETING_BODY = (
        "VoltHome\n"
        "[https://link.volthome.example.com/ls/click?upn=u001.AAA]\n"
        "With HomeBattery, you can charge your electric car and power your home.\n"
        "Order Yours\n[https://link.volthome.example.com/ls/click?upn=u001.BBB]\n"
        "Learn More\n[https://link.volthome.example.com/ls/click?upn=u001.CCC]\n"
        "Watch Now\n[https://link.volthome.example.com/ls/click?upn=u001.DDD]\n"
        "Unsubscribe [https://link.volthome.example.com/ls/click?upn=u001.EEE]\n"
        "Do not reply - this email address is not actively monitored.\n"
    )

    def test_marketing_structure_regression_is_skipped(self):
        # Real leak: marketing mail with no bulk headers at all.
        result = check(sender="energy@volthome.example.com",
                       subject="Maximize Your Home Energy Savings",
                       body=self.MARKETING_BODY)
        assert result.skip
        assert "content score" in result.reason

    def test_skip_reason_names_contributing_signals(self):
        result = check(sender="energy@volthome.example.com", subject="Big Sale Ends Soon",
                       body=self.MARKETING_BODY)
        assert result.skip
        for named in ("unsubscribe-text", "tracker-links"):
            assert named in result.reason

    def test_single_weak_signal_does_not_skip(self):
        # A friend writing the word 'unsubscribe' must not trigger alone.
        result = check(body="Ha, I finally hit unsubscribe on that list you mentioned.")
        assert not result.skip

    def test_one_tracked_link_plus_chatty_text_does_not_skip(self):
        result = check(body=(
            "Check out this article: https://news.example.com/story?utm_source=share\n"
            "Thought of you when I read it. Lunch next week?"
        ))
        assert not result.skip

    def test_marketing_subdomain_plus_sale_subject_skips(self):
        result = check(sender="offers@mkt.shopco.example.com",
                       subject="48 hours only: 30% off everything",
                       body="Our biggest event of the season.")
        assert result.skip
        assert "content score" in result.reason
