"""Behavior of the automated-sender filter (runs before any model call)."""

from autoreply.gmail.filter import should_skip


def check(sender="alice@example.com", headers=None, subject="Hi",
          body="Real question - do you have time to look at the draft this week?",
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

    def test_do_not_respond_wording_skips(self):
        # Real leak: a satisfaction survey said "respond", not "reply", and
        # passed every layer.
        result = check(
            sender="surveys@insureco.example.com",
            subject="How did we do?",
            body="We value your feedback.\nPlease do not respond to this email.",
        )
        assert result.skip


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


class TestMinimumContent:
    """Near-empty bodies are undraftable: image-only marketing HTML reduces
    to footer boilerplate, and generation with no conditioning content
    parrots the facts block instead of replying to anything."""

    def test_near_empty_body_is_skipped(self):
        result = check(sender="someone@example.com", subject="(no subject)", body="Hi.")
        assert result.skip
        assert "insufficient content" in result.reason

    def test_empty_body_is_skipped(self):
        assert check(body="").skip

    def test_short_but_real_question_passes(self):
        result = check(body="Are we still on for lunch tomorrow at noon? Let me know either way.")
        assert not result.skip

    def test_allowlisted_sender_exempt_from_content_minimum(self):
        result = check(sender="friend@example.com", body="Hi.",
                       allowlist=["friend@example.com"])
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

    def test_esp_greeting_subdomain_plus_unsubscribe_skips(self):
        # Real leak: newsletter sent from a hello.<brand> ESP subdomain with
        # no bulk headers; "hello" wasn't in the subdomain vocabulary.
        result = check(sender="beanroast@hello.beanroast.example.com",
                       subject="Your summer menu is here",
                       body="Try the new oat latte.\nUnsubscribe | View in browser")
        assert result.skip
        assert "marketing-subdomain" in result.reason

    def test_clicks_link_domain_counts_toward_tracker_links(self):
        # Real leak follow-up: the ESP used clicks.<brand> link domains; the
        # tracker regex knew 'link(s)' and 'click' but not 'clicks'.
        result = check(
            sender="team@corporate.example.com",
            subject="New drinks this week",
            body=(
                "New drinks land in stores this week.\n"
                "SHOP ( https://clicks.brand.example.com/f/a/AAA )\n"
                "FIND US ( https://clicks.brand.example.com/f/a/BBB )\n"
                "Instagram ( https://clicks.brand.example.com/f/a/CCC )\n"
                "Unsubscribe ( https://clicks.brand.example.com/f/a/DDD )\n"
            ),
        )
        assert result.skip
        assert "tracker-links" in result.reason

    def test_soft_do_not_respond_variant_counts_as_signal(self):
        # "respond" must count for the soft signal too, not only the exact
        # hard-skip phrase.
        result = check(body=(
            "Please do not respond directly to this message.\n"
            "To opt-out of future surveys, unsubscribe here."
        ))
        assert result.skip
        assert "do-not-reply-text" in result.reason
