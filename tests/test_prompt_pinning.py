"""Byte-exact pins of the rendered prompt strings.

The deployed adapter was trained on prompts rendered exactly like the
strings below; any byte of drift puts inference off-distribution. With a
facts identity.name whose first token is "Marcus", training and inference
prompts must render byte-identically to today's production output. Do not
edit the PINNED_* strings to make a refactor pass.
"""

from autoreply.facts import persona_name
from autoreply.training.formatting import to_messages, to_prompt_messages

RECORD = {
    "thread_id": "1",
    "incoming": {
        "from": "alice@example.com",
        "subject": "Lunch?",
        "date": "2026-06-01T10:00:00+00:00",
        "body": "Are you free on Thursday?",
    },
    "reply": {"date": "2026-06-01T11:00:00+00:00", "body": "Thursday works great."},
}

FACTS_BLOCK = "- identity / name: Marcus Chong\n- availability / meetings: weekday evenings"

PINNED_TRAINING_PROMPT = (
    "Write the reply Marcus would send to this email, in his usual voice and length.\n"
    "\n"
    "From: alice@example.com\n"
    "Subject: Lunch?\n"
    "\n"
    "Are you free on Thursday?"
)

PINNED_INFERENCE_PROMPT_WITH_FACTS = (
    "Write the reply Marcus would send to this email, in his usual voice and length.\n"
    "\n"
    "Known facts about Marcus - rely on these and do not invent personal details:\n"
    "- identity / name: Marcus Chong\n"
    "- availability / meetings: weekday evenings\n"
    "\n"
    "From: alice@example.com\n"
    "Subject: Lunch?\n"
    "\n"
    "Are you free on Thursday?"
)


class TestPersonaNameDerivation:
    def test_first_token_of_identity_name(self, tmp_path):
        facts = tmp_path / "facts.yaml"
        facts.write_text("identity:\n  name: Marcus Chong\n", encoding="utf-8")
        assert persona_name(facts) == "Marcus"

    def test_missing_file_returns_none(self, tmp_path):
        assert persona_name(tmp_path / "nope.yaml") is None

    def test_missing_identity_name_returns_none(self, tmp_path):
        facts = tmp_path / "facts.yaml"
        facts.write_text("identity:\n  employer: Acme\n", encoding="utf-8")
        assert persona_name(facts) is None


class TestPinnedPromptBytes:
    def test_training_prompt_is_pinned(self):
        assert to_messages(RECORD, name="Marcus")[0]["content"] == PINNED_TRAINING_PROMPT

    def test_inference_prompt_without_facts_equals_training_prompt(self):
        assert to_prompt_messages(RECORD, name="Marcus")[0]["content"] == PINNED_TRAINING_PROMPT

    def test_inference_prompt_with_facts_is_pinned(self):
        content = to_prompt_messages(RECORD, facts=FACTS_BLOCK, name="Marcus")[0]["content"]
        assert content == PINNED_INFERENCE_PROMPT_WITH_FACTS

    def test_default_name_resolution_reproduces_pinned_bytes(self, tmp_path, monkeypatch):
        # End-to-end proof of the hard constraint: a facts.yaml with
        # identity.name "Marcus Chong" and no explicit name argument must
        # yield exactly the production bytes.
        (tmp_path / "facts.yaml").write_text(
            "identity:\n  name: Marcus Chong\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        assert to_messages(RECORD)[0]["content"] == PINNED_TRAINING_PROMPT


class TestNeutralFallback:
    def test_no_facts_file_falls_back_to_neutral_phrasing(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)  # no facts.yaml here
        with caplog.at_level("WARNING"):
            content = to_messages(RECORD)[0]["content"]
        assert content.startswith(
            "Write the reply the user would send to this email, "
            "in their usual voice and length."
        )
        assert any("facts" in r.message for r in caplog.records)

    def test_grounding_line_uses_neutral_name_without_facts_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = to_prompt_messages(RECORD, facts="- employer: Acme")[0]["content"]
        assert "Known facts about the user - rely on these" in content
