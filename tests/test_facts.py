"""Behavior of personal-facts loading and prompt injection."""

from autoreply.facts import load_facts, render_facts
from autoreply.training.formatting import to_messages, to_prompt_messages

RECORD = {
    "thread_id": "1",
    "incoming": {
        "from": "recruiter@example.com",
        "subject": "Opportunity",
        "date": None,
        "body": "Are you open to new roles?",
    },
    "reply": {"date": None, "body": "No thanks."},
}


class TestRenderFacts:
    def test_flattens_nested_mapping_to_bullet_lines(self):
        data = {
            "identity": {"employer": "Acme Corp", "role": "Engineer"},
            "availability": "Weekday evenings after 6pm",
        }
        text = render_facts(data)
        assert "- identity / employer: Acme Corp" in text
        assert "- identity / role: Engineer" in text
        assert "- availability: Weekday evenings after 6pm" in text

    def test_renders_lists_as_separate_bullets(self):
        data = {"relationships": [{"name": "Dana", "relation": "sister"}, "cousin Ben at Acme"]}
        text = render_facts(data)
        assert "Dana" in text
        assert "sister" in text
        assert "cousin Ben at Acme" in text


class TestLoadFacts:
    def test_missing_file_returns_empty_string(self, tmp_path):
        assert load_facts(tmp_path / "nope.yaml") == ""

    def test_loads_and_renders_yaml(self, tmp_path):
        path = tmp_path / "facts.yaml"
        path.write_text("identity:\n  employer: Acme Corp\n", encoding="utf-8")
        assert "Acme Corp" in load_facts(path)


class TestPromptInjection:
    def test_inference_prompt_includes_facts_and_grounding_instruction(self):
        content = to_prompt_messages(RECORD, facts="- employer: Acme Corp", name="Sam")[0]["content"]
        assert "- employer: Acme Corp" in content
        assert "do not invent" in content.lower()
        assert "Are you open to new roles?" in content

    def test_inference_prompt_without_facts_is_unchanged_shape(self):
        content = to_prompt_messages(RECORD, name="Sam")[0]["content"]
        assert "do not invent" not in content.lower()
        assert "Are you open to new roles?" in content

    def test_training_messages_never_include_facts(self):
        # The adapter was trained without a facts block; to_messages must
        # keep producing exactly that shape.
        content = to_messages(RECORD, name="Sam")[0]["content"]
        assert "do not invent" not in content.lower()
