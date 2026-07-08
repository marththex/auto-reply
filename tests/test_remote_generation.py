"""Behavior of remote (llama.cpp server) prompt rendering and payloads."""

from autoreply.generation import build_completion_payload, render_gemma4_prompt
from autoreply.training.formatting import to_prompt_messages

RECORD = {
    "incoming": {
        "from": "alice@example.com",
        "subject": "Lunch?",
        "date": None,
        "body": "Are you free Thursday?",
    },
}


def test_renders_gemma4_turn_structure_without_bos():
    prompt = render_gemma4_prompt(to_prompt_messages(RECORD, name="Sam"))
    # llama.cpp adds <bos> during server-side tokenization - never render it.
    assert not prompt.startswith("<bos>")
    assert prompt.startswith("<|turn>user\n")
    assert prompt.endswith("<turn|>\n<|turn>model\n")
    assert "Are you free Thursday?" in prompt


def test_rendered_prompt_matches_training_template_shape():
    # Must stay in lockstep with the shape the adapter was trained on
    # (verified against tokenizer.apply_chat_template output).
    prompt = render_gemma4_prompt([{"role": "user", "content": "X"}])
    assert prompt == "<|turn>user\nX<turn|>\n<|turn>model\n"


def test_completion_payload_shape():
    payload = build_completion_payload("PROMPT", max_new_tokens=700, temperature=0.7)
    assert payload["prompt"] == "PROMPT"
    assert payload["n_predict"] == 700
    assert payload["temperature"] == 0.7
    assert "<turn|>" in payload["stop"]


def test_bounded_body_truncates_long_incoming():
    from autoreply.gmail.bridge import bounded_body

    long_text = "word " * 5000
    result = bounded_body(long_text, limit=6000)
    assert len(result) <= 6000 + 50  # small allowance for the truncation marker
    assert result.endswith("[... truncated ...]")


def test_bounded_body_leaves_short_text_alone():
    from autoreply.gmail.bridge import bounded_body

    assert bounded_body("short email", limit=6000) == "short email"
