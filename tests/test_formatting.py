"""Behavior of pair -> chat-message formatting shared by training and inference."""

from autoreply.training.formatting import reply_instruction, to_messages, to_prompt_messages

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


def test_training_messages_are_user_then_assistant():
    messages = to_messages(RECORD, name="Sam")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Thursday works great."


def test_user_turn_contains_instruction_and_email_context():
    content = to_messages(RECORD, name="Sam")[0]["content"]
    assert reply_instruction("Sam") in content
    assert "From: alice@example.com" in content
    assert "Subject: Lunch?" in content
    assert "Are you free on Thursday?" in content


def test_missing_subject_is_omitted_not_crashed():
    record = {**RECORD, "incoming": {**RECORD["incoming"], "subject": ""}}
    content = to_messages(record, name="Sam")[0]["content"]
    assert "Subject:" not in content
    assert "Are you free on Thursday?" in content


def test_prompt_messages_for_inference_have_no_assistant_turn():
    messages = to_prompt_messages(RECORD, name="Sam")
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == to_messages(RECORD, name="Sam")[0]["content"]
