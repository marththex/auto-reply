"""Turn (incoming, reply) records into chat messages.

Single source of truth for the prompt shape: training, baseline inference,
and the draft-creation flow must all format emails identically. The persona
name comes from the facts file (identity.name, first token); callers that
render many prompts resolve it once via facts.persona_name and pass it in.
Prompt bytes for a given name are pinned by tests/test_prompt_pinning.py.
"""

import logging

from autoreply.facts import DEFAULT_FACTS_PATH, persona_name

log = logging.getLogger(__name__)


def reply_instruction(name: str | None) -> str:
    if not name:
        return (
            "Write the reply the user would send to this email, "
            "in their usual voice and length."
        )
    return f"Write the reply {name} would send to this email, in his usual voice and length."


def to_messages(record: dict, *, name: str | None = None) -> list[dict]:
    """Training example: incoming email as user turn, reply as assistant turn."""
    return [
        {"role": "user", "content": _user_prompt(record, name=name)},
        {"role": "assistant", "content": record["reply"]["body"]},
    ]


def to_prompt_messages(record: dict, facts: str = "", *, name: str | None = None) -> list[dict]:
    """Inference prompt: user turn only.

    Without facts, identical to the training prompt shape. With facts, a
    grounding block is inserted - inference-only on purpose: the adapter was
    trained without it, and retraining with facts baked in is a future step.
    """
    return [{"role": "user", "content": _user_prompt(record, facts, name=name)}]


def _default_name() -> str | None:
    name = persona_name()
    if name is None:
        log.warning(
            "no identity.name in %s - prompts will address 'the user'", DEFAULT_FACTS_PATH
        )
    return name


def _user_prompt(record: dict, facts: str = "", *, name: str | None = None) -> str:
    if name is None:
        name = _default_name()
    incoming = record["incoming"]
    headers = [
        f"{label}: {incoming[key]}"
        for label, key in (("From", "from"), ("Subject", "subject"))
        if incoming.get(key)
    ]
    grounding = (
        [
            f"Known facts about {name or 'the user'} - "
            "rely on these and do not invent personal details:",
            facts,
            "",
        ]
        if facts
        else []
    )
    return "\n".join([reply_instruction(name), "", *grounding, *headers, "", incoming["body"]])
