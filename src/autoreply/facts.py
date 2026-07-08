"""Personal facts injected into draft-generation prompts.

The model invents plausible personal details when it lacks context (confirmed
failure mode), so inference prompts carry a human-maintained facts block.
Lives in facts.yaml at the repo root - gitignored, personal data. Copy
facts.example.yaml to get started.
"""

from pathlib import Path

import yaml

DEFAULT_FACTS_PATH = Path("facts.yaml")


def load_facts(path: str | Path = DEFAULT_FACTS_PATH) -> str:
    """Rendered facts block, or empty string if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return ""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return render_facts(data) if data else ""


def persona_name(path: str | Path = DEFAULT_FACTS_PATH) -> str | None:
    """First whitespace token of identity.name, or None if unavailable.

    Prompts address the mailbox owner by first name; it comes from the same
    facts file that grounds inference so there is a single identity source.
    """
    path = Path(path)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    identity = data.get("identity")
    if not isinstance(identity, dict):
        return None
    name = str(identity.get("name") or "").strip()
    return name.split()[0] if name else None


def render_facts(data) -> str:
    """Flatten nested YAML into '- key / subkey: value' bullet lines."""
    return "\n".join(_flatten(data, []))


def _flatten(node, keys: list[str]) -> list[str]:
    if isinstance(node, dict):
        lines = []
        for key, value in node.items():
            lines += _flatten(value, keys + [str(key)])
        return lines
    if isinstance(node, list):
        lines = []
        for item in node:
            lines += _flatten(item, keys)
        return lines
    prefix = " / ".join(keys)
    return [f"- {prefix}: {node}" if prefix else f"- {node}"]
