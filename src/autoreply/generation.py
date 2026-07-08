"""Shared reply generation: local GPU (unsloth) or remote llama.cpp server.

Heavy imports live inside functions; the module is importable without GPU
dependencies installed. Remote mode needs only the stdlib.
"""

import gc
import json
import urllib.request

# Gemma 4 turn markers. Kept in lockstep with the tokenizer's chat template
# the adapter was trained on (pinned by tests/test_remote_generation.py).
# No <bos>: llama.cpp adds it during server-side tokenization.
_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"


def render_gemma4_prompt(messages: list[dict]) -> str:
    """Pure-Python rendering of the Gemma 4 chat template (no tokenizer dep)."""
    parts = [
        f"{_TURN_OPEN}{m['role']}\n{m['content']}{_TURN_CLOSE}\n" for m in messages
    ]
    return "".join(parts) + f"{_TURN_OPEN}model\n"


def build_completion_payload(prompt: str, *, max_new_tokens: int,
                             temperature: float) -> dict:
    return {
        "prompt": prompt,
        "n_predict": max_new_tokens,
        "temperature": temperature,
        "stop": [_TURN_CLOSE],
        "cache_prompt": False,
    }


def generate_reply_remote(endpoint: str, record: dict, *, facts: str = "",
                          name: str | None = None,
                          max_new_tokens: int = 700, temperature: float = 0.7,
                          timeout: float = 600.0) -> str:
    """Generate via a llama.cpp server's /completion endpoint."""
    from autoreply.training.formatting import to_prompt_messages

    prompt = render_gemma4_prompt(to_prompt_messages(record, facts=facts, name=name))
    payload = build_completion_payload(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"llama.cpp server returned {e.code}: {detail}") from e
    return body["content"].strip()


def load_model(name_or_adapter: str, max_seq_len: int = 4096):
    """Load base model or LoRA adapter dir, ready for inference."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name_or_adapter,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_reply(model, tokenizer, record: dict, *, facts: str = "",
                   name: str | None = None,
                   max_new_tokens: int = 700, temperature: float = 0.7) -> str:
    from autoreply.training.formatting import to_prompt_messages

    # Render to text, then tokenize the string: tokenizing inside
    # apply_chat_template breaks on Gemma4Processor (expects multimodal
    # content parts). The rendered text already starts with <bos>.
    prompt_text = tokenizer.apply_chat_template(
        to_prompt_messages(record, facts=facts, name=name),
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        text=prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        # Unsloth Gemma 4 docs: use_cache=False corrupts attention on E2B.
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def free_vram() -> None:
    """Release cached VRAM so a second model can load on a 16 GB card.

    The caller must drop its own model/tokenizer references first (`del` or
    reassignment) - deleting them inside a helper only unbinds the helper's
    parameters and frees nothing.
    """
    import torch

    gc.collect()
    torch.cuda.empty_cache()
