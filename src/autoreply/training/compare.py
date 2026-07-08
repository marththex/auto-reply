"""Side-by-side eval: base Gemma 4 E2B vs fine-tuned checkpoint vs actual reply.

Generates replies for every eval.jsonl prompt with both models and writes a
markdown file for manual quality judgment. Loads the two models sequentially,
freeing VRAM in between, so it fits alongside nothing else on a 16 GB card.
"""

import argparse
from pathlib import Path

from autoreply.facts import DEFAULT_FACTS_PATH, load_facts, persona_name
from autoreply.training.train_lora import MODEL_ID, load_records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned replies on eval set.")
    parser.add_argument("--adapter", default="models/adapter",
                        help="Adapter dir or a specific epoch checkpoint under .../checkpoints")
    parser.add_argument("--eval", dest="eval_path", default="data/eval.jsonl")
    parser.add_argument("--out", default="data/comparison.md")
    parser.add_argument("--max-new-tokens", type=int, default=700,
                        help="Covers p90 reply length (~490 words)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--facts", default=None,
                        help="Path to facts.yaml; injects grounding into both models' prompts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only compare the first N eval records (quick sanity checks)")
    args = parser.parse_args(argv)

    args.facts_text = load_facts(args.facts) if args.facts else ""
    # Name resolution is independent of --facts: prompts must stay in the
    # training distribution ("Write the reply <name> ...") even when the
    # grounding block is off.
    args.persona = persona_name(args.facts or DEFAULT_FACTS_PATH)
    records = load_records(args.eval_path)
    if args.limit:
        records = records[: args.limit]
    base_replies = _generate_all(MODEL_ID, records, args)
    tuned_replies = _generate_all(args.adapter, records, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Base vs fine-tuned ({args.adapter}) on {len(records)} eval pairs\n")
        for i, record in enumerate(records):
            incoming = record["incoming"]
            f.write(f"\n---\n\n## {i + 1}. {incoming.get('subject') or '(no subject)'}\n\n")
            f.write(f"**Incoming** (from {incoming.get('from', '?')}):\n\n")
            f.write(_quote(incoming["body"]))
            f.write("\n\n### Base model\n\n" + _quote(base_replies[i]))
            f.write("\n\n### Fine-tuned\n\n" + _quote(tuned_replies[i]))
            f.write("\n\n### Actual reply\n\n" + _quote(record["reply"]["body"]) + "\n")
    print(f"Wrote {len(records)} comparisons -> {out_path}")


def _generate_all(model_name: str, records: list[dict], args) -> list[str]:
    from autoreply.generation import free_vram, generate_reply, load_model

    model, tokenizer = load_model(model_name, max_seq_len=args.max_seq_len)
    replies = []
    for record in records:
        replies.append(generate_reply(
            model, tokenizer, record, facts=args.facts_text, name=args.persona,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        ))
        print(f"  [{model_name}] {len(replies)}/{len(records)}")
    del model, tokenizer
    free_vram()
    return replies


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines())


if __name__ == "__main__":
    main()
