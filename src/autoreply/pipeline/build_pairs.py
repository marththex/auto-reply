"""CLI: turn a Takeout mbox export into clean (incoming, reply) JSONL pairs.

Usage: build-pairs path/to/export.mbox [--out data/pairs.jsonl] [--min-words 15]
"""

import argparse
import json
import statistics
from pathlib import Path

from autoreply.pipeline.cleaning import clean_reply_body, word_count
from autoreply.pipeline.mbox import Pair, ParsedMessage, infer_my_email, pair_replies, parse_mbox

_HISTOGRAM_BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, None)]


def build_dataset(pairs: list[Pair], min_words: int = 15) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {
        "pairs_found": len(pairs),
        "kept": 0,
        "filtered_short": 0,
        "filtered_empty_incoming": 0,
        "reply_word_counts": [],
    }
    for pair in pairs:
        incoming_body = clean_reply_body(pair.incoming.body)
        reply_body = clean_reply_body(pair.reply.body)
        if not incoming_body:
            stats["filtered_empty_incoming"] += 1
            continue
        words = word_count(reply_body)
        if words < min_words:
            stats["filtered_short"] += 1
            continue
        stats["kept"] += 1
        stats["reply_word_counts"].append(words)
        records.append({
            "thread_id": pair.reply.thread_key,
            "incoming": {
                "from": pair.incoming.sender,
                "subject": pair.incoming.subject,
                "date": pair.incoming.date.isoformat() if pair.incoming.date else None,
                "body": incoming_body,
            },
            "reply": {
                "date": pair.reply.date.isoformat() if pair.reply.date else None,
                "body": reply_body,
            },
        })
    return records, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (incoming, reply) training pairs from a Gmail Takeout mbox export."
    )
    parser.add_argument("mbox_path", help="Path to the Takeout .mbox file")
    parser.add_argument("--out", default="data/pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--min-words", type=int, default=15,
                        help="Drop replies shorter than this after cleaning")
    parser.add_argument("--my-email", default=None,
                        help="Your address, for exports without X-Gmail-Labels; "
                             "inferred from Sent labels if omitted")
    args = parser.parse_args(argv)

    messages = parse_mbox(args.mbox_path, my_email=args.my_email)
    my_email = args.my_email or infer_my_email(messages)
    records, stats = build_dataset(pair_replies(messages), min_words=args.min_words)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _print_summary(messages, stats, args.min_words, out_path, my_email)


def _print_summary(
    messages: list[ParsedMessage],
    stats: dict,
    min_words: int,
    out_path: Path,
    my_email: str | None,
) -> None:
    sent = sum(1 for m in messages if m.is_sent)
    print(f"Parsed {len(messages)} messages ({sent} sent) as {my_email or '<unknown sender>'}")
    print(f"Pairs found: {stats['pairs_found']}")
    print(f"  kept: {stats['kept']}")
    print(f"  dropped, reply under {min_words} words: {stats['filtered_short']}")
    print(f"  dropped, incoming empty after cleaning: {stats['filtered_empty_incoming']}")

    counts = sorted(stats["reply_word_counts"])
    if counts:
        p90 = counts[min(len(counts) - 1, int(len(counts) * 0.9))]
        print(
            f"Reply length (words): min {counts[0]} / median {int(statistics.median(counts))}"
            f" / p90 {p90} / max {counts[-1]}"
        )
        histogram = []
        for lo, hi in _HISTOGRAM_BUCKETS:
            n = sum(1 for c in counts if c >= lo and (hi is None or c < hi))
            label = f"{lo}-{hi - 1}" if hi else f"{lo}+"
            histogram.append(f"{label}: {n}")
        print("  distribution: " + " | ".join(histogram))

    print(f"Wrote {stats['kept']} pairs -> {out_path}")


if __name__ == "__main__":
    main()
