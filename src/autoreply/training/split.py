"""Stratified train/eval split of pairs.jsonl by reply length.

Deterministic (fixed seed) and refuses to overwrite an existing split, so
eval stays consistent across training runs.

Usage: make-split data/pairs.jsonl [--eval-fraction 0.07] [--force]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
_BUCKETS = [(200, "200+"), (100, "100-199"), (50, "50-99"), (25, "25-49"), (0, "15-24")]


def bucket_label(words: int) -> str:
    for floor, label in _BUCKETS:
        if words >= floor:
            return label
    return _BUCKETS[-1][1]


def split_records(records: list[dict], fraction: float = 0.07) -> tuple[list[dict], list[dict]]:
    """Hold out ~fraction of each length bucket (at least one per bucket)."""
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_bucket[bucket_label(len(r["reply"]["body"].split()))].append(i)

    rng = random.Random(SEED)
    eval_indices: set[int] = set()
    for label in sorted(by_bucket):
        indices = by_bucket[label]
        n_hold = max(1, round(len(indices) * fraction))
        eval_indices.update(rng.sample(indices, n_hold))

    train = [r for i, r in enumerate(records) if i not in eval_indices]
    eval_ = [r for i, r in enumerate(records) if i in eval_indices]
    return train, eval_


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split pairs.jsonl into train/eval, stratified by reply length."
    )
    parser.add_argument("pairs_path", help="Path to pairs.jsonl")
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--eval-out", default="data/eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.07)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split (re-randomizes eval!)")
    args = parser.parse_args(argv)

    train_path, eval_path = Path(args.train_out), Path(args.eval_out)
    if not args.force and (train_path.exists() or eval_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing split ({train_path}, {eval_path}). "
            "Keeping eval consistent across runs; pass --force to re-split."
        )

    records = [
        json.loads(line)
        for line in Path(args.pairs_path).read_text(encoding="utf-8").splitlines()
    ]
    train, eval_ = split_records(records, fraction=args.eval_fraction)

    for path, subset in ((train_path, train), (eval_path, eval_)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Split {len(records)} pairs -> {len(train)} train / {len(eval_)} eval (seed {SEED})")
    for label in [label for _, label in reversed(_BUCKETS)]:
        n_eval = sum(1 for r in eval_ if bucket_label(len(r["reply"]["body"].split())) == label)
        n_all = sum(1 for r in records if bucket_label(len(r["reply"]["body"].split())) == label)
        print(f"  {label:>8} words: {n_all - n_eval} train / {n_eval} eval")
