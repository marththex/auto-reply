"""Behavior of the stratified train/eval split."""

import json

import pytest

from autoreply.training.split import bucket_label, main, split_records


def record(words: int, tag: str) -> dict:
    return {
        "thread_id": tag,
        "incoming": {"from": "a@example.com", "subject": tag, "date": None, "body": "q"},
        "reply": {"date": None, "body": " ".join(["word"] * words)},
    }


def corpus() -> list[dict]:
    """10 records in each of the five length buckets."""
    records = []
    for i, words in enumerate((20, 30, 60, 150, 300)):
        records += [record(words, f"b{i}-{j}") for j in range(10)]
    return records


class TestBucketLabel:
    def test_boundaries(self):
        assert bucket_label(15) == "15-24"
        assert bucket_label(24) == "15-24"
        assert bucket_label(25) == "25-49"
        assert bucket_label(99) == "50-99"
        assert bucket_label(100) == "100-199"
        assert bucket_label(200) == "200+"


class TestSplitRecords:
    def test_every_bucket_represented_in_eval(self):
        train, eval_ = split_records(corpus(), fraction=0.1)
        eval_buckets = {bucket_label(len(r["reply"]["body"].split())) for r in eval_}
        assert eval_buckets == {"15-24", "25-49", "50-99", "100-199", "200+"}
        assert len(eval_) == 5  # 10% of each 10-record bucket

    def test_deterministic_across_calls(self):
        first_train, first_eval = split_records(corpus(), fraction=0.1)
        second_train, second_eval = split_records(corpus(), fraction=0.1)
        assert first_eval == second_eval
        assert first_train == second_train

    def test_disjoint_and_complete(self):
        records = corpus()
        train, eval_ = split_records(records, fraction=0.1)
        train_ids = {r["thread_id"] for r in train}
        eval_ids = {r["thread_id"] for r in eval_}
        assert not train_ids & eval_ids
        assert len(train) + len(eval_) == len(records)

    def test_tiny_bucket_still_holds_out_one(self):
        records = [record(20, f"t{j}") for j in range(3)]
        train, eval_ = split_records(records, fraction=0.07)
        assert len(eval_) == 1
        assert len(train) == 2


class TestCli:
    def write_pairs(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in corpus():
                f.write(json.dumps(r) + "\n")
        return path

    def test_writes_both_files(self, tmp_path):
        pairs = self.write_pairs(tmp_path)
        train, eval_ = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
        main([str(pairs), "--train-out", str(train), "--eval-out", str(eval_)])
        n_train = len(train.read_text(encoding="utf-8").splitlines())
        n_eval = len(eval_.read_text(encoding="utf-8").splitlines())
        assert n_train + n_eval == 50

    def test_refuses_to_overwrite_existing_split(self, tmp_path):
        pairs = self.write_pairs(tmp_path)
        args = [str(pairs), "--train-out", str(tmp_path / "train.jsonl"),
                "--eval-out", str(tmp_path / "eval.jsonl")]
        main(args)
        with pytest.raises(SystemExit):
            main(args)
        main(args + ["--force"])  # explicit opt-in re-randomization
