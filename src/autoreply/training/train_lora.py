"""LoRA fine-tune Gemma 4 E2B on (incoming, reply) pairs with Unsloth.

Run in the training environment (Linux or native Windows; Windows torch
comes from the pinned cu128 index automatically):
    uv sync --group train
    uv run train-lora

VRAM-sensitive knobs (--batch-size, --grad-accum, --max-seq-len) are flags,
not constants: confirm they fit before the first run. Unsloth's own 16 GB
guidance is batch 1 / grad-accum 4.

Heavy imports live inside main() so the rest of the package stays importable
without GPU dependencies.
"""

import argparse
import json
from pathlib import Path

from autoreply.facts import persona_name
from autoreply.training.formatting import to_messages

MODEL_ID = "unsloth/gemma-4-E2B-it"
CHAT_TEMPLATE = "gemma-4"
# Gemma 4 turn markers per Unsloth docs (unsloth.ai/docs/models/gemma-4/train).
# train_on_responses_only masks everything but the user's replies from the loss.
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


def load_records(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Gemma 4 E2B with Unsloth.")
    parser.add_argument("--train", dest="train_path", default="data/train.jsonl")
    parser.add_argument("--eval", dest="eval_path", default="data/eval.jsonl")
    parser.add_argument("--out", default="models/adapter",
                        help="Adapter output dir; per-epoch checkpoints go in <out>/checkpoints")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # unsloth MUST be imported before datasets/trl: it patches SFTTrainer's
    # dataset prep, and the unpatched trl path tokenizes in a way that breaks
    # train_on_responses_only's marker matching (every label masked to -100).
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Resolve the persona name once (from facts.yaml identity.name) so every
    # training prompt addresses the mailbox owner consistently.
    name = persona_name()

    def as_dataset(records: list[dict]) -> Dataset:
        texts = [
            tokenizer.apply_chat_template(to_messages(r, name=name), tokenize=False)
            for r in records
        ]
        return Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # renamed from tokenizer= in newer trl
        train_dataset=as_dataset(load_records(args.train_path)),
        eval_dataset=as_dataset(load_records(args.eval_path)),
        args=SFTConfig(
            output_dir=f"{args.out}/checkpoints",
            dataset_text_field="text",
            max_length=args.max_seq_len,  # renamed from max_seq_length in trl 0.24
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_steps=5,  # warmup_ratio is deprecated
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=None,  # keep every epoch checkpoint for comparison
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    result = trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nFinal train loss: {result.training_loss:.4f}")
    print("Per-epoch eval losses:")
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            print(f"  epoch {entry.get('epoch', '?'):>4}: {entry['eval_loss']:.4f}")
    print(f"Adapter saved to {args.out} (checkpoints in {args.out}/checkpoints)")


if __name__ == "__main__":
    main()
