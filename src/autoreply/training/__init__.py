"""LoRA fine-tuning of Gemma 4 E2B on the (incoming, reply) pairs.

split.py       - stratified train/eval split (make-split)
formatting.py  - pair -> chat messages, shared by training and inference
train_lora.py  - Unsloth LoRA training with per-epoch checkpoints (train-lora)
compare.py     - base vs fine-tuned vs actual on eval set (compare-replies)
"""
