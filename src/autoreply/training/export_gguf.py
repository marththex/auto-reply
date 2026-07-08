"""Export the LoRA adapter to GGUF for llama.cpp (text-only NAS inference).

Re-run after every retrain:

    uv run export-gguf --name my-lora-vN-epochM \\
        --source-checkpoint checkpoints/checkpoint-XXX --eval-loss X.XXX

Steps: (1) filter the PEFT adapter to language-model tensors - the
audio/vision tower LoRA weights are dead weight for email drafting and break
text-only conversion - renaming to the text-model layout; (2) run llama.cpp's
convert_lora_to_gguf.py on the filtered copy; (3) write metadata.json
recording provenance next to the output.

Needs a llama.cpp checkout (LLAMA_CPP_DIR env var or --llama-cpp) and the
train dependency group (torch + gguf).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ADAPTER = "models/adapter"
DEFAULT_BASE = "unsloth/gemma-4-E2B-it"
# Where the GGUF + metadata land; point AUTOREPLY_MODELS_DIR at the volume
# your inference host mounts (e.g. a share the llama.cpp box can read).
DEFAULT_OUT_DIR = os.environ.get("AUTOREPLY_MODELS_DIR", "models/gguf")
TEXT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]

_LANGUAGE_SEGMENT = ".language_model."


def keep_tensor(name: str) -> bool:
    """Only language-model LoRA tensors survive a text-only export."""
    return _LANGUAGE_SEGMENT in name


def rename_tensor(name: str) -> str:
    """Map multimodal naming to the text-model layout llama.cpp expects."""
    return name.replace(_LANGUAGE_SEGMENT, ".", 1)


def build_metadata(*, adapter_name: str, source_checkpoint: str,
                   eval_loss: float | None, base_gguf: str) -> dict:
    return {
        "adapter_name": adapter_name,
        "source_checkpoint": source_checkpoint,
        "eval_loss": eval_loss,
        "base_gguf": base_gguf,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export LoRA adapter to GGUF for llama.cpp.")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--name", required=True,
                        help="Output name, e.g. my-lora-v2-epoch3")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="HF id/dir of the unquantized base (architecture reference)")
    parser.add_argument("--base-gguf", default="gemma-4-e2b-base-q5.gguf",
                        help="Base GGUF filename recorded in metadata")
    parser.add_argument("--source-checkpoint", default="", help="For metadata")
    parser.add_argument("--eval-loss", type=float, default=None, help="For metadata")
    parser.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR"),
                        help="Path to a llama.cpp checkout (or set LLAMA_CPP_DIR)")
    args = parser.parse_args(argv)

    if not args.llama_cpp or not Path(args.llama_cpp, "convert_lora_to_gguf.py").exists():
        raise SystemExit(
            "llama.cpp checkout not found. Clone it and pass --llama-cpp or set "
            "LLAMA_CPP_DIR:  git clone --depth 1 https://github.com/ggml-org/llama.cpp"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gguf = out_dir / f"{args.name}.gguf"

    with tempfile.TemporaryDirectory() as tmp:
        filtered_dir = _write_filtered_adapter(Path(args.adapter), Path(tmp), args.base)
        _run_converter(args.llama_cpp, filtered_dir, out_gguf, args.base)

    meta = build_metadata(
        adapter_name=args.name,
        source_checkpoint=args.source_checkpoint,
        eval_loss=args.eval_loss,
        base_gguf=args.base_gguf,
    )
    meta_path = out_dir / f"{args.name}.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {out_gguf} ({out_gguf.stat().st_size / 1e6:.0f} MB)")
    print(f"Metadata -> {meta_path}")


def _write_filtered_adapter(adapter_dir: Path, tmp: Path, base: str) -> Path:
    """Copy of the adapter with only language-model tensors, text-style names."""
    import torch  # noqa: F401  (safetensors torch backend)
    from safetensors.torch import load_file, save_file

    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    kept = {rename_tensor(k): v for k, v in tensors.items() if keep_tensor(k)}
    dropped = len(tensors) - len(kept)
    print(f"Filtered adapter: kept {len(kept)} language-model tensors, "
          f"dropped {dropped} tower/embed tensors")

    filtered = tmp / "filtered-adapter"
    filtered.mkdir()
    save_file(kept, filtered / "adapter_model.safetensors")

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = TEXT_TARGET_MODULES
    config["base_model_name_or_path"] = base
    (filtered / "adapter_config.json").write_text(json.dumps(config, indent=2),
                                                  encoding="utf-8")
    for extra in ("tokenizer.json", "tokenizer_config.json"):
        if (adapter_dir / extra).exists():
            shutil.copy2(adapter_dir / extra, filtered / extra)
    return filtered


def _run_converter(llama_cpp: str, adapter_dir: Path, out_gguf: Path, base: str) -> None:
    cmd = [
        sys.executable, str(Path(llama_cpp) / "convert_lora_to_gguf.py"),
        "--base-model-id", base,
        "--outfile", str(out_gguf),
        "--outtype", "f16",
        str(adapter_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
