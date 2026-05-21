#!/usr/bin/env python3
"""Merge a PEFT LoRA checkpoint into the base model and save as a standalone
HF directory (full safetensors, no adapter at inference time).

Usage:
    # Argparse form — recommended for scripting
    python merge_peft_checkpoint.py \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 \\
        --checkpoint-dir results/Qwen3-4B-sft-respair-new-3/checkpoint-425 \\
        --output-dir    results/Qwen3-4B-sft-respair-new-3-merged

    # Or just run with the hardcoded defaults below.
    python merge_peft_checkpoint.py
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---- defaults (used when called with no flags) -----------------------------
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CHECKPOINT_DIR = "results/Qwen3-4B-sft-respair-new-3/checkpoint-425"
DEFAULT_OUTPUT_DIR = "results/Qwen3-4B-sft-respair-new-3-merged"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def merge_peft_checkpoint(
    base_model_name: str,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Load base + PEFT adapter, merge_and_unload, save to output_dir."""
    print(f"[merge_peft_checkpoint] device={device}  dtype={dtype}")
    print(f"[merge_peft_checkpoint] base_model={base_model_name}")
    print(f"[merge_peft_checkpoint] checkpoint_dir={checkpoint_dir}")
    print(f"[merge_peft_checkpoint] output_dir={output_dir}")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )

    print("Loading PEFT adapter on top of the base model...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        device_map=device if device == "cuda" else None,
    )

    print("merge_and_unload()...")
    merged_model = model.merge_and_unload()

    print("Loading tokenizer (preferring checkpoint dir, falling back to base)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"  ↳ tokenizer not in checkpoint ({e}); using base model's.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Merge complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model",     default=DEFAULT_BASE_MODEL,
                   help=f"HF model id or local path to the base model. Default: {DEFAULT_BASE_MODEL}")
    p.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                   help=f"Path to the PEFT adapter dir (contains adapter_model.safetensors + "
                        f"adapter_config.json). Default: {DEFAULT_CHECKPOINT_DIR}")
    p.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write the merged standalone model. Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--device",         default=None,
                   help='Device for the merge ("cuda" or "cpu"). Default: auto-detect.')
    p.add_argument("--dtype",          default="float16", choices=["float16", "bfloat16", "float32"],
                   help="Torch dtype for the merged weights. Default: float16.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output dir (skip the existence check).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_device()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    if not os.path.isdir(args.checkpoint_dir):
        print(f"[error] checkpoint dir does not exist: {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(os.path.join(args.checkpoint_dir, "adapter_model.safetensors")) and \
       not os.path.exists(os.path.join(args.checkpoint_dir, "adapter_model.bin")):
        print(f"[error] no adapter_model.{{safetensors,bin}} in {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(2)
    if os.path.isdir(args.output_dir) and not args.force \
       and os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"[skip] {args.output_dir} already looks like a merged model "
              f"(config.json present). Pass --force to redo the merge.")
        return

    merge_peft_checkpoint(
        base_model_name=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
