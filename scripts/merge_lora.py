"""Merge a LoRA adapter into the base model so evaluate.py can load it.

  python scripts/merge_lora.py --base Qwen/Qwen3-1.7B \
      --adapter artifacts/sft_qwen3_1p7b/final --out artifacts/sft_qwen3_1p7b/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
