"""LoRA SFT on Qwen3 for the retrieval-agent task.

Dataset rows must look like {"messages": [ {role, content}, ... ]}.

  python scripts/train_sft.py --model Qwen/Qwen3-1.7B \
      --train data/sft/train.jsonl --val data/sft/val.jsonl \
      --out artifacts/sft_qwen3_1p7b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out", default="artifacts/sft_qwen3")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})
    print("train:", len(ds["train"]), "val:", len(ds["validation"]))
    print("sample keys:", list(ds["train"][0].keys()))

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGETS,
    )
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.model, args=cfg,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        peft_config=lora,
    )
    print("assistant_only_loss =", trainer.args.assistant_only_loss)
    trainer.train(resume_from_checkpoint=args.resume or None)
    out = Path(args.out) / "final"
    trainer.save_model(out)
    trainer.processing_class.save_pretrained(out)
    print("saved to", out)


if __name__ == "__main__":
    main()
