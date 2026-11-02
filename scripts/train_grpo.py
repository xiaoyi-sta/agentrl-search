"""GRPO on top of the SFT model: online RL with verifiable rewards.

  # sanity (20 steps, ~10 min)
  python scripts/train_grpo.py --model artifacts/sft_1p7b/merged --steps 20 \
      --out artifacts/grpo_smoke

  # full run (120 steps)
  python scripts/train_grpo.py --model artifacts/sft_1p7b/merged --steps 120 \
      --out artifacts/grpo_1p7b
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from agentrl.env.search_env import SearchEnv
from agentrl.prompts import SYSTEM_PROMPT
from agentrl.reward.rewards import answer_reward, budget_reward, format_reward

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_dataset(path: str, limit: int) -> Dataset:
    rows = [json.loads(line) for line in open(path)]
    if limit:
        rows = rows[:limit]
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": r["question"]}]
                for r in rows
            ],
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "supporting_titles": [r.get("supporting_titles", []) for r in rows],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/sft_1p7b/merged")
    ap.add_argument("--data", default="data/processed/splits/rl1200.jsonl")
    ap.add_argument("--out", default="artifacts/grpo_1p7b")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient")
    ap.add_argument("--lam", type=float, default=0.1, help="tool-budget penalty")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-completions", action="store_true", help="print sampled completions")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--debug-completions", action="store_true")
    ap.add_argument("--mask-truncated", action="store_true",
                    help="drop completions that hit the length limit")
    ap.add_argument("--loss-type", default="", help="e.g. dr_grpo")
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    args = ap.parse_args()

    ds = build_dataset(args.data, args.limit)
    print("train prompts:", len(ds))

    if args.dry_run:
        print("prompt[0]:", ds[0]["prompt"][1]["content"][:120])
        return

    kwargs = dict(
        output_dir=args.out,
        max_steps=args.steps,
        num_generations=args.generations,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion,
        max_tool_calling_iterations=args.max_turns,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        mask_truncated_completions=args.mask_truncated,
        entropy_coef=args.entropy_coef,
        log_completions=args.log_completions,
        num_completions_to_print=2 if args.log_completions else None,
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        chat_template_kwargs={"enable_thinking": False},
    )
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = [k for k in kwargs if k not in valid]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print("ignored GRPOConfig args:", dropped)
    if args.loss_type and "loss_type" in valid:
        kwargs["loss_type"] = args.loss_type

    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGETS)

    # TRL logs reward functions by name, so a named closure is safer than partial()
    def budget_with_lam(completions, environments=None, **kw):
        return budget_reward(completions, environments=environments,
                             max_calls=args.max_turns, lam=args.lam, **kw)

    _dbg = {"n": 0}

    def debug_answer(completions, answer=None, **kw):
        if args.debug_completions and _dbg["n"] < 2:
            _dbg["n"] += 1
            for i, c in enumerate(completions[:2]):
                print(f"=== COMPLETION {_dbg['n']}.{i} ===", flush=True)
                if isinstance(c, str):
                    print(f"len={len(c)} tail={c[-500:]!r}", flush=True)
                else:
                    msgs = [m for m in c if isinstance(m, dict)]
                    print(f"n_msgs={len(msgs)} roles={[m.get('role') for m in msgs]}", flush=True)
                    for m in msgs[-3:]:
                        print(f"  {m.get('role')}: {str(m.get('content'))[-400:]!r}", flush=True)
        return answer_reward(completions, answer=answer, **kw)

    trainer = GRPOTrainer(
        model=args.model,
        args=GRPOConfig(**kwargs),
        train_dataset=ds,
        environment_factory=SearchEnv,
        reward_funcs=[
            debug_answer if args.debug_completions else answer_reward,
            format_reward,
            budget_with_lam,
        ],
        peft_config=lora,
    )
    print("use_vllm =", getattr(trainer.args, "use_vllm", None),
          "| generations =", trainer.args.num_generations)
    trainer.train()
    out = Path(args.out) / "final"
    trainer.save_model(out)
    trainer.processing_class.save_pretrained(out)
    print("saved to", out)


if __name__ == "__main__":
    main()
