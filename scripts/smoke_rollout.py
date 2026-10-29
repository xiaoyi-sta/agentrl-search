"""One-step GRPO smoke test on a tiny model (local M4, no GPU needed)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from agentrl.env.search_env import SearchEnv
from agentrl.reward.rewards import answer_reward, budget_reward, format_reward

MODEL = "models/Qwen3-0.6B"
QUESTIONS = ROOT / "data/processed/splits/eval200.jsonl"

SYSTEM = (
    "You are a research agent. Use the search tool when you need facts. "
    "You may call tools several times. When you are confident, reply with "
    "<answer>your short answer</answer> and nothing else."
)


def build_dataset(n: int = 2) -> Dataset:
    rows = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")][:n]
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                ]
                for r in rows
            ],
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "supporting_titles": [r["supporting_titles"] for r in rows],
        }
    )


def main() -> None:
    trainer = GRPOTrainer(
        model=MODEL,
        args=GRPOConfig(
            output_dir="artifacts/smoke_grpo",
            max_steps=1,
            num_generations=2,
            per_device_train_batch_size=2,
            max_completion_length=256,
            max_tool_calling_iterations=3,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=2,
            report_to="none",
            use_vllm=False,
            chat_template_kwargs={"enable_thinking": False},
        ),
        train_dataset=build_dataset(),
        environment_factory=SearchEnv,
        reward_funcs=[answer_reward, format_reward, budget_reward],
    )
    trainer.train()


if __name__ == "__main__":
    main()