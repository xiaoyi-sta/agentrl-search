"""Validate generated SFT trajectories by replaying them in the training env.

Checks, in order:
  1. dedup by qid
  2. final assistant turn has <answer>
  3. type B has >= 2 tool calls
  4. answer matches gold (A/B) or is an abstain phrase (C)
  5. replayed observations equal the recorded ones (env / corpus drift detector)
  6. length budget

Usage:
  python scripts/validate_sft.py --inputs data/sft/raw_a.jsonl data/sft/raw_b.jsonl data/sft/raw_c.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.env.search_env import SearchEnv  # noqa: E402
from agentrl.eval.metrics import em  # noqa: E402
from agentrl.eval.runner import parse_answer, parse_tool_call  # noqa: E402

ABSTAIN = ("unknown", "not found", "cannot find", "no information", "not mentioned")


def execute(env: SearchEnv, name: str, args: dict) -> str:
    if name == "search":
        return env.search(str(args.get("query", "")))
    if name == "calculator":
        return env.calculator(str(args.get("expression", "")))
    return f"unknown tool: {name}"


def replay(msgs: list[dict], item: dict) -> tuple[bool, int, str]:
    env = SearchEnv()
    env.reset(question=item.get("question", ""), answer=item.get("answer", ""))
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        call = parse_tool_call(m["content"])
        if call is None:
            continue
        obs = execute(env, *call)
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if not nxt or nxt["role"] != "tool":
            return False, env.calls, "missing_observation"
        if nxt["content"] != obs:
            return False, env.calls, "observation_drift"
    return True, env.calls, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()

    kept, drop, seen = [], {}, set()
    for path in args.inputs:
        for line in open(path):
            r = json.loads(line)
            if r["qid"] in seen:
                drop["dup"] = drop.get("dup", 0) + 1
                continue
            msgs = r["messages"]
            ans = parse_answer(msgs[-1]["content"])
            if not ans:
                drop["no_answer_tag"] = drop.get("no_answer_tag", 0) + 1
                continue
            ok, calls, why = replay(msgs, r)
            if not ok:
                drop[why] = drop.get(why, 0) + 1
                continue
            if r["type"] == "B" and calls < 2:
                drop["one_hop"] = drop.get("one_hop", 0) + 1
                continue
            if r["type"] in ("A", "B") and em(ans, r["answer"]) < 1.0:
                drop["wrong_answer"] = drop.get("wrong_answer", 0) + 1
                continue
            if r["type"] == "C" and not any(k in ans.lower() for k in ABSTAIN):
                drop["not_abstain"] = drop.get("not_abstain", 0) + 1
                continue
            if sum(len(m["content"]) for m in msgs) > 9000:
                drop["too_long"] = drop.get("too_long", 0) + 1
                continue
            seen.add(r["qid"])
            kept.append({"messages": msgs, "meta": {"qid": r["qid"], "type": r["type"], "calls": calls}})

    random.Random(0).shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio)) if kept else 0
    for path, part in ((args.val, kept[:n_val]), (args.train, kept[n_val:])):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"kept={len(kept)} train={len(kept)-n_val} val={n_val}")
    print("dropped:", drop)
    if kept:
        avg_calls = sum(x["meta"]["calls"] for x in kept) / len(kept)
        print(f"avg_calls={avg_calls:.2f}")


if __name__ == "__main__":
    main()
