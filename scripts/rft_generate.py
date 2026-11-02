"""Rejection sampling with plain HF generation (no vLLM needed).

For each question, sample K full trajectories with the SFT model, then keep
the ones that end with a correct <answer>. Output schema matches
data/sft/train.jsonl, so validate_sft.py can filter it directly.

  python scripts/rft_generate.py --model /root/autodl-tmp/agentrl-search/artifacts/sft_1p7b/merged \
      --out data/rft/raw.jsonl --k 3 --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.env.search_env import SearchEnv
from agentrl.eval.runner import parse_answer, parse_tool_call
from agentrl.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS


def execute(env, name, args):
    if name == "search":
        return env.search(str(args.get("query", "")))
    if name == "calculator":
        return env.calculator(str(args.get("expression", "")))
    return f"unknown tool: {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default="data/processed/splits/rl1200.jsonl")
    ap.add_argument("--out", default="data/rft/raw.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    qs = [json.loads(l) for l in open(args.questions)][: args.limit]
    episodes = []
    for q in qs:
        for j in range(args.k):
            episodes.append({"q": q, "branch": j,
                             "msgs": [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": q["question"]}],
                             "env": None, "done": False})

    for turn in range(args.max_turns):
        active = [e for e in episodes if not e["done"]]
        if not active:
            break
        for i in range(0, len(active), args.batch):
            chunk = active[i:i + args.batch]
            prompts = [tok.apply_chat_template(
                e["msgs"], tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True,
                enable_thinking=False) for e in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=4096).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=args.temperature,
                                     top_p=0.95, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, text in zip(chunk, texts):
                text = text.strip()
                e["msgs"].append({"role": "assistant", "content": text})
                call = parse_tool_call(text)
                if call is None:
                    e["done"] = True
                    continue
                if e["env"] is None:
                    e["env"] = SearchEnv()
                    e["env"].reset(question=e["q"]["question"])
                e["msgs"].append({"role": "tool", "content": execute(e["env"], *call)})
        print(f"turn {turn}: {len(active)} active", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(args.out, "w") as f:
        for e in episodes:
            ans = parse_answer(e["msgs"][-1]["content"])
            f.write(json.dumps({
                "qid": f'{e["q"]["qid"]}_{e["branch"]}',
                "type": "B", "question": e["q"]["question"], "answer": e["q"]["answer"],
                "messages": e["msgs"], "calls": e["env"].calls if e["env"] else 0,
            }, ensure_ascii=False) + "\n")
            kept += 1
    print("wrote", kept, "trajectories to", args.out)


if __name__ == "__main__":
    main()
