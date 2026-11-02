"""Evaluate a model on the retrieval-agent task (greedy, resumable).

Usage:
  python scripts/evaluate.py --model Qwen/Qwen3-1.7B \
      --split data/processed/splits/eval200.jsonl --out artifacts/eval_base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.eval.badcase import annotate, distribution  # noqa: E402
from agentrl.eval.metrics import confusion_2x2, contains_answer, em, f1, summarize  # noqa: E402
from agentrl.eval.runner import Runner  # noqa: E402
from agentrl.retrieval.bm25 import BM25Index  # noqa: E402


def load_model(name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:  # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True)
    return model.to(device).eval(), tokenizer


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="data/processed/splits/eval200.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out", default="artifacts/eval_base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    args = ap.parse_args()

    device = args.device or pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / "traces.jsonl"

    done = set()
    if traces_path.exists():
        for line in traces_path.open(encoding="utf-8"):
            done.add(json.loads(line)["qid"])
        print(f"resume: {len(done)} traces already done")

    questions = [json.loads(l) for l in Path(args.split).open(encoding="utf-8")]
    questions = [q for q in questions if q["qid"] not in done]
    if args.limit:
        questions = questions[: args.limit]

    print(f"loading {args.model} on {device} ...")
    model, tokenizer = load_model(args.model, device)
    index = BM25Index.from_jsonl(args.corpus)
    runner = Runner(model, tokenizer, index, max_turns=args.max_turns,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens, device=device)

    t0 = time.time()
    with traces_path.open("a", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            r = runner.run_one(q["question"])
            trace = {**q, **r}
            trace["em"] = em(r["pred"], q["answer"])
            trace["f1"] = f1(r["pred"], q["answer"])
            trace["em_lenient"] = em(r.get("pred_lenient"), q["answer"])
            trace["f1_lenient"] = f1(r.get("pred_lenient"), q["answer"])
            trace["em_contains"] = contains_answer(r.get("pred_lenient"), q["answer"])
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(questions)}] em={trace['em']:.0f} calls={r['calls']} "
                  f"pred={str(r['pred'])[:40]!r} gold={q['answer'][:40]!r}", flush=True)

    traces = [json.loads(l) for l in traces_path.open(encoding="utf-8")]
    traces = annotate(traces)
    summary = summarize(traces)
    summary["badcase"] = distribution(traces)
    summary["confusion_2x2"] = confusion_2x2(traces)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fails = [t for t in traces if t["em"] == 0]
    for t in fails[:3]:
        print("\n---", t["label"], "|", t["question"][:90])
        print("   pred:", t["pred"], "| gold:", t["answer"], "| queries:", t["queries"])


if __name__ == "__main__":
    main()