"""Stage 1 gate: BM25 Recall@k on the eval split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.retrieval.bm25 import BM25Index, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--questions", default="data/processed/splits/eval200.jsonl")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="artifacts/retrieval_report.json")
    args = ap.parse_args()

    index = BM25Index.from_jsonl(args.corpus)
    questions = [json.loads(line) for line in Path(args.questions).open(encoding="utf-8")]
    if args.sample:
        questions = questions[: args.sample]

    report = {"corpus_docs": len(index.docs), "n_questions": len(questions)}
    for k in (1, 3, 5):
        r = evaluate(index, questions, k=k)
        report[f"k={k}"] = {kk: vv for kk, vv in r.items() if kk != "misses"}
        if k == 3:
            report["misses"] = r["misses"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "misses"}, ensure_ascii=False, indent=2))
    for m in report.get("misses", [])[:3]:
        print("MISS:", m["question"][:80], "| 缺:", m["missing"])


if __name__ == "__main__":
    main()
