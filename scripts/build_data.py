"""Stage 1: parquet -> corpus.jsonl + questions.jsonl + splits + stats.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.data.build import (  # noqa: E402
    build_corpus,
    load_rows,
    make_splits,
    stats,
    to_question_record,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/hotpot_dev.parquet")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--rl", type=int, default=1200)
    ap.add_argument("--sft", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    rows = load_rows(args.parquet)
    corpus = build_corpus(rows)
    splits = make_splits(rows, args.eval, args.rl, args.sft)
    clean, ev, rl, sft = splits

    n = write_jsonl(
        out / "corpus.jsonl",
        ({"doc_id": i, "title": t, "text": txt} for i, (t, txt) in enumerate(corpus.items())),
    )
    write_jsonl(out / "questions.jsonl", (to_question_record(r) for r in clean))
    for name, part in (("eval200", ev), ("rl1200", rl), ("sft2000", sft)):
        write_jsonl(out / "splits" / f"{name}.jsonl", (to_question_record(r) for r in part))

    s = stats(rows, corpus, splits)
    s["corpus_written"] = n
    (out / "stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
