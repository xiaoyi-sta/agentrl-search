"""Build corpus / questions / splits for the HotpotQA retrieval agent."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .normalize import normalize_row


def load_rows(parquet: str | Path) -> list[dict]:
    df = pd.read_parquet(parquet)
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        rows.append(normalize_row(rec, i))
    return rows


def build_corpus(rows: Iterable[dict]) -> dict[str, str]:
    """title -> paragraph text, keeping the longest variant on collisions."""
    corpus: dict[str, str] = {}
    for row in rows:
        for p in row["paras"]:
            text = " ".join(p["sentences"]).strip()
            title = p["title"]
            if not text:
                continue
            if title not in corpus or len(text) > len(corpus[title]):
                corpus[title] = text
    return corpus


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Proportional sample over (type, level) with largest-remainder rounding."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["type"], r["level"])].append(r)

    total = len(rows)
    quotas: dict[tuple, int] = {}
    remainders = []
    for key, items in groups.items():
        exact = n * len(items) / total
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))
    left = n - sum(quotas.values())
    for _, key in sorted(remainders, reverse=True)[:left]:
        quotas[key] += 1

    picked: list[dict] = []
    for key, items in groups.items():
        rng.shuffle(items)
        picked.extend(items[: quotas[key]])
    rng.shuffle(picked)
    return picked[:n]


def make_splits(rows: list[dict], n_eval=200, n_rl=1200, n_sft=2000, seed=42):
    clean = [r for r in rows if r["support_covered"]]
    rng = random.Random(seed)
    shuffled = clean[:]
    rng.shuffle(shuffled)

    eval_rows = stratified_sample(shuffled, n_eval, seed)
    eval_ids = {r["qid"] for r in eval_rows}
    rest = [r for r in shuffled if r["qid"] not in eval_ids]

    rl_rows = rest[:n_rl]
    rl_ids = {r["qid"] for r in rl_rows}
    sft_rows = [r for r in rest if r["qid"] not in rl_ids][:n_sft]

    assert not (eval_ids & rl_ids)
    assert not (eval_ids & {r["qid"] for r in sft_rows})
    assert not (rl_ids & {r["qid"] for r in sft_rows})
    return clean, eval_rows, rl_rows, sft_rows


def to_question_record(row: dict) -> dict:
    return {
        "qid": row["qid"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_titles": row["supporting_titles"],
    }


def stats(rows, corpus, splits) -> dict:
    clean, ev, rl, sft = splits
    n_paras = [len(r["paras"]) for r in rows]
    words = [len(t.split()) for t in corpus.values()]
    sup_counts = [len(r["supporting"]) for r in rows]
    return {
        "questions_total": len(rows),
        "questions_clean": len(clean),
        "support_not_covered": len(rows) - len(clean),
        "corpus_docs": len(corpus),
        "avg_para_words": round(sum(words) / max(1, len(words)), 1),
        "avg_paras_per_question": round(sum(n_paras) / max(1, len(n_paras)), 2),
        "avg_supporting": round(sum(sup_counts) / max(1, len(sup_counts)), 2),
        "type_dist": dict(Counter(r["type"] for r in rows)),
        "level_dist": dict(Counter(r["level"] for r in rows)),
        "eval_type_dist": dict(Counter(r["type"] for r in ev)),
        "eval_level_dist": dict(Counter(r["level"] for r in ev)),
        "splits": {"eval": len(ev), "rl": len(rl), "sft": len(sft)},
    }
