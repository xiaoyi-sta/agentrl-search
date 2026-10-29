"""Minimal BM25 retrieval over the HotpotQA distractor corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP = set(
    "a an the of in on at to for from by with and or is are was were be been "
    "what which who whom when where why how does do did that this these those "
    "as it its their his her".split()
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def tokenize_query(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP]


class BM25Index:
    def __init__(self, docs: list[dict]):
        """docs: [{"title": str, "text": str}, ...]"""
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["title"] + " . " + d["text"]) for d in docs])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        docs = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
        return cls(docs)

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = self.bm25.get_scores(tokenize_query(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            {"title": self.docs[i]["title"], "text": self.docs[i]["text"], "score": float(scores[i])}
            for i in top
        ]


def evaluate(index: BM25Index, questions: list[dict], k: int = 3) -> dict:
    recalls, all_hits, misses = [], [], []
    for q in questions:
        gold = set(q["supporting_titles"])
        if not gold:
            continue
        got = {d["title"] for d in index.search(q["question"], k=k)}
        hit = len(gold & got)
        recalls.append(hit / len(gold))
        all_hits.append(hit == len(gold))
        if hit < len(gold):
            misses.append(
                {
                    "qid": q["qid"],
                    "question": q["question"],
                    "missing": sorted(gold - got),
                    "got": sorted(got),
                }
            )
    return {
        "k": k,
        "n": len(recalls),
        "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "all_hit_at_k": round(float(np.mean(all_hits)), 4) if all_hits else 0.0,
        "misses": misses[:10],
    }
