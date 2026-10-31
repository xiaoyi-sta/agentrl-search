"""Metrics for the retrieval-agent evaluation."""

from __future__ import annotations

import re
import string
from collections import Counter

PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = s.translate(PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str | None, gold: str) -> float:
    """Diagnostic: does the predicted text contain the gold answer as a phrase?

    Phrase-level (word-boundary) match, so "2 acre" does NOT match
    "approximately 2 acres" — this is a deliberate lower bound.
    """
    if not pred or not gold:
        return 0.0
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    if not g:
        return 0.0
    return float(re.search(rf"\b{re.escape(g)}\b", p) is not None)


def retrieved_gold(trace: dict) -> bool:
    gold = {t.lower() for t in trace.get("supporting_titles", [])}
    got = {t.lower() for t in trace.get("retrieved_titles", [])}
    return bool(gold & got)


def summarize(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {}
    calls = [t.get("calls", 0) for t in traces]
    invalid = [t.get("invalid_calls", 0) for t in traces]
    total_calls = sum(calls)
    return {
        "n": n,
        "em": round(sum(t.get("em", 0.0) for t in traces) / n, 4),
        "f1": round(sum(t.get("f1", 0.0) for t in traces) / n, 4),
        "em_lenient": round(sum(t.get("em_lenient", 0.0) for t in traces) / n, 4),
        "f1_lenient": round(sum(t.get("f1_lenient", 0.0) for t in traces) / n, 4),
        "tag_rate": round(sum(1 for t in traces if t.get("tagged")) / n, 4),
        "em_contains": round(sum(t.get("em_contains", 0.0) for t in traces) / n, 4),
        "avg_calls": round(total_calls / n, 2),
        "invalid_call_rate": round(sum(invalid) / max(1, total_calls), 4),
        "strict_call_rate": round(sum(t.get("strict_calls", 0) for t in traces) / max(1, total_calls), 4),
        "no_answer_rate": round(sum(1 for t in traces if t.get("pred") is None) / n, 4),
        "stopped_early_rate": round(sum(1 for t in traces if t.get("stopped_early")) / n, 4),
        "max_turns_rate": round(sum(1 for t in traces if t.get("max_turns_hit")) / n, 4),
        "tool_hit_rate": round(sum(1 for t in traces if retrieved_gold(t)) / n, 4),
        "avg_out_tokens": round(sum(t.get("out_tokens", 0) for t in traces) / n, 1),
    }


def confusion_2x2(traces: list[dict]) -> dict:
    """retrieval x correctness: separates retrieval failures from reasoning ones."""
    cells = {"hit_correct": 0, "hit_wrong": 0, "miss_correct": 0, "miss_wrong": 0}
    for t in traces:
        hit = retrieved_gold(t)
        correct = t.get("em", 0.0) == 1.0
        key = ("hit" if hit else "miss") + ("_correct" if correct else "_wrong")
        cells[key] += 1
    return cells