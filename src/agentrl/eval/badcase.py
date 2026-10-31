"""Rule-based bad-case attribution.

One primary label per trace (priority order below), plus every rule that fired.
Rules must stay mutually interpretable: each label answers "what should be
fixed next?".
"""

from __future__ import annotations

from collections import Counter

from .metrics import retrieved_gold

PRIORITY = ["max_turns_hit", "format_error", "premature_answer", "no_answer", "retrieval_miss", "reasoning_error"]


def label_trace(t: dict) -> list[str]:
    labels: list[str] = []
    if t.get("max_turns_hit"):
        labels.append("max_turns_hit")
    if t.get("invalid_calls", 0) > 0:
        labels.append("format_error")
    if t.get("calls", 0) == 0 and t.get("em_lenient", 0.0) < 1.0:
        labels.append("premature_answer")
    if t.get("calls", 0) > 0 and not t.get("tagged", True):
        labels.append("no_answer")
    if t.get("calls", 0) > 0 and t.get("em", 0.0) == 0 and t.get("tagged", True):
        if retrieved_gold(t):
            labels.append("reasoning_error")
        else:
            labels.append("retrieval_miss")
    return labels


def primary(labels: list[str]) -> str:
    for name in PRIORITY:
        if name in labels:
            return name
    return "correct" if not labels else labels[0]


def annotate(traces: list[dict]) -> list[dict]:
    for t in traces:
        labels = label_trace(t)
        t["labels"] = labels
        t["label"] = primary(labels)
    return traces


def distribution(traces: list[dict]) -> dict:
    return dict(Counter(t["label"] for t in traces))