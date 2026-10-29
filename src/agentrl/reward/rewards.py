"""Rule-based rewards for the HotpotQA retrieval agent.

Each function has the signature TRL expects::

    f(completions, **kwargs) -> list[float]

``completions`` is either a list of strings or a list of message lists
(conversational). Dataset columns arrive as kwargs (lists), and when the
trainer uses ``environment_factory`` an ``environments`` kwarg is added.
"""

from __future__ import annotations

import re
import string

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
TOOL_RE = re.compile(r"<tool_call>", re.I)

ARTICLES = {"a", "an", "the"}
PUNCT = set(string.punctuation)


def completion_text(completion) -> str:
    """Flatten a completion (str or chat messages) into the assistant text."""
    if isinstance(completion, str):
        return completion
    parts = []
    for msg in completion or []:
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                parts.append(str(msg.get("content") or ""))
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def extract_answer(completion) -> str:
    """Last <answer>...</answer> content, or "" if missing."""
    text = completion_text(completion)
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip().strip("\"'") if matches else ""


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in PUNCT)
    words = [w for w in text.split() if w not in ARTICLES]
    return " ".join(words).strip()


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold) and bool(normalize(gold)))


def f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    counts = {}
    for w in p:
        counts[w] = counts.get(w, 0) + 1
    overlap = 0
    for w in g:
        if counts.get(w, 0) > 0:
            overlap += 1
            counts[w] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _as_list(value, n):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def answer_reward(completions, answer=None, **kwargs) -> list[float]:
    """0.7 * EM + 0.2 * F1 against the gold answer."""
    n = len(completions)
    golds = _as_list(answer, n)
    out = []
    for comp, gold in zip(completions, golds):
        pred = extract_answer(comp)
        if not pred or not gold:
            out.append(0.0)
            continue
        out.append(0.7 * em(pred, gold) + 0.2 * f1(pred, gold))
    return out


def format_reward(completions, **kwargs) -> list[float]:
    """0.1 when the answer is present, short, and no tool call leaks into it."""
    out = []
    for comp in completions:
        pred = extract_answer(comp)
        ok = bool(pred) and len(pred) <= 300 and not TOOL_RE.search(pred)
        out.append(0.1 if ok else 0.0)
    return out


def budget_reward(
    completions, environments=None, max_calls: int = 5, lam: float = 0.1, **kwargs
) -> list[float]:
    """-lam * (tool calls / max_calls); 0 when no environment is available."""
    if not environments:
        return [0.0] * len(completions)
    out = []
    for env in environments:
        calls = float(getattr(env, "calls", 0))
        out.append(-lam * min(1.0, calls / max(1, max_calls)))
    return out