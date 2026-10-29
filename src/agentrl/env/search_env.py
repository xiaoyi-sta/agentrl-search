"""Per-rollout search environment exposed to the policy as tools.

TRL's ``environment_factory`` instantiates this class once per rollout and
turns every public method (except ``reset`` / ``get_reward``) into a tool.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOP = set(
    "a an the of in on at to for from by with and or is are was were be been "
    "what which who whom when where why how does do did that this these those "
    "as it its their his her".split()
)

DEFAULT_CORPUS = Path("data/processed/corpus.jsonl")


def _tok(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _qtok(text: str) -> list[str]:
    return [t for t in _tok(text) if t not in STOP]


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def safe_eval(expression: str):
    """Evaluate arithmetic without eval(); raises ValueError on anything else."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return ev(ast.parse(expression, mode="eval"))


class SearchEnv:
    """Search + calculator environment for HotpotQA-style questions."""

    _index = None  # (docs, bm25): class-level so 4 rollouts share one index

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        max_calls: int = 5,
        top_k: int = 3,
        max_words: int = 60,
    ):
        self.corpus_path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
        self.max_calls = max_calls
        self.top_k = top_k
        self.max_words = max_words
        self.question = ""
        self.answer = ""
        self.supporting_titles: list[str] = []
        self.calls = 0
        self.queries: list[str] = []

    # ------------------------------------------------------------------ #
    @classmethod
    def _ensure_index(cls, corpus_path: Path):
        if cls._index is None:
            docs = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
            bm25 = BM25Okapi([_tok(d["title"] + " . " + d["text"]) for d in docs])
            cls._index = (docs, bm25)
        return cls._index

    # ---------------------------------------------------------- TRL hooks
    def reset(self, question=None, answer=None, supporting_titles=None, **kwargs) -> None:
        """Start a new episode (called once per rollout).

        Args:
            question: The task question.
            answer: Gold answer; used by reward functions only.
            supporting_titles: Gold passage titles (optional).

        Returns:
            None. Returning a string here would override the dataset prompt.
        """
        self.question = question or ""
        self.answer = answer or ""
        self.supporting_titles = list(supporting_titles or [])
        self.calls = 0
        self.queries = []
        return None

    # --------------------------------------------------------------- tools
    def search(self, query: str) -> str:
        """Search the Wikipedia corpus and return the most relevant passages.

        Args:
            query: Keywords or entity names to look up, e.g. "Motörhead band".

        Returns:
            The top-3 passages formatted as "[i] Title: text", or an error
            message if the search budget is exhausted.
        """
        if self.calls >= self.max_calls:
            return (
                "ERROR: search budget exhausted. "
                "Answer with the evidence you already have."
            )
        self.calls += 1
        self.queries.append(query)

        if self._index is None:
            self._index = self._ensure_index(self.corpus_path)
        docs, bm25 = self._index
        scores = bm25.get_scores(_qtok(query))
        top = np.argsort(scores)[::-1][: self.top_k]

        lines = []
        for rank, i in enumerate(top, 1):
            text = " ".join(docs[i]["text"].split()[: self.max_words])
            lines.append(f"[{rank}] {docs[i]['title']}: {text}")
        return "\n".join(lines)

    def calculator(self, expression: str) -> str:
        """Evaluate a basic arithmetic expression.

        Args:
            expression: An arithmetic expression such as "3 * (12 + 5)".

        Returns:
            The numeric result, or an error message.
        """
        try:
            return str(safe_eval(expression))
        except Exception as exc:  # noqa: BLE001 - report back to the model
            return f"ERROR: cannot evaluate {expression!r}: {exc}"