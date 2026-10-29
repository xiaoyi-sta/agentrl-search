"""Normalize heterogeneous HotpotQA rows into one stable schema.

The parquet on the Hub stores `context` in one of three shapes:

1. columnar struct: {"title": [t1, t2, ...], "sentences": [[s, ...], [...]]}
2. list of structs: [{"title": t1, "sentences": [...]}, ...]
3. legacy JSON:    [[t1, [s, ...]], ...]

Downstream code only ever sees [{"title", "sentences"}] and
[{"title", "sent_id"}], so schema drift is handled in exactly one place.
"""

from __future__ import annotations

from typing import Any


def _py(obj: Any) -> Any:
    """pyarrow / numpy containers -> plain Python lists; scalars untouched."""
    if obj is None:
        return None
    if hasattr(obj, "to_pylist"):
        return obj.to_pylist()
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return obj.tolist()
    return obj


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def to_paras(ctx: Any) -> list[dict]:
    """Return [{"title": str, "sentences": list[str]}, ...]."""
    ctx = _py(ctx)
    if ctx is None:
        return []

    if isinstance(ctx, dict):
        titles, sents = _py(ctx.get("title")), _py(ctx.get("sentences"))
        if _is_seq(titles) and _is_seq(sents):          # shape 1 (columnar)
            pairs = zip(titles, sents)
        else:                                            # mapping {title: sentences}
            pairs = ctx.items()
        return [
            {"title": str(t), "sentences": [str(x) for x in (_py(ss) or [])]}
            for t, ss in pairs
        ]

    out: list[dict] = []
    for item in ctx:                                     # shape 2 / 3
        item = _py(item)
        if isinstance(item, dict):
            out.append(
                {
                    "title": str(item["title"]),
                    "sentences": [str(x) for x in (_py(item.get("sentences")) or [])],
                }
            )
        else:
            out.append({"title": str(item[0]), "sentences": [str(x) for x in (item[1] or [])]})
    return out


def to_support(sf: Any) -> list[dict]:
    """Return [{"title": str, "sent_id": int}, ...]."""
    sf = _py(sf)
    if sf is None:
        return []
    if isinstance(sf, dict):
        return [
            {"title": str(t), "sent_id": int(i)}
            for t, i in zip(sf.get("title", []), sf.get("sent_id", []))
        ]
    return [{"title": str(x["title"]), "sent_id": int(x["sent_id"])} for x in sf]


def normalize_row(row: dict, idx: int) -> dict:
    """One parquet row -> stable record used by every later stage."""
    paras = to_paras(row.get("context"))
    sup = to_support(row.get("supporting_facts"))
    titles = {p["title"] for p in paras}
    return {
        "qid": str(row.get("id") or f"q{idx}"),
        "question": str(row["question"]).strip(),
        "answer": str(row["answer"]).strip(),
        "type": str(row.get("type") or "unknown"),
        "level": str(row.get("level") or "unknown"),
        "supporting": sup,
        "supporting_titles": list(dict.fromkeys(s["title"] for s in sup)),
        "paras": paras,
        "support_covered": all(s["title"] in titles for s in sup),
    }
