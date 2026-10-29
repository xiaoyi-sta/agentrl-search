"""Smoke tests for normalize / corpus / splits. Run: python tests/test_stage1.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentrl.data.build import build_corpus, make_splits
from agentrl.data.normalize import normalize_row, to_paras, to_support


def test_to_paras_handles_dict_and_list():
    as_dict = {"A": ["s1", "s2"], "B": ["s3"]}
    as_list = [{"title": "A", "sentences": ["s1", "s2"]}, {"title": "B", "sentences": ["s3"]}]
    assert to_paras(as_dict) == to_paras(as_list)
    assert len(to_paras(as_list)) == 2


def test_to_support_handles_dict_and_list():
    assert to_support({"title": ["A"], "sent_id": [0]}) == [{"title": "A", "sent_id": 0}]
    assert to_support([{"title": "A", "sent_id": 0}]) == [{"title": "A", "sent_id": 0}]


def _row(i):
    return {
        "id": f"q{i}",
        "question": f"question {i}?",
        "answer": "yes",
        "type": "comparison" if i % 2 else "bridge",
        "level": "hard" if i % 3 else "easy",
        "context": [{"title": f"T{i}", "sentences": ["a b c"]}, {"title": "Common", "sentences": ["d e f"]}],
        "supporting_facts": {"title": [f"T{i}"], "sent_id": [0]},
    }


def test_corpus_dedup_and_splits_disjoint():
    rows = [normalize_row(_row(i), i) for i in range(300)]
    corpus = build_corpus(rows)
    assert "Common" in corpus and len(corpus) == 301
    clean, ev, rl, sft = make_splits(rows, n_eval=40, n_rl=60, n_sft=80)
    ids = [set(r["qid"] for r in part) for part in (ev, rl, sft)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert (len(ev), len(rl), len(sft)) == (40, 60, 80)


if __name__ == "__main__":
    test_to_paras_handles_dict_and_list()
    test_to_support_handles_dict_and_list()
    test_corpus_dedup_and_splits_disjoint()
    print("all stage-1 tests passed")
