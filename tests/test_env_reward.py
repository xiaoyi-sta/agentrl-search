"""Assertions for the env + reward layer. Run: python tests/test_env_reward.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.env.search_env import SearchEnv, safe_eval
from agentrl.reward.rewards import (
    answer_reward,
    budget_reward,
    em,
    extract_answer,
    f1,
    format_reward,
)


def test_em_articles_and_case():
    assert em("The Beatles", "beatles") == 1.0
    assert em("2 acre", "2 acres") == 0.0
    assert em("", "paris") == 0.0


def test_f1_unicode():
    assert f1("Motörhead", "motörhead") == 1.0
    assert 0.0 < f1("Tim Burton director", "Tim Burton") < 1.0


def test_extract_answer_from_chat():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<tool_call>search</tool_call>"},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "So <answer>Paris</answer>"},
    ]
    assert extract_answer(messages) == "Paris"
    assert extract_answer("no answer here") == ""


def test_format_reward():
    assert format_reward(["<answer>Paris</answer>"]) == [0.1]
    assert format_reward(["Paris"]) == [0.0]
    assert format_reward(["<answer></answer>"]) == [0.0]
    assert format_reward(["<answer>" + "x" * 400 + "</answer>"]) == [0.0]


def test_answer_and_budget_rewards():
    comp = ["<answer>The Beatles</answer>"]
    assert answer_reward(comp, answer=["beatles"])[0] == 0.7 + 0.2

    class FakeEnv:
        calls = 5

    assert budget_reward(comp, environments=[FakeEnv()], max_calls=5, lam=0.1) == [-0.1]
    assert budget_reward(comp, environments=None) == [0.0]


def test_calculator_is_safe():
    assert safe_eval("3 * (12 + 5)") == 51
    env = SearchEnv(corpus_path="__missing__.jsonl")
    assert "ERROR" in env.calculator("__import__('os').system('ls')")


class _FakeBM25:
    def get_scores(self, tokens):
        import numpy as np

        return np.array([1.0])


def test_env_state_machine():
    env = SearchEnv(corpus_path="__fake__.jsonl", max_calls=2)
    env._index = (
        [{"title": "Ed Wood (film)", "text": "Ed Wood is a 1994 film directed by Tim Burton."}],
        _FakeBM25(),
    )
    env.reset(question="Who directed Ed Wood?", answer="Tim Burton")
    out = env.search("Ed Wood director")
    assert out.startswith("[1] Ed Wood (film):")
    assert env.calls == 1
    env.search("again")
    assert "budget exhausted" in env.search("third call")
    assert env.calls == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("all env/reward tests passed")