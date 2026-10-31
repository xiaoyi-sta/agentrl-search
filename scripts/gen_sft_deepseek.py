"""Generate SFT trajectories with DeepSeek V4.1 Flash, using the TRAINING env.

Why SearchEnv and not eval Tools: SFT observations must be byte-identical to
the observations the GRPO environment will produce, otherwise the student
learns a distribution it never sees at RL time.

Types
  A  format samples  : comparison questions, 1-2 searches, teach the protocol
  B  two-hop samples : bridge questions, optional oracle first hop, must search again
  C  abstain samples : fake entities, model must answer <answer>unknown</answer>

Examples
  export DEEPSEEK_API_KEY=...
  python scripts/gen_sft_deepseek.py --type B --n 20 --workers 4 --out data/sft/smoke_b.jsonl
  python scripts/gen_sft_deepseek.py --type B --n 2400 --workers 12 --out data/sft/raw_b.jsonl
  python scripts/gen_sft_deepseek.py --type B --n 4 --fake --out /tmp/fake_b.jsonl   # offline test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentrl.env.search_env import SearchEnv  # noqa: E402
from agentrl.eval.runner import parse_answer, parse_tool_call  # noqa: E402
from agentrl.prompts import SYSTEM_PROMPT  # noqa: E402

HINTS = {
    "A": "Use one or two searches, then finish with <answer>...</answer>.",
    "B": "This is a two-hop question. Search AGAIN with different keywords before answering.",
    "C": "The entity may not exist. If the passages do not contain the answer, reply <answer>unknown</answer>.",
}
FAKE_TEMPLATES = [
    "What is the population of {n}?",
    "Who founded the company {n}?",
    "When was {n} released?",
    "What is the capital of {n}?",
    "Which river flows through {n}?",
]
SYL = ["zor", "bax", "quin", "vel", "drun", "mal", "tir", "ven", "kro", "sil"]

# DeepSeek V4.1 Flash, RMB per million tokens (official pricing page).
PRICE = {"hit_off": 0.02, "hit_peak": 0.04, "miss_off": 1.0, "miss_peak": 2.0,
         "out_off": 4.0, "out_peak": 8.0}

LOCK = threading.Lock()
DROPS: dict[str, int] = {}
STATS = {"requests": 0, "in": 0, "out": 0, "hit": 0, "miss": 0, "errors": 0}
STOP = threading.Event()


def cost(peak: bool = False) -> float:
    hit = STATS["hit"]
    miss = STATS["miss"] if STATS["miss"] else max(0, STATS["in"] - hit)
    if peak:
        return (miss * PRICE["miss_peak"] + hit * PRICE["hit_peak"] + STATS["out"] * PRICE["out_peak"]) / 1e6
    return (miss * PRICE["miss_off"] + hit * PRICE["hit_off"] + STATS["out"] * PRICE["out_off"]) / 1e6


def execute(env: SearchEnv, name: str, args: dict) -> str:
    if name == "search":
        return env.search(str(args.get("query", "")))
    if name == "calculator":
        return env.calculator(str(args.get("expression", "")))
    return f"unknown tool: {name}"


def fake_questions(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        name = "".join(rng.choice(SYL) for _ in range(2)).capitalize() + str(rng.randint(1, 99))
        out.append(rng.choice(FAKE_TEMPLATES).format(n=name))
    return out


def load_pool(parquet: str, exclude: set[str], qtype: str | None) -> list[dict]:
    import pandas as pd

    from agentrl.data.normalize import normalize_row

    df = pd.read_parquet(parquet)
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        r = normalize_row(rec, i)
        if r["qid"] in exclude or not r["support_covered"]:
            continue
        if qtype and r["type"] != qtype:
            continue
        rows.append(r)
    return rows


def chat(client, model: str, messages: list[dict], max_tokens: int = 512) -> str:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            code = getattr(exc, "status_code", None)
            if code in (401, 403) or "invalid_api_key" in msg.lower():
                raise RuntimeError(f"auth failed, fix the key first: {msg}") from exc
            if attempt == 3:
                with LOCK:
                    STATS["errors"] += 1
                raise
            time.sleep(2 ** attempt + random.random())
            continue

        u = resp.usage
        with LOCK:
            STATS["requests"] += 1
            STATS["in"] += getattr(u, "prompt_tokens", 0) or 0
            STATS["out"] += getattr(u, "completion_tokens", 0) or 0
            STATS["hit"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
            STATS["miss"] += getattr(u, "prompt_cache_miss_tokens", 0) or 0
        return (resp.choices[0].message.content or "").strip()
    return ""


def _tool_call(query: str) -> str:
    return '<tool_call>{"name": "search", "arguments": {"query": %s}}</tool_call>' % json.dumps(query)


def fake_turn(turn: int, item: dict, has_oracle: bool, last_was_tool: bool) -> str:
    """Scripted stand-in used by --fake to test the pipeline without spending money."""
    if turn == 0:
        return _tool_call(" ".join(item["question"].split()[:5]))
    if has_oracle and turn == 1:
        return _tool_call(" ".join(item["question"].split()[-5:]))
    ans = item.get("answer") or "unknown"
    return f"<answer>{ans}</answer>"


def rollout(chat_fn, item: dict, hint: str, oracle_title: str | None, fake: bool = False,
            max_turns: int = 5):
    env = SearchEnv()
    env.reset(question=item["question"], answer=item.get("answer", ""),
              supporting_titles=item.get("supporting_titles", []))
    gen = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + hint},
           {"role": "user", "content": item["question"]}]
    saved = [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": item["question"]}]

    if oracle_title:
        first = '<tool_call>{"name": "search", "arguments": {"query": %s}}</tool_call>' % json.dumps(oracle_title)
        obs = env.search(oracle_title)
        gen += [{"role": "assistant", "content": first},
                {"role": "user", "content": "[TOOL OUTPUT]\n" + obs}]
        saved += [{"role": "assistant", "content": first}, {"role": "tool", "content": obs}]

    for turn in range(max_turns):
        if fake:
            text = fake_turn(turn, item, bool(oracle_title), False)
        else:
            text = chat_fn(gen)
        if not text:
            return None, env
        gen.append({"role": "assistant", "content": text})
        saved.append({"role": "assistant", "content": text})
        call = parse_tool_call(text)
        if call is None:
            break
        name, args = call
        obs = execute(env, name, args)
        gen.append({"role": "user", "content": "[TOOL OUTPUT]\n" + obs})
        saved.append({"role": "tool", "content": obs})
    return saved, env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["A", "B", "C"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--parquet", default="data/raw/hotpot_dev.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-ratio", type=float, default=0.5,
                    help="fraction of B samples with a teacher-forced first hop")
    ap.add_argument("--budget", type=float, default=20.0, help="stop after this many RMB (off-peak estimate)")
    ap.add_argument("--fake", action="store_true", help="offline smoke test, no API calls")
    args = ap.parse_args()
    os.chdir(ROOT)          # corpus path is relative to the repo root

    client = None
    if not args.fake:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=args.base_url)

    if client is not None:
        try:
            client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                extra_body={"thinking": {"type": "disabled"}},
            )
            print("api key ok")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"API key check failed, nothing was generated: {exc}") from exc

    exclude: set[str] = set()
    for p in Path("data/processed/splits").glob("*.jsonl"):
        exclude |= {json.loads(line)["qid"] for line in p.open()}

    done: set[str] = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["qid"])
            except Exception:  # noqa: BLE001
                pass
        print(f"resume: {len(done)} done")

    if args.type == "C":
        items = [{"qid": f"fake_{i}", "question": q, "answer": ""}
                 for i, q in enumerate(fake_questions(args.n + len(done), args.seed))]
    else:
        qtype = "bridge" if args.type == "B" else "comparison"
        pool = load_pool(args.parquet, exclude, qtype)
        random.Random(args.seed).shuffle(pool)
        items = pool[: args.n + len(done)]
    items = [x for x in items if x["qid"] not in done][: args.n]
    print(f"type={args.type} items={len(items)} fake={args.fake}")

    def work(idx_item):
        idx, item = idx_item
        if STOP.is_set():
            return None, "stopped"
        oracle = None
        if args.type == "B" and item.get("supporting_titles") and (idx % 100) < args.oracle_ratio * 100:
            oracle = item["supporting_titles"][0]
        try:
            msgs, env = rollout(
                (lambda m: chat(client, args.model, m)) if not args.fake else None,
                item, HINTS[args.type], oracle, fake=args.fake)
        except Exception as exc:  # noqa: BLE001
            print("ERR", item["qid"], exc, file=sys.stderr)
            if "auth failed" in str(exc):
                STOP.set()
                return None, "auth_failed"
            return None, "error"
        if not msgs:
            return None, "empty"
        if parse_answer(msgs[-1]["content"]) is None:
            return None, "no_answer_tag"
        return {
            "qid": item["qid"], "question": item["question"],
            "answer": item.get("answer", ""), "type": args.type,
            "messages": msgs, "calls": env.calls, "queries": env.queries,
            "supporting_titles": item.get("supporting_titles", []),
        }, "ok"

    started = time.time()
    ok = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex, out_path.open("a") as f:
        for rec, why in ex.map(work, enumerate(items)):
            DROPS[why] = DROPS.get(why, 0) + 1
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            if ok and ok % 50 == 0:
                print(f"{ok} ok | {(time.time()-started)/60:.1f} min | ¥{cost():.2f} | {STATS}", flush=True)
            if cost() > args.budget:
                STOP.set()

    print(f"\nsaved {ok} | elapsed {(time.time()-started)/60:.1f} min")
    print(f"tokens in={STATS['in']:,} out={STATS['out']:,} hit={STATS['hit']:,} "
          f"miss={STATS['miss']:,} errors={STATS['errors']}")
    print(f"cost: off-peak ¥{cost(False):.2f} | peak ¥{cost(True):.2f}")
    print("drops:", DROPS)
    if "auth_failed" in DROPS:
        raise SystemExit("aborted: API key invalid (rotate it at platform.deepseek.com)")
    if STOP.is_set():
        print("!! stopped by budget guard")


if __name__ == "__main__":
    main()
