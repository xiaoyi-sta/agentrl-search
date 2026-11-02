"""Multi-turn rollout loop used for evaluation (greedy, reproducible).

Deliberately NOT built on GRPOTrainer: evaluation must be deterministic and
must share the same parser as training, so the loop is explicit here.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time

from agentrl.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
LOOSE_JSON_RE = re.compile(r"\{.*\}", re.S)

# Some small models invent <search ...> instead of the native <tool_call> JSON.
# We still execute them (so the baseline is measurable) but count them apart.
XML_PATTERNS = [
    (re.compile(r'<search\s+query\s*=\s*["\']([^"\']+)["\']\s*/?>', re.I), "search"),
    (re.compile(r"<search>\s*(.*?)\s*</search>", re.S | re.I), "search"),
    (re.compile(r"<search_query>\s*(.*?)\s*</search_query>", re.S | re.I), "search"),
    (re.compile(r"<calculator>\s*(.*?)\s*</calculator>", re.S | re.I), "calculator"),
]

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> str:
    """Arithmetic only; no names, no calls, no attribute access."""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = _eval(ast.parse(expr, mode="eval"))
        return str(round(value, 6))
    except Exception as exc:  # noqa: BLE001
        return f"calculator error: {exc}"


def _parse_xml(text: str) -> tuple[str, dict] | None:
    """Fallback for models that emit <search ...> instead of <tool_call> JSON."""
    for pat, name in XML_PATTERNS:
        m = pat.search(text)
        if m:
            key = "query" if name == "search" else "expression"
            return name, {key: m.group(1).strip()}
    return None


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse `<tool_call>{"name":..,"arguments":..}</tool_call>` defensively."""
    raw = None
    m = TOOL_CALL_RE.search(text)
    if m:
        raw = m.group(1)
    else:
        m = LOOSE_JSON_RE.search(text)
        if m and '"name"' in m.group(0):
            raw = m.group(0)
    if not raw:
        return _parse_xml(text)
    obj = None
    for candidate in (raw, raw.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            break
        except Exception:  # noqa: BLE001
            continue
    if not isinstance(obj, dict):
        return _parse_xml(text)
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:  # noqa: BLE001
            args = {}
    if not name:
        return _parse_xml(text)
    return str(name), args if isinstance(args, dict) else {}


def parse_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    return ans or None


def truncate_words(text: str, n: int = 60) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


class Tools:
    """Executes tool calls and records usage, mirroring SearchEnv at eval time."""

    def __init__(self, index, max_words_per_doc: int = 60, top_k: int = 3):
        self.index = index
        self.max_words_per_doc = max_words_per_doc
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.queries: list[str] = []
        self.retrieved_titles: list[str] = []

    def execute(self, name: str, args: dict) -> str:
        if name == "search":
            query = str(args.get("query", "")).strip()
            self.calls += 1
            self.queries.append(query)
            if not query:
                return "search error: empty query"
            docs = self.index.search(query, k=self.top_k)
            self.retrieved_titles.extend(d["title"] for d in docs)
            lines = []
            for i, d in enumerate(docs, 1):
                body = " ".join(d["text"].split()[: self.max_words_per_doc])
                lines.append(f"[{i}] {d['title']}: {body}")
            return "\n".join(lines) if lines else "no results"
        if name == "calculator":
            return safe_eval(str(args.get("expression", "")))
        return f"unknown tool: {name}"


class Runner:
    """Greedy multi-turn rollout with a HuggingFace chat model."""

    def __init__(
        self,
        model,
        tokenizer,
        index,
        max_turns: int = 5,
        max_new_tokens: int = 256,
        device: str = "cpu",
        tools: Tools | None = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tools = tools or Tools(index)

    def _render(self, messages: list[dict]) -> str:
        kwargs = dict(tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate(self, prompt: str) -> tuple[str, int]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if self.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), int(new_tokens.shape[0])

    def run_one(self, question: str) -> dict:
        self.tools.reset()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        raw_outputs: list[str] = []
        invalid_calls = 0
        strict_calls = 0
        out_tokens = 0
        answer = None
        turns = 0
        started = time.time()

        for turn in range(self.max_turns):
            prompt = self._render(messages)
            text, n_tok = self._generate(prompt)
            out_tokens += n_tok
            turns = turn + 1
            raw_outputs.append(text)

            call = parse_tool_call(text)
            if call is not None and turn < self.max_turns - 1:
                name, args = call
                if TOOL_CALL_RE.search(text):
                    strict_calls += 1
                if name not in ("search", "calculator"):
                    invalid_calls += 1
                obs = self.tools.execute(name, args)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": obs})
                continue

            if call is not None and turn == self.max_turns - 1:
                invalid_calls += 1
            answer = parse_answer(text)
            break

        last = raw_outputs[-1] if raw_outputs else None
        if last and ("<tool_call" in last or "<search" in last):
            last = None
        pred_lenient = answer if answer is not None else last

        return {
            "turns": turns,
            "queries": list(self.tools.queries),
            "retrieved_titles": list(self.tools.retrieved_titles),
            "raw_outputs": raw_outputs,
            "pred": answer,
            "pred_lenient": pred_lenient,
            "tagged": answer is not None,
            "calls": self.tools.calls,
            "strict_calls": strict_calls,
            "invalid_calls": invalid_calls,
            "out_tokens": out_tokens,
            "stopped_early": answer is None and turns < self.max_turns,
            "max_turns_hit": answer is None and turns >= self.max_turns,
            "latency_s": round(time.time() - started, 2),
        }