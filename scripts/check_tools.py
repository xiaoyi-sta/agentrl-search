import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from agentrl.env.search_env import SearchEnv

tools = []
for name, fn in inspect.getmembers(SearchEnv, inspect.isfunction):
    if name.startswith("_") or name in {"reset", "get_reward"}:
        continue
    tools.append({
        "type": "function",
        "function": {
            "name": name,
            "description": (fn.__doc__ or "").split("Args:")[0].strip(),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    })

print("tools =", [t["function"]["name"] for t in tools])

tok = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-0.6B")
prompt = tok.apply_chat_template(
    [{"role": "system", "content": "You are a research agent."},
     {"role": "user", "content": "Who directed Ed Wood?"}],
    tools=tools,
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)
print("-" * 40)
print(prompt[-700:])