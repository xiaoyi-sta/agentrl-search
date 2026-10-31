"""Single source of truth for the agent prompt and tool schemas.

Training (SFT / GRPO) and evaluation MUST import from here, otherwise a
score change can be caused by a prompt change instead of the model.
"""

SYSTEM_PROMPT = """You are a research assistant answering multi-hop questions with a Wikipedia search tool.

Rules:
1. Call `search` whenever you need facts. Use SHORT keyword queries (entity names, 2-6 words), never full sentences.
2. You may call `search` several times; each call returns the top-3 passages with titles.
3. Use `calculator` for arithmetic instead of doing it in your head.
4. When you know the answer, stop and reply exactly:
<answer>your short answer</answer>
5. If two searches do not help, answer with your best guess rather than looping.
6. Tool call format is EXACTLY:
<tool_call>{"name": "search", "arguments": {"query": "KEYWORDS_FROM_THE_QUESTION"}}</tool_call>
   Replace KEYWORDS_FROM_THE_QUESTION with names taken from the CURRENT question.
   Never reuse this placeholder or any example query verbatim.
   Do not use <search> tags. Only the JSON form above is executed."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the Wikipedia corpus and return the top-3 passages with their titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short keyword query such as 'Motörhead band' or 'Ed Wood director'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '3 * (12 + 5)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]