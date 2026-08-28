"""Probe whether the configured upstream supports streaming tool calls.

``/agents`` is inert against a provider that rejects ``tools`` or never emits
``delta.tool_calls``, and an OpenAI-compatible proxy is not obliged to do either.
Run this against each model in ``ALLOWED_MODELS`` before relying on the endpoint:

    uv run python scripts/agents_tool_probe.py

Also reports which attribute (if any) the model streams reasoning tokens on, which
is what decides whether ``reasoning`` parts ever show up.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "搜尋課本內容",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


async def main():
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    for model in ("gpt-oss-120b", "gemma-4-31B-it"):
        print(f"=== {model} ===")
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "請幫我查課本裡光合作用的定義"}],
                tools=[TOOL],
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
            )
            saw_tool_calls = False
            fragments = []
            finish = None
            reasoning_attr = set()
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = choice.finish_reason
                delta = choice.delta
                for attribute in ("reasoning_content", "reasoning"):
                    if getattr(delta, attribute, None):
                        reasoning_attr.add(attribute)
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    saw_tool_calls = True
                    for call in tool_calls:
                        fragments.append(
                            (
                                call.index,
                                call.id,
                                call.function.name if call.function else None,
                                call.function.arguments if call.function else None,
                            )
                        )
            print("  streamed delta.tool_calls:", saw_tool_calls)
            print("  finish_reason:", finish)
            print("  reasoning attrs seen:", sorted(reasoning_attr) or "none")
            for fragment in fragments[:8]:
                print("   ", fragment)
            print("  total fragments:", len(fragments))
        except Exception as e:
            print("  FAILED:", type(e).__name__, str(e)[:300])


asyncio.run(main())
