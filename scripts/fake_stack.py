"""Run the app against a fake upstream and a fake Langfuse, for UI work.

The real deployment needs NCHC credentials, a Langfuse project holding the
preset dataset and the character prompts, and an indexed corpus. None of that is
needed to exercise the protocol, so this wires the actual FastAPI app to
stand-ins and serves it on :8000. The agent loop, the preset registry, the tool
registry, the SSE framing and the event ordering are all the real thing.

    uv run python scripts/fake_stack.py            # serve on :8000
    uv run python scripts/fake_stack.py --demo     # drive both endpoints once

`--demo` is the smoke test: it posts a `teacher-student` run (the dataset preset
below) to /agents and a plain turn to /chat through the app in-process and
prints both streams, so the two wire formats can be eyeballed side by side
without a server or a client.

`build_fake_app()` is the wiring on its own, so the same stack can be booted
from somewhere else — `tests/app/stack_integration_test.py` serves it over a
real socket and asserts on what comes back. Importing this module has no side
effects; everything happens in that call.
"""

import asyncio
import contextlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import uvicorn  # noqa: E402

from app.dependencies import (  # noqa: E402
    Settings,
    get_langfuse_client,
    get_openai_client,
    get_rag_pipeline,
    get_settings,
)
from app.main import app  # noqa: E402
from rag.config import get_rag_config  # noqa: E402


def chunk(*, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content)
    if tool_calls is not None:
        delta.tool_calls = tool_calls
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def call(index, *, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def text_chunks(body, size=6):
    return [chunk(content=body[at : at + size]) for at in range(0, len(body), size)]


def tool_turn(call_id, name, arguments):
    """A turn that asks for one tool, with the arguments split across chunks."""
    half = len(arguments) // 2
    return [
        chunk(
            tool_calls=[call(0, call_id=call_id, name=name, arguments=arguments[:half])]
        ),
        chunk(tool_calls=[call(0, arguments=arguments[half:])]),
        chunk(finish_reason="tool_calls"),
    ]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        await asyncio.sleep(0.02)
        return self._chunks.pop(0)


class FakeCompletions:
    """Answers based on what the request asks for, so the loop drives itself."""

    def __init__(self):
        self.turn = 0

    async def create(self, **kwargs):
        tools = {t["function"]["name"] for t in kwargs.get("tools") or []}
        self.turn += 1
        subagent = any(
            m.get("content", "").startswith("PROMPT")
            for m in kwargs["messages"]
            if isinstance(m.get("content"), str)
        )

        if subagent:
            return FakeStream(
                text_chunks(
                    "光反應發生在類囊體膜上，水被分解後放出氧氣，"
                    "同時產生 ATP 和 NADPH，提供給暗反應使用。"
                )
                + [chunk(finish_reason="stop")]
            )

        called = [
            c["function"]["name"]
            for m in kwargs["messages"]
            if m.get("role") == "assistant"
            for c in (m.get("tool_calls") or [])
        ]

        if "rag_search" in tools and "rag_search" not in called:
            return FakeStream(
                text_chunks("我先去翻課本第三章，再讓學生試著回答看看。")
                + tool_turn(
                    "call_1",
                    "rag_search",
                    json.dumps({"query": "第三章 光反應 類囊體"}, ensure_ascii=False),
                )
            )

        if "summon_subagent" in tools and "summon_subagent" not in called:
            return FakeStream(
                tool_turn(
                    "call_2",
                    "summon_subagent",
                    json.dumps({"prompt": "請解釋光反應"}, ensure_ascii=False),
                )
            )

        return FakeStream(
            text_chunks(
                "講得不錯，方向都對。要修正一個地方："
                "課本第三章把 NADPH 稱為「還原力」，"
                "而且強調光反應本身不固定二氧化碳——"
                "固定二氧化碳是暗反應（卡爾文循環）的工作。"
            )
            + [chunk(finish_reason="stop")]
        )


class FakePrompt:
    """Both prompt shapes a preset can name.

    An orchestrator's `prompt_name` is a Langfuse *text* prompt, compiled with no
    variables into a plain string; a summoned character's is a *chat* prompt
    compiled with `task=` into a message list.
    """

    def compile(self, **variables):
        if not variables:
            return "你是一位老師，會先查課本，再讓學生試著回答，最後補充訂正。"
        return [
            {"role": "system", "content": "PROMPT 你是一位認真的學生。"},
            {"role": "user", "content": variables.get("task", "")},
        ]


# One dataset item: a preset this server does not ship, served alongside the
# code defaults — which is what a real deployment does to add a behaviour
# without a release.
TEACHER_STUDENT_PRESET = {
    "name": "teacher-student",
    "description": "老師先查課本，召喚學生作答，再補充訂正。",
    "max_steps": 8,
    "orchestrator": "teacher",
    "characters": [
        {
            "id": "teacher",
            "display_name": "老師",
            "role": "teacher",
            "prompt_name": "agents/teacher-system",
            "tools": ["rag_search", "summon_subagent"],
        },
        {
            "id": "student",
            "display_name": "學生",
            "role": "student",
            "prompt_name": "agents/student",
            "tools": ["rag_search"],
            "max_steps": 3,
        },
    ],
}


class FakeLangfuse:
    @contextlib.contextmanager
    def start_as_current_observation(self, **_kwargs):
        yield SimpleNamespace(update=lambda **_k: None)

    def update_current_generation(self, **_kwargs):
        pass

    def get_prompt(self, _name, type=None):
        return FakePrompt()

    def get_dataset(self, _name):
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    id="teacher-student",
                    input=TEACHER_STUDENT_PRESET,
                    metadata=None,
                )
            ]
        )


class FakeRAG:
    """Enough of `RAGPipeline` for both /agents and the retrieval admin screen.

    `build` deliberately takes its time, in short awaits: the admin screen's
    "stop the rebuild" affordance is only exercisable against a build that is
    still running and that cancels promptly, and a fake that returned at once
    would leave that path untested by hand.
    """

    #: How long a fake re-index pretends to take, and the step it sleeps in.
    BUILD_SECONDS = 45.0
    BUILD_STEP_SECONDS = 0.25

    def __init__(self):
        self._values = get_rag_config().model_dump()
        self.is_built = True
        self.corpus_dataset_names = ["corpus/ver3/biology"]

    def config_snapshot(self):
        return dict(self._values)

    def apply_overrides(self, overrides):
        self._values.update(overrides)

    async def build(self, corpus_dataset_names, **_kwargs):
        elapsed = 0.0
        while elapsed < self.BUILD_SECONDS:
            await asyncio.sleep(self.BUILD_STEP_SECONDS)
            elapsed += self.BUILD_STEP_SECONDS
        self.corpus_dataset_names = list(corpus_dataset_names)
        self.is_built = True

    async def retrieve(self, *, query, **_kwargs):
        await asyncio.sleep(0.4)
        return {
            "context": (
                "第三章 3-2 光合作用的過程\n"
                "光合作用可分為光反應與暗反應兩階段。光反應在葉綠體的類囊體膜上進行，"
                "葉綠素吸收光能後將水分解，釋出氧氣，"
                "並將能量暫存於 ATP 與還原力 NADPH。"
            ),
            "reference_chunks": [41, 42, 57],
        }


def fake_settings() -> Settings:
    """The model knobs the fakes answer to.

    Passed as constructor arguments rather than exported into the environment:
    a developer's real .env would otherwise decide which model this asks for and
    the allow-list would reject it, and mutating the environment of whoever
    imported us is not ours to do.
    """
    return Settings(
        openai_api_key="fake",
        openai_default_model="gpt-oss-120b",
        allowed_models="gpt-oss-120b",
    )


def build_fake_app():
    """Point the real app at the fakes and hand it back.

    Mutates the process-wide `app.main:app` — there is only one — so a caller
    that has other plans for it (a test suite) is responsible for restoring
    `dependency_overrides` and `router.lifespan_context` afterwards.
    """
    settings = fake_settings()
    app.dependency_overrides[get_settings] = lambda: settings

    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    app.dependency_overrides[get_openai_client] = lambda: fake_openai
    app.dependency_overrides[get_langfuse_client] = FakeLangfuse
    # One pipeline for the whole process, not one per request: the retrieval
    # admin screen tunes it and follows a rebuild across several calls, and a
    # fresh instance each time would forget both.
    fake_rag = FakeRAG()
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_rag

    # The lifespan validates models and builds a corpus; neither applies here.
    # The preset registry is built lazily on the first request instead, off the
    # fake Langfuse above — and explicitly dropped first, so a registry left on
    # `app.state` by whoever ran before us cannot serve the wrong dataset.
    app.state.preset_registry = None
    app.router.lifespan_context = lambda _app: contextlib.nullcontext()
    return app


DEMO_REQUESTS = [
    (
        "/agents",
        {
            "messages": [{"role": "user", "content": "幫我根據課本第三章解釋光反應"}],
            "stream": True,
            "preset": "teacher-student",
        },
    ),
    (
        "/chat",
        {
            "messages": [{"role": "user", "content": "幫我根據課本第三章解釋光反應"}],
            "stream": True,
        },
    ),
]


async def demo():
    """Drive both endpoints in-process and print what they put on the wire."""
    import httpx

    transport = httpx.ASGITransport(app=build_fake_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://demo", timeout=60.0
    ) as client:
        for path, body in DEMO_REQUESTS:
            print(f"\n=== POST {path} {json.dumps(body, ensure_ascii=False)}\n")
            async with client.stream("POST", path, json=body) as response:
                print(f"--- {response.status_code} {response.headers['content-type']}")
                async for line in response.aiter_lines():
                    if line:
                        print(line)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        asyncio.run(demo())
    else:
        uvicorn.run(build_fake_app(), host="127.0.0.1", port=8000, log_level="warning")
