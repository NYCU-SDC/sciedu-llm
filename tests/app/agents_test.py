import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b,custom-model"
# Pinned so a developer's local .env cannot decide which model these tests
# ask for — an OPENAI_DEFAULT_MODEL outside ALLOWED_MODELS reds the whole suite.
os.environ["OPENAI_DEFAULT_MODEL"] = "gpt-oss-120b"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import (
    Settings,
    get_langfuse_client,
    get_openai_client,
    get_rag_pipeline,
    get_settings,
)
from app.main import app

# --- fakes ------------------------------------------------------------------
# Deliberately self-contained rather than shared with chat_test.py via a
# conftest: the repo already duplicates these per test module (chat_test.py vs
# title_test.py), and moving them would touch the chat suite that pins the
# legacy stream format.


class _FakeSpan:
    def __init__(self):
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakePrompt:
    def __init__(self, name: str):
        self.name = name

    def compile(self, **variables):
        return [
            {"role": "system", "content": f"PROMPT<{self.name}>"},
            {"role": "user", "content": variables.get("task", "")},
        ]


class _FakeLangfuse:
    def __init__(self):
        self.spans: list[_FakeSpan] = []
        self.observations: list[dict] = []
        self.prompt_requests: list[str] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        span = _FakeSpan()
        self.spans.append(span)
        yield span

    def update_current_generation(self, **_kw):
        pass

    def get_prompt(self, name, type=None):
        self.prompt_requests.append(name)
        return _FakePrompt(name)

    def observation_names(self) -> list[str]:
        return [observation["name"] for observation in self.observations]


def _text_chunk(content: str | None = None, finish_reason: str | None = None):
    """A plain content delta. No `tool_calls` attribute at all, on purpose —
    the engine has to cope with providers that omit it."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def _reasoning_chunk(reasoning: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning_content=reasoning),
                finish_reason=None,
            )
        ]
    )


def _tool_call_chunk(
    index: int = 0,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
):
    fragment = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[fragment]),
                finish_reason=finish_reason,
            )
        ]
    )


def _usage_chunk(prompt: int = 5, completion: int = 7):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeAsyncStreamThenError:
    def __init__(self, chunks, exc):
        self._chunks = list(chunks)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise self._exc


class _ScriptedCompletions:
    """Returns a different stream per call so a multi-step loop can be driven.

    Each script entry is either a list of chunks or an exception. Once the script
    runs out, the last entry is reused so a runaway loop shows up as an assertion
    failure rather than an IndexError.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        entry = (
            self._script.pop(0)
            if len(self._script) > 1
            else self._script[0]
            if self._script
            else []
        )
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, tuple):
            chunks, exc = entry
            return _FakeAsyncStreamThenError(chunks, exc)
        return _FakeAsyncStream(entry)


class _FakeRAGPipeline:
    def __init__(self):
        self.retrieve_calls: list[str] = []

    async def retrieve(self, *, query: str, **_kwargs):
        self.retrieve_calls.append(query)
        return {"context": f"CTX for {query}", "reference_chunks": [1, 2]}


class _FailingRAGPipeline:
    async def retrieve(self, *, query: str, **_kwargs):
        raise RuntimeError("connection to https://secret.internal:8080 refused")


class _SlowRAGPipeline:
    async def retrieve(self, *, query: str, **_kwargs):
        import asyncio

        await asyncio.sleep(5)
        return {"context": "never", "reference_chunks": []}


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def _of_type(events: list[dict], type_: str) -> list[dict]:
    return [event for event in events if event["type"] == type_]


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_langfuse():
    return _FakeLangfuse()


@pytest.fixture
def client(fake_langfuse):
    app.dependency_overrides[get_langfuse_client] = lambda: fake_langfuse
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def override_openai():
    def _install(script):
        completions = _ScriptedCompletions(script)
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        app.dependency_overrides[get_openai_client] = lambda: fake
        return completions

    yield _install
    app.dependency_overrides.pop(get_openai_client, None)


@pytest.fixture
def override_rag():
    def _install(pipeline):
        app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
        return pipeline

    yield _install
    app.dependency_overrides.pop(get_rag_pipeline, None)


@pytest.fixture
def override_settings():
    def _install(**overrides):
        settings = Settings(**overrides)
        app.dependency_overrides[get_settings] = lambda: settings
        return settings

    yield _install
    app.dependency_overrides.pop(get_settings, None)


def _post(client, **body):
    payload = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    payload.update(body)
    return client.post("/agents", json=payload)


# --- plain text -------------------------------------------------------------


def test_agents_plain_text_run_emits_typed_events(client, override_openai):
    override_openai(
        [[_text_chunk("Hello, "), _text_chunk("world!"), _text_chunk(None, "stop")]]
    )

    response = _post(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse(response.text) == [
        {"type": "agent_start", "agent": "assistant"},
        {
            "type": "part_start",
            "index": 0,
            "part": {"type": "text", "id": "p0", "agent": "assistant"},
        },
        {"type": "delta", "index": 0, "delta": "Hello, "},
        {"type": "delta", "index": 0, "delta": "world!"},
        {
            "type": "part_end",
            "index": 0,
            "part": {
                "type": "text",
                "id": "p0",
                "agent": "assistant",
                "text": "Hello, world!",
            },
        },
        {"type": "agent_end", "agent": "assistant"},
        {"type": "done", "finishReason": "stop", "status": "completed"},
    ]


def test_agents_omits_cast_without_a_second_character(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    response = _post(client, tools=["rag_search"])

    assert "cast" not in _types(_parse_sse(response.text))


def test_agents_emits_cast_when_subagent_tool_is_registered(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, tools=["summon_subagent"])

    cast = _of_type(_parse_sse(response.text), "cast")[0]
    assert [c["id"] for c in cast["characters"]] == ["assistant", "subagent"]
    assert cast["characters"][0]["displayName"] == "助教"
    assert cast["characters"][1]["displayName"] == "學生"


def test_agents_streams_reasoning_as_its_own_part(client, override_openai):
    override_openai(
        [[_reasoning_chunk("先想一下"), _text_chunk("答案"), _text_chunk(None, "stop")]]
    )

    events = _parse_sse(_post(client).text)
    starts = _of_type(events, "part_start")
    assert [s["part"]["type"] for s in starts] == ["reasoning", "text"]


# --- rag_search -------------------------------------------------------------


def _rag_search_script(query: str = "光合作用"):
    """Step 1 asks for rag_search with fragmented arguments; step 2 answers."""
    return [
        [
            _tool_call_chunk(0, call_id="call_1", name="rag_search", arguments='{"que'),
            _tool_call_chunk(0, arguments=f'ry": "{query}"' + "}"),
            _text_chunk(None, "tool_calls"),
        ],
        [_text_chunk("Grounded answer"), _text_chunk(None, "stop")],
    ]


def test_agents_rag_search_streams_arguments_then_result(
    client, override_openai, override_rag
):
    override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"]).text)

    tool_call_start = _of_type(events, "part_start")[0]
    assert tool_call_start["part"] == {
        "type": "tool_call",
        "id": "p0",
        "agent": "assistant",
        "tool_call_id": "call_1",
        "name": "rag_search",
    }
    # The raw JSON fragments stream as deltas, exactly as the model produced them.
    assert [e["delta"] for e in _of_type(events, "delta") if e["index"] == 0] == [
        '{"que',
        'ry": "光合作用"}',
    ]
    tool_call_end = [e for e in _of_type(events, "part_end") if e["index"] == 0][0]
    assert tool_call_end["part"]["arguments"] == {"query": "光合作用"}

    result = [e["part"] for e in _of_type(events, "part_end") if e["index"] == 1][0]
    assert result["type"] == "tool_result"
    assert result["status"] == "ok"
    assert result["tool_call_id"] == "call_1"
    assert "CTX for 光合作用" in result["content"]
    # A textbook search is a step worth showing, so it is not internal.
    assert "internal" not in result


def test_agents_rag_search_forwards_the_models_query_not_the_users(
    client, override_openai, override_rag
):
    override_openai(_rag_search_script())
    pipeline = override_rag(_FakeRAGPipeline())

    _post(client, tools=["rag_search"])

    assert pipeline.retrieve_calls == ["光合作用"]


def test_agents_second_step_replays_the_assistant_and_tool_messages(
    client, override_openai, override_rag
):
    completions = override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    _post(client, tools=["rag_search"])

    assert len(completions.calls) == 2
    messages = completions.calls[1]["messages"]
    assistant, tool = messages[-2], messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "rag_search",
                "arguments": '{"query": "光合作用"}',
            },
        }
    ]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"
    assert "CTX for 光合作用" in tool["content"]


def test_agents_enable_rag_registers_rag_search(client, override_openai, override_rag):
    completions = override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    _post(client, enable_rag=True)

    names = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    assert names == ["rag_search"]


def test_agents_accepts_spec_shaped_tool_objects(client, override_openai, override_rag):
    completions = override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    _post(
        client,
        tools=[{"type": "function", "function": {"name": "rag_search"}}],
    )

    names = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    assert names == ["rag_search"]


# --- summon_subagent --------------------------------------------------------


def _summon_script():
    return [
        [
            _tool_call_chunk(
                0,
                call_id="call_1",
                name="summon_subagent",
                arguments='{"prompt": "解釋光反應"}',
            ),
            _text_chunk(None, "tool_calls"),
        ],
        [_text_chunk("光反應發生在類囊體膜上"), _text_chunk(None, "stop")],
        [_text_chunk("講得不錯，不過…"), _text_chunk(None, "stop")],
    ]


def test_agents_summon_subagent_streams_the_subagent_inline(client, override_openai):
    override_openai(_summon_script())

    events = _parse_sse(_post(client, tools=["summon_subagent"]).text)

    assert _types(events) == [
        "cast",
        "agent_start",  # assistant
        "part_start",  # tool_call summon_subagent (internal)
        "delta",
        "part_end",
        "agent_start",  # subagent
        "part_start",  # the subagent's visible text
        "delta",
        "part_end",
        "agent_end",  # subagent
        "part_start",  # tool_result (internal)
        "part_end",
        "agent_start",  # the floor goes back to the orchestrator
        "part_start",
        "delta",
        "part_end",
        "agent_end",
        "done",
    ]

    summon_start = events[5]
    assert summon_start == {
        "type": "agent_start",
        "agent": "subagent",
        "parent": "assistant",
        "summonedBy": "call_1",
    }
    # The mechanism is hidden; what the subagent says is not.
    assert events[2]["part"]["internal"] is True
    assert events[11]["part"]["internal"] is True
    assert "internal" not in events[8]["part"]
    assert events[8]["part"] == {
        "type": "text",
        "id": "p1",
        "agent": "subagent",
        "text": "光反應發生在類囊體膜上",
    }
    # The subagent's answer is handed back to the orchestrator.
    assert events[11]["part"]["content"] == "光反應發生在類囊體膜上"


def test_agents_subagent_gets_its_langfuse_prompt_and_the_summon_task(
    client, override_openai, fake_langfuse
):
    completions = override_openai(_summon_script())

    _post(client, tools=["summon_subagent"])

    assert fake_langfuse.prompt_requests == ["agents/subagent"]
    assert completions.calls[1]["messages"] == [
        {"role": "system", "content": "PROMPT<agents/subagent>"},
        {"role": "user", "content": "解釋光反應"},
    ]


def test_agents_subagent_cannot_summon_another_subagent(
    client, override_openai, override_rag
):
    completions = override_openai(_summon_script())
    override_rag(_FakeRAGPipeline())

    _post(client, tools=["rag_search", "summon_subagent"])

    orchestrator_tools = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    subagent_tools = [t["function"]["name"] for t in completions.calls[1]["tools"]]
    assert orchestrator_tools == ["rag_search", "summon_subagent"]
    assert subagent_tools == ["rag_search"]


def test_agents_subagent_failure_still_closes_the_speaker(client, override_openai):
    script = _summon_script()
    # The subagent dies after saying something.
    script[1] = ([_text_chunk("光反應…")], RuntimeError("boom"))
    override_openai(script)

    events = _parse_sse(_post(client, tools=["summon_subagent"]).text)

    # The frontend is never left showing the subagent as still speaking.
    assert {"type": "agent_end", "agent": "subagent"} in events
    result = [
        e["part"]
        for e in _of_type(events, "part_end")
        if e["part"]["type"] == "tool_result"
    ][0]
    assert result["status"] == "error"
    assert result["code"] == "subagent_failed"
    assert "光反應…" in result["content"]
    # The orchestrator carries on and the run still completes.
    assert _types(events)[-1] == "done"


# --- tool failures ----------------------------------------------------------


def _bad_tool_script(name: str, arguments: str, *, steps: int = 1):
    script = []
    for step in range(steps):
        script.append(
            [
                _tool_call_chunk(
                    0, call_id=f"call_{step}", name=name, arguments=arguments
                ),
                _text_chunk(None, "tool_calls"),
            ]
        )
    script.append([_text_chunk("Answering anyway"), _text_chunk(None, "stop")])
    return script


def _tool_result(events: list[dict]) -> dict:
    return [
        e["part"]
        for e in _of_type(events, "part_end")
        if e["part"]["type"] == "tool_result"
    ][0]


def test_agents_unknown_tool_name_becomes_a_recoverable_error(
    client, override_openai, override_rag
):
    override_openai(_bad_tool_script("search_textbook", "{}"))
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"]).text)

    result = _tool_result(events)
    assert result["status"] == "error"
    assert result["code"] == "unknown_tool"
    assert "rag_search" in result["content"]
    assert _types(events)[-1] == "done"


def test_agents_malformed_tool_arguments_become_a_recoverable_error(
    client, override_openai, override_rag
):
    override_openai(_bad_tool_script("rag_search", '{"query": '))
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"]).text)

    result = _tool_result(events)
    assert result["code"] == "invalid_arguments"
    # The model is handed the schema it should have matched.
    assert '"query"' in result["content"]
    assert _types(events)[-1] == "done"


def test_agents_tool_argument_schema_violation_becomes_a_recoverable_error(
    client, override_openai, override_rag
):
    override_openai(_bad_tool_script("rag_search", '{"topic": "x"}'))
    override_rag(_FakeRAGPipeline())

    result = _tool_result(_parse_sse(_post(client, tools=["rag_search"]).text))

    assert result["code"] == "invalid_arguments"
    assert "query" in result["content"]


def test_agents_tool_exception_does_not_leak_the_upstream_message(
    client, override_openai, override_rag
):
    override_openai(_rag_search_script())
    override_rag(_FailingRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"]).text)

    result = _tool_result(events)
    assert result["code"] == "tool_failed"
    assert "secret.internal" not in result["content"]
    assert _types(events)[-1] == "done"


def test_agents_tool_timeout_becomes_a_recoverable_error(
    client, override_openai, override_rag, override_settings
):
    override_settings(agents_tool_timeout_seconds=0.05)
    override_openai(_rag_search_script())
    override_rag(_SlowRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"]).text)

    result = _tool_result(events)
    assert result["code"] == "tool_timeout"
    assert _types(events)[-1] == "done"


def test_agents_consecutive_tool_failures_disable_tools(
    client, override_openai, override_rag
):
    completions = override_openai(_bad_tool_script("nope", "{}", steps=3))
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"], max_steps=8).text)

    assert len(completions.calls) == 4
    assert "tools" in completions.calls[2]
    # After three failures in a row the model is made to answer with what it has.
    assert "tools" not in completions.calls[3]
    assert _types(events)[-1] == "done"


def test_agents_truncates_an_oversized_tool_result(
    client, override_openai, override_rag
):
    class _HugeRAGPipeline:
        async def retrieve(self, *, query: str, **_kwargs):
            return {"context": "字" * 20000, "reference_chunks": [1]}

    override_openai(_rag_search_script())
    override_rag(_HugeRAGPipeline())

    result = _tool_result(_parse_sse(_post(client, tools=["rag_search"]).text))

    assert len(result["content"]) < 20000
    assert result["content"].endswith("（內容過長，已截斷）")


# --- step budget ------------------------------------------------------------


def test_agents_max_steps_forces_a_final_tool_free_answer(
    client, override_openai, override_rag
):
    completions = override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, tools=["rag_search"], max_steps=1).text)

    assert len(completions.calls) == 2
    assert "tools" not in completions.calls[1]
    assert completions.calls[1]["messages"][-1]["role"] == "system"
    done = _of_type(events, "done")[0]
    assert done == {"type": "done", "finishReason": "max_steps", "status": "completed"}


def test_agents_forces_tool_choice_only_on_the_first_step(
    client, override_openai, override_rag
):
    completions = override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    _post(client, tools=["rag_search"], tool_choice="required")

    assert completions.calls[0]["tool_choice"] == "required"
    # Re-forcing it would leave the model unable to ever stop and answer.
    assert completions.calls[1]["tool_choice"] == "auto"


# --- terminal failures ------------------------------------------------------


def test_agents_upstream_failure_ends_the_stream_with_an_error_event(
    client, override_openai
):
    override_openai([([_text_chunk("Par"), _text_chunk("tial")], RuntimeError("boom"))])

    events = _parse_sse(_post(client).text)

    assert _types(events) == [
        "agent_start",
        "part_start",
        "delta",
        "delta",
        "agent_end",
        "error",
    ]
    assert events[-1] == {
        "type": "error",
        "error": "Error while communicating with the OpenAI API",
        "code": "upstream_error",
    }


def test_agents_upstream_failure_records_partial_output(
    client, override_openai, fake_langfuse
):
    override_openai([([_text_chunk("Par"), _text_chunk("tial")], RuntimeError("boom"))])

    _post(client)

    names = fake_langfuse.observation_names()
    generation_update = fake_langfuse.spans[names.index("generation")].updates[0]
    assert generation_update["output"] == "Partial"
    assert generation_update["level"] == "ERROR"
    agents_update = fake_langfuse.spans[names.index("agents")].updates[0]
    assert agents_update["output"] == "Partial"
    assert agents_update["level"] == "ERROR"


# --- request validation -----------------------------------------------------


def test_agents_rejects_an_unknown_tool_name(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, tools=["search"])

    assert response.status_code == 400
    assert "Unknown tool 'search'" in response.json()["detail"]


def test_agents_returns_503_when_a_tool_needs_unconfigured_rag(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(None)

    response = _post(client, tools=["rag_search"])

    assert response.status_code == 503
    assert "requires RAG" in response.json()["detail"]


def test_agents_rejects_out_of_range_max_steps(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    assert _post(client, max_steps=99).status_code == 422
    assert _post(client, max_steps=0).status_code == 422


def test_agents_rejects_a_tool_choice_naming_an_unregistered_tool(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    response = _post(
        client,
        tools=["rag_search"],
        tool_choice={"type": "function", "function": {"name": "summon_subagent"}},
    )

    assert response.status_code == 400
    assert "not one of this run's tools" in response.json()["detail"]


def test_agents_rejects_required_tool_choice_without_tools(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, tool_choice="required")

    assert response.status_code == 400
    assert "at least one tool" in response.json()["detail"]


def test_agents_rejects_a_disallowed_model(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, model="gpt-4")

    assert response.status_code == 400
    assert "is not allowed" in response.json()["detail"]


def test_agents_rejects_an_invalid_request_body(client):
    # `stream` is required by the schema
    response = client.post(
        "/agents", json={"messages": [{"role": "user", "content": "Hi"}]}
    )
    assert response.status_code == 422


# --- non-streaming ----------------------------------------------------------


def test_agents_non_streaming_returns_the_same_parts(
    client, override_openai, override_rag
):
    override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())
    streamed = _parse_sse(_post(client, tools=["rag_search"]).text)

    override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())
    response = _post(client, tools=["rag_search"], stream=False)

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert body["status"] == "completed"
    assert body["finishReason"] == "stop"
    assert "cast" not in body
    assert body["parts"] == [event["part"] for event in _of_type(streamed, "part_end")]


def test_agents_non_streaming_includes_the_cast(client, override_openai):
    override_openai(_summon_script())

    body = _post(client, tools=["summon_subagent"], stream=False).json()

    assert [c["id"] for c in body["cast"]] == ["assistant", "subagent"]
    assert [p["agent"] for p in body["parts"]] == [
        "assistant",
        "subagent",
        "assistant",
        "assistant",
    ]


def test_agents_non_streaming_reports_a_failed_status(client, override_openai):
    override_openai([([_text_chunk("Par")], RuntimeError("boom"))])

    body = _post(client, stream=False).json()

    assert body["status"] == "failed"
    assert body["finishReason"] == "upstream_error"
    assert body["parts"][0]["text"] == "Par"


# --- tracing ----------------------------------------------------------------


def test_agents_traces_a_nested_observation_tree(
    client, override_openai, override_rag, fake_langfuse
):
    override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    _post(client, tools=["rag_search"])

    assert fake_langfuse.observation_names() == [
        "agents",
        "generation",
        "tool-rag_search",
        "generation",
    ]
    kinds = [o["as_type"] for o in fake_langfuse.observations]
    assert kinds == ["agent", "generation", "tool", "generation"]
    tool_update = fake_langfuse.spans[2].updates[0]
    assert tool_update["metadata"] == {"status": "ok", "code": None}


def test_agents_traces_the_subagent_as_its_own_agent_span(
    client, override_openai, fake_langfuse
):
    override_openai(_summon_script())

    _post(client, tools=["summon_subagent"])

    assert fake_langfuse.observation_names() == [
        "agents",
        "generation",
        "tool-summon_subagent",
        "subagent",
        "generation",
        "generation",
    ]


def test_agents_records_usage_on_the_generation(client, override_openai, fake_langfuse):
    override_openai([[_text_chunk("ok", "stop"), _usage_chunk()]])

    _post(client)

    generation_update = fake_langfuse.spans[1].updates[0]
    assert generation_update["usage_details"] == {"input": 5, "output": 7}


def test_agents_propagates_session_and_user_attributes(
    client, override_openai, monkeypatch
):
    override_openai([[_text_chunk("ok", "stop")]])
    calls: list[dict] = []

    def _recorder(**kwargs):
        calls.append(kwargs)
        import contextlib

        return contextlib.nullcontext()

    monkeypatch.setattr("app.routers.agents.propagate_attributes", _recorder)

    response = _post(client, session="s-1", user="u-1")
    # The trace context is entered lazily inside the generator, so the body has to
    # be consumed before asserting.
    assert response.status_code == 200
    _parse_sse(response.text)

    assert calls == [{"session_id": "s-1", "user_id": "u-1"}]
