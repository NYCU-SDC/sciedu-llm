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
    get_preset_registry,
    get_rag_pipeline,
    get_settings,
)
from app.main import app
from app.presets import DEFAULT_PRESETS, Preset, PresetCharacter, PresetNotFoundError

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
    """Stands in for both prompt shapes a preset can name.

    An orchestrator's ``prompt_name`` is a Langfuse *text* prompt compiled with
    no variables; a summoned character's is a *chat* prompt compiled with
    ``task=``. The fake tells them apart the same way the real client does —
    by what it was asked to compile.
    """

    def __init__(self, name: str):
        self.name = name

    def compile(self, **variables):
        if not variables:
            return f"TEXT<{self.name}>"
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


class _FailingPromptLangfuse(_FakeLangfuse):
    def get_prompt(self, name, type=None):
        self.prompt_requests.append(name)
        raise RuntimeError("langfuse is down")


class _StubRegistry:
    """An in-memory stand-in for ``PresetRegistry``.

    The real registry's job is loading and caching a Langfuse dataset, which
    ``presets_test.py`` covers on its own; here the only thing that matters is
    which preset a name resolves to, so the tests hand-build the ones they need.
    """

    def __init__(self, presets: dict[str, Preset]):
        self._presets = dict(presets)
        self.requested: list[str] = []

    async def get(self, name: str) -> Preset:
        self.requested.append(name)
        preset = self._presets.get(name)
        if preset is None:
            raise PresetNotFoundError(name)
        return preset

    def names(self) -> list[str]:
        return sorted(self._presets)

    def snapshot(self) -> dict[str, Preset]:
        return dict(self._presets)


# --- test presets -----------------------------------------------------------
# Named for what each one exercises rather than for a product behaviour: the
# code defaults are covered by `presets_test.py`, these are engine fixtures.

TEST_PRESETS: dict[str, Preset] = {
    # One character, no tools: the plain streaming path.
    "solo": Preset(
        name="solo",
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="助教")],
    ),
    # One character with the textbook tool.
    "solo-rag": Preset(
        name="solo-rag",
        orchestrator="assistant",
        characters=[
            PresetCharacter(id="assistant", display_name="助教", tools=["rag_search"])
        ],
    ),
    # The same, with a one-step budget so the forced final turn fires.
    "solo-rag-1step": Preset(
        name="solo-rag-1step",
        max_steps=1,
        orchestrator="assistant",
        characters=[
            PresetCharacter(id="assistant", display_name="助教", tools=["rag_search"])
        ],
    ),
    # The same, but the model must call a tool on its first step.
    "solo-rag-required": Preset(
        name="solo-rag-required",
        tool_choice="required",
        orchestrator="assistant",
        characters=[
            PresetCharacter(id="assistant", display_name="助教", tools=["rag_search"])
        ],
    ),
    # Two characters: the orchestrator may summon the second one.
    "pair": Preset(
        name="pair",
        orchestrator="assistant",
        characters=[
            PresetCharacter(
                id="assistant", display_name="助教", tools=["summon_subagent"]
            ),
            PresetCharacter(
                id="subagent",
                display_name="學生",
                role="student",
                prompt_name="agents/subagent",
            ),
        ],
    ),
    # Both characters may search; only the orchestrator may summon.
    "pair-rag": Preset(
        name="pair-rag",
        orchestrator="assistant",
        characters=[
            PresetCharacter(
                id="assistant",
                display_name="助教",
                tools=["rag_search", "summon_subagent"],
            ),
            PresetCharacter(
                id="subagent",
                display_name="學生",
                role="student",
                prompt_name="agents/subagent",
                tools=["rag_search"],
            ),
        ],
    ),
    # Retrieval as an unconditional pre-step rather than a tool the model may
    # call. No default ships this way; a dataset preset still may.
    "solo-forced-rag": Preset(
        name="solo-forced-rag",
        rag_mode="forced",
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="助教")],
    ),
    # Pins its own model, inside the allow-list.
    "solo-custom-model": Preset(
        name="solo-custom-model",
        model="custom-model",
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="助教")],
    ),
    # Pins a model this deployment does not allow — a misconfiguration, not a
    # bad request.
    "solo-bad-model": Preset(
        name="solo-bad-model",
        model="gpt-4",
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="助教")],
    ),
}


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
def registry():
    """The default serving map: every code default plus the engine test presets."""
    return _StubRegistry({**DEFAULT_PRESETS, **TEST_PRESETS})


@pytest.fixture
def client(fake_langfuse, registry):
    app.dependency_overrides[get_langfuse_client] = lambda: fake_langfuse
    app.dependency_overrides[get_preset_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def override_langfuse():
    def _install(langfuse):
        app.dependency_overrides[get_langfuse_client] = lambda: langfuse
        return langfuse

    yield _install


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
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "preset": "solo",
    }
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


def test_agents_omits_cast_for_a_single_character_preset(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    response = _post(client, preset="solo-rag")

    assert "cast" not in _types(_parse_sse(response.text))


def test_agents_emits_cast_for_a_two_character_preset(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, preset="pair")

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


def test_agents_sse_does_not_escape_non_ascii(client, override_openai):
    override_openai([[_text_chunk("光合作用"), _text_chunk(None, "stop")]])

    body = _post(client).text

    # Unlike /chat, the typed protocol ships UTF-8 rather than \uXXXX escapes.
    assert "光合作用" in body
    assert "\\u5149" not in body


# --- presets ----------------------------------------------------------------


def test_agents_runs_the_default_preset_when_none_is_named(
    client, override_openai, override_rag, registry
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    response = client.post(
        "/agents",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    assert response.status_code == 200
    # Settings.agents_default_preset
    assert registry.requested == ["default-agents"]


def test_agents_rejects_an_unknown_preset(client, override_openai):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, preset="nope")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unknown preset 'nope'" in detail
    # The available names are listed so a client can recover without docs.
    assert "default-agents" in detail
    assert "solo" in detail


def test_agents_prepends_the_orchestrator_text_prompt(
    client, override_openai, override_rag, fake_langfuse
):
    completions = override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    _post(client, preset="default-agents")

    # An orchestrator prompt is a *text* prompt, compiled with no variables and
    # prepended as a system message.
    assert fake_langfuse.prompt_requests == ["agents/teacher-system"]
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "TEXT<agents/teacher-system>"},
        {"role": "user", "content": "Hi"},
    ]


def test_agents_scopes_tools_per_character(client, override_openai, override_rag):
    completions = override_openai(_summon_script())
    override_rag(_FakeRAGPipeline())

    _post(client, preset="default-agents")

    teacher_tools = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    student_tools = [t["function"]["name"] for t in completions.calls[1]["tools"]]
    assert teacher_tools == ["rag_search", "summon_subagent"]
    # Only the orchestrator may summon; the student gets its own narrower list.
    assert student_tools == ["rag_search"]


def test_agents_uses_the_model_named_by_the_preset(client, override_openai):
    completions = override_openai([[_text_chunk("ok", "stop")]])

    _post(client, preset="solo-custom-model")

    assert completions.calls[0]["model"] == "custom-model"


def test_agents_returns_503_for_a_preset_model_outside_the_allow_list(
    client, override_openai
):
    override_openai([[_text_chunk("ok", "stop")]])

    response = _post(client, preset="solo-bad-model")

    # A preset is server configuration, so a bad model in one is a 503 rather
    # than a 400 blaming the client.
    assert response.status_code == 503
    assert "not in the allowed models list" in response.json()["detail"]


def test_agents_returns_502_when_the_orchestrator_prompt_cannot_be_loaded(
    client, override_openai, override_rag, override_langfuse
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())
    override_langfuse(_FailingPromptLangfuse())

    response = _post(client, preset="default-agents")

    assert response.status_code == 502
    assert "Failed to load prompt 'agents/teacher-system'" in response.json()["detail"]


def test_agents_ignores_legacy_request_fields(client, override_openai):
    completions = override_openai([[_text_chunk("ok", "stop")]])

    response = _post(
        client,
        preset="solo",
        tools=["rag_search", "summon_subagent"],
        tool_choice="required",
        max_steps=99,
        enable_rag=True,
        model="gpt-4",
    )

    # An older client's body is accepted and every removed field is dropped:
    # the preset, not the request, decides all of this.
    assert response.status_code == 200
    assert "tools" not in completions.calls[0]
    assert completions.calls[0]["model"] == "gpt-oss-120b"


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

    events = _parse_sse(_post(client, preset="solo-rag").text)

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

    _post(client, preset="solo-rag")

    assert pipeline.retrieve_calls == ["光合作用"]


def test_agents_second_step_replays_the_assistant_and_tool_messages(
    client, override_openai, override_rag
):
    completions = override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    _post(client, preset="solo-rag")

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


def test_agents_registers_only_the_presets_tools(client, override_openai, override_rag):
    completions = override_openai([[_text_chunk("ok", "stop")]])
    override_rag(_FakeRAGPipeline())

    _post(client, preset="solo-rag")

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

    events = _parse_sse(_post(client, preset="pair").text)

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

    _post(client, preset="pair")

    # The summoned character's own `prompt_name`, compiled as a chat prompt with
    # the summoner's brief as `task`.
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

    _post(client, preset="pair-rag")

    orchestrator_tools = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    subagent_tools = [t["function"]["name"] for t in completions.calls[1]["tools"]]
    assert orchestrator_tools == ["rag_search", "summon_subagent"]
    assert subagent_tools == ["rag_search"]


def test_agents_subagent_failure_still_closes_the_speaker(client, override_openai):
    script = _summon_script()
    # The subagent dies after saying something.
    script[1] = ([_text_chunk("光反應…")], RuntimeError("boom"))
    override_openai(script)

    events = _parse_sse(_post(client, preset="pair").text)

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

    events = _parse_sse(_post(client, preset="solo-rag").text)

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

    events = _parse_sse(_post(client, preset="solo-rag").text)

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

    result = _tool_result(_parse_sse(_post(client, preset="solo-rag").text))

    assert result["code"] == "invalid_arguments"
    assert "query" in result["content"]


def test_agents_tool_exception_does_not_leak_the_upstream_message(
    client, override_openai, override_rag
):
    override_openai(_rag_search_script())
    override_rag(_FailingRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

    result = _tool_result(events)
    assert result["code"] == "tool_failed"
    assert "secret.internal" not in result["content"]
    assert _types(events)[-1] == "done"


# --- tool calls that never became a call ------------------------------------
# A model can ask for a tool in a way the engine cannot execute: no tool name in
# the stream, an id that duplicates another call in the same turn, or a
# `finish_reason: "tool_calls"` with no call attached (a provider-side tool-call
# parser dropping the payload). These used to be dropped with nothing but a
# server-side warning: no `tool_result`, no message back to the model, and — when
# it was the turn's only call — the loop read "no tool calls" as "the answer is
# done" and ended the run with no text at all. The model could not see that its
# call went nowhere, so it could neither retry nor tell the user.


def _lost_note(completions) -> str | None:
    """The system note about lost calls, from the last request made."""
    for message in reversed(completions.calls[-1]["messages"]):
        if message.get("role") == "system" and "沒有被執行" in message.get(
            "content", ""
        ):
            return message["content"]
    return None


def test_agents_a_tool_call_with_no_name_is_reported_to_the_model(
    client, override_openai, override_rag
):
    completions = override_openai(
        [
            [
                # Arguments arrive, the tool name never does.
                _tool_call_chunk(0, call_id="call_1", arguments='{"query": "x"}'),
                _text_chunk(None, "tool_calls"),
            ],
            [_text_chunk("我先直接回答。"), _text_chunk(None, "stop")],
        ]
    )
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

    # The run does not end on the lost call: the model gets another turn...
    assert len(completions.calls) == 2
    # ...and is told why it has no result to work with.
    note = _lost_note(completions)
    assert note is not None and "沒有工具名稱" in note
    # ...and the user gets a real answer instead of an empty turn.
    assert "我先直接回答。" in "".join(e["delta"] for e in _of_type(events, "delta"))
    assert _types(events)[-1] == "done"


def test_agents_a_duplicate_tool_call_id_is_closed_out_and_reported(
    client, override_openai, override_rag
):
    completions = override_openai(
        [
            [
                _tool_call_chunk(
                    0, call_id="dup", name="rag_search", arguments='{"query": "a"}'
                ),
                _tool_call_chunk(
                    1, call_id="dup", name="rag_search", arguments='{"query": "b"}'
                ),
                _text_chunk(None, "tool_calls"),
            ],
            [_text_chunk("答案"), _text_chunk(None, "stop")],
        ]
    )
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

    results = [
        e["part"]
        for e in _of_type(events, "part_end")
        if e["part"]["type"] == "tool_result"
    ]
    # One real result, and one marking the duplicate as never executed — the
    # frontend would otherwise be left with a tool call that never resolves.
    assert [r["status"] for r in results] == ["error", "ok"]
    lost = next(r for r in results if r["status"] == "error")
    assert lost["code"] == "lost_tool_call"
    # Every tool_call part is closed, so nothing is left mid-stream.
    calls_started = [
        e for e in _of_type(events, "part_start") if e["part"]["type"] == "tool_call"
    ]
    calls_ended = [
        e for e in _of_type(events, "part_end") if e["part"]["type"] == "tool_call"
    ]
    assert len(calls_started) == len(calls_ended) == 2

    note = _lost_note(completions)
    assert note is not None and "重複" in note


def test_agents_a_turn_claiming_tool_calls_but_sending_none_still_answers(
    client, override_openai, override_rag
):
    completions = override_openai(
        [
            # The upstream says it is calling a tool and sends no call at all.
            [_text_chunk(None, "tool_calls")],
            [_text_chunk("改用已知內容回答"), _text_chunk(None, "stop")],
        ]
    )
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

    assert len(completions.calls) == 2
    note = _lost_note(completions)
    assert note is not None and "finish_reason=tool_calls" in note
    assert "改用已知內容回答" in "".join(e["delta"] for e in _of_type(events, "delta"))
    assert _types(events)[-1] == "done"


def test_agents_a_lost_call_note_lands_after_this_turns_tool_results(
    client, override_openai, override_rag
):
    """Ordering matters: a `role: "tool"` message has to follow its assistant
    message with no other role in between, or a strict upstream rejects it."""
    completions = override_openai(
        [
            [
                _tool_call_chunk(
                    0, call_id="dup", name="rag_search", arguments='{"query": "a"}'
                ),
                _tool_call_chunk(
                    1, call_id="dup", name="rag_search", arguments='{"query": "b"}'
                ),
                _text_chunk(None, "tool_calls"),
            ],
            [_text_chunk("答案"), _text_chunk(None, "stop")],
        ]
    )
    override_rag(_FakeRAGPipeline())

    _post(client, preset="solo-rag")

    roles = [m["role"] for m in completions.calls[1]["messages"]]
    assert roles == ["user", "assistant", "tool", "system"]


def test_agents_tool_timeout_becomes_a_recoverable_error(
    client, override_openai, override_rag, override_settings
):
    override_settings(agents_tool_timeout_seconds=0.05)
    override_openai(_rag_search_script())
    override_rag(_SlowRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

    result = _tool_result(events)
    assert result["code"] == "tool_timeout"
    assert _types(events)[-1] == "done"


def test_agents_consecutive_tool_failures_disable_tools(
    client, override_openai, override_rag
):
    completions = override_openai(_bad_tool_script("nope", "{}", steps=3))
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag").text)

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

    result = _tool_result(_parse_sse(_post(client, preset="solo-rag").text))

    assert len(result["content"]) < 20000
    assert result["content"].endswith("（內容過長，已截斷）")


# --- step budget ------------------------------------------------------------


def test_agents_max_steps_forces_a_final_tool_free_answer(
    client, override_openai, override_rag
):
    completions = override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())

    events = _parse_sse(_post(client, preset="solo-rag-1step").text)

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

    _post(client, preset="solo-rag-required")

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


def test_agents_returns_503_when_a_preset_tool_needs_unconfigured_rag(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(None)

    response = _post(client, preset="solo-rag")

    assert response.status_code == 503
    assert "requires RAG" in response.json()["detail"]


def test_agents_returns_503_when_a_forced_rag_preset_has_no_pipeline(
    client, override_openai, override_rag
):
    override_openai([[_text_chunk("ok", "stop")]])
    override_rag(None)

    # No default forces retrieval any more, but a dataset preset still may.
    response = _post(client, preset="solo-forced-rag")

    assert response.status_code == 503
    assert "requires RAG" in response.json()["detail"]


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
    streamed = _parse_sse(_post(client, preset="solo-rag").text)

    override_openai(_rag_search_script())
    override_rag(_FakeRAGPipeline())
    response = _post(client, preset="solo-rag", stream=False)

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert body["status"] == "completed"
    assert body["finishReason"] == "stop"
    assert "cast" not in body
    assert body["parts"] == [event["part"] for event in _of_type(streamed, "part_end")]


def test_agents_non_streaming_includes_the_cast(client, override_openai):
    override_openai(_summon_script())

    body = _post(client, preset="pair", stream=False).json()

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

    _post(client, preset="solo-rag")

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

    _post(client, preset="pair")

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
