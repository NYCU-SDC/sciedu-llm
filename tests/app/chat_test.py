import contextlib
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
    get_langfuse_client,
    get_openai_client,
    get_preset_registry,
    get_rag_pipeline,
)
from app.main import app
from app.presets import DEFAULT_PRESETS, PresetNotFoundError

# /chat is now a preset run (`default-chat-plain`, or `default-chat` when
# `enable_rag` is set) rendered back into the legacy frame format, so these
# tests pin two things at once: the external contract, which must not have
# moved, and the fact that none of the engine's typed protocol leaks through.


class _FakeSpan:
    def __init__(self):
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class _FakeLangfuse:
    def __init__(self):
        self.spans: list[_FakeSpan] = []
        self.observations: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        span = _FakeSpan()
        self.spans.append(span)
        yield span

    def update_current_generation(self, **_kw):
        pass

    def observation_names(self) -> list[str]:
        return [observation["name"] for observation in self.observations]


class _StubRegistry:
    """Serves the code defaults only — which is exactly what /chat runs on."""

    async def get(self, name: str):
        preset = DEFAULT_PRESETS.get(name)
        if preset is None:
            raise PresetNotFoundError(name)
        return preset

    def names(self) -> list[str]:
        return sorted(DEFAULT_PRESETS)

    def snapshot(self):
        return dict(DEFAULT_PRESETS)


def _chunk(content: str | None = None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def _usage_chunk(prompt: int = 5, completion: int = 7):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _answer(content: str | None, finish_reason: str | None = "stop"):
    """The stream a plain single-turn answer arrives as.

    The engine always streams, even for `stream: false` requests — the
    non-streaming response is the same events folded up — so both paths are
    driven by chunk scripts.
    """
    chunks = []
    if content is not None:
        chunks.append(_chunk(content))
    chunks.append(_chunk(None, finish_reason))
    return chunks


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCompletions:
    def __init__(self, *, stream_chunks=None, exc=None):
        self._stream_chunks = stream_chunks
        self._exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _FakeAsyncStream(self._stream_chunks or [])


def _make_fake_openai(*, stream_chunks=None, exc=None):
    completions = _FakeCompletions(stream_chunks=stream_chunks, exc=exc)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


@pytest.fixture
def fake_langfuse():
    return _FakeLangfuse()


@pytest.fixture
def client(fake_langfuse):
    app.dependency_overrides[get_langfuse_client] = lambda: fake_langfuse
    app.dependency_overrides[get_preset_registry] = _StubRegistry
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def override_openai():
    created: list = []

    def _install(**kwargs):
        fake, completions = _make_fake_openai(**kwargs)
        app.dependency_overrides[get_openai_client] = lambda: fake
        created.append(completions)
        return completions

    yield _install
    app.dependency_overrides.pop(get_openai_client, None)


def test_chat_non_streaming_returns_full_message(client, override_openai):
    completions = override_openai(stream_chunks=_answer("Hello, world!"))

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"content": "Hello, world!", "finishReason": "stop"}
    # The engine always streams upstream; `stream: false` only changes how the
    # events are rendered back to the client.
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_chat_non_streaming_uses_default_model_when_not_provided(
    client, override_openai
):
    completions = override_openai(stream_chunks=_answer("ok"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # Matches Settings.openai_default_model default
    assert completions.calls[0]["model"] == "gpt-oss-120b"


def test_chat_non_streaming_uses_request_model_when_provided(client, override_openai):
    completions = override_openai(stream_chunks=_answer("ok"))

    client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "model": "custom-model",
        },
    )

    assert completions.calls[0]["model"] == "custom-model"


def test_chat_rejects_model_not_in_allowed_list(client, override_openai):
    completions = override_openai(stream_chunks=_answer("ok"))

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "model": "gpt-4",
        },
    )

    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]
    # The disallowed request never reaches the OpenAI API.
    assert completions.calls == []


def test_chat_non_streaming_tolerates_a_choiceless_response(client, override_openai):
    override_openai(stream_chunks=[SimpleNamespace(choices=[])])

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # The old implementation 502'd on `choices: []`. The engine skips chunks it
    # cannot read and the run simply produces nothing, which is the same answer
    # the streaming path has always given for a silent upstream.
    assert response.status_code == 200
    assert response.json() == {"content": "", "finishReason": None}


def test_chat_non_streaming_handles_null_content(client, override_openai):
    override_openai(stream_chunks=_answer(None, "stop"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 200
    assert response.json() == {"content": "", "finishReason": "stop"}


def test_chat_non_streaming_returns_502_on_openai_error(client, override_openai):
    override_openai(exc=RuntimeError("Connection timeout"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 502
    # The upstream exception text is no longer echoed: it routinely carries base
    # URLs and occasionally credentials, so it goes to the log and Langfuse only.
    assert response.json()["detail"] == "Error while communicating with the OpenAI API"


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_chat_streaming_yields_deltas_and_final_chunk(client, override_openai):
    completions = override_openai(
        stream_chunks=[
            _chunk("Hello"),
            _chunk(", "),
            _chunk("world!"),
            _chunk(None, "stop"),
        ]
    )

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert events == [
        {"delta": "Hello", "isFinished": False},
        {"delta": ", ", "isFinished": False},
        {"delta": "world!", "isFinished": False},
        {"delta": "", "isFinished": True},
    ]
    assert completions.calls[0]["stream"] is True


def test_chat_streaming_never_emits_typed_protocol_frames(client, override_openai):
    override_openai(
        stream_chunks=[_chunk("Hi"), _chunk(None, "stop")],
    )

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    events = _parse_sse(response.text)
    # `part_start` / `delta` / `agent_start` / `done` belong to /agents. A legacy
    # client only ever sees `delta` + `isFinished`.
    assert all(set(event) <= {"delta", "isFinished", "error"} for event in events)
    assert not any("type" in event for event in events)


def test_chat_streaming_escapes_non_ascii_like_the_legacy_endpoint(
    client, override_openai
):
    override_openai(stream_chunks=[_chunk("光合作用"), _chunk(None, "stop")])

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    # Byte parity with the old endpoint: `json.dumps` defaults, so Chinese is
    # \uXXXX-escaped on the wire. (/agents deliberately does the opposite.)
    assert "\\u5149\\u5408\\u4f5c\\u7528" in response.text
    assert "光合作用" not in response.text
    assert _parse_sse(response.text)[0] == {"delta": "光合作用", "isFinished": False}


def test_chat_streaming_skips_empty_non_final_chunks(client, override_openai):
    override_openai(
        stream_chunks=[
            _chunk(None),  # empty, not final -> skipped
            _chunk("Hi"),
            _chunk(""),  # empty string, not final -> skipped
            _chunk(None, "stop"),
        ]
    )

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    events = _parse_sse(response.text)
    assert events == [
        {"delta": "Hi", "isFinished": False},
        {"delta": "", "isFinished": True},
    ]


def test_chat_streaming_skips_chunks_with_no_choices(client, override_openai):
    override_openai(
        stream_chunks=[
            SimpleNamespace(choices=[]),
            _chunk("Hi"),
            _chunk(None, "stop"),
        ]
    )

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    events = _parse_sse(response.text)
    assert events == [
        {"delta": "Hi", "isFinished": False},
        {"delta": "", "isFinished": True},
    ]


def test_chat_streaming_emits_error_event_on_openai_error(client, override_openai):
    override_openai(exc=RuntimeError("boom"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    # For streaming, the OpenAI call happens inside the generator (so the
    # generation nests under the chat span), which is after the 200/SSE headers
    # are sent. The failure therefore surfaces as a terminal error event, not an
    # HTTP 502.
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events == [
        {
            "delta": "",
            "isFinished": True,
            "error": "Error while communicating with the OpenAI API",
        }
    ]


class _FakeAsyncStreamThenError:
    """Yields the given chunks, then raises on the next iteration."""

    def __init__(self, chunks, exc):
        self._chunks = list(chunks)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise self._exc


def test_chat_streaming_records_partial_output_on_midstream_error(
    client, fake_langfuse
):
    # A stream that produces two tokens and then fails mid-iteration.
    completions = _FakeCompletions()

    async def _create(**kwargs):
        completions.calls.append(kwargs)
        return _FakeAsyncStreamThenError(
            [_chunk("Par"), _chunk("tial")], RuntimeError("mid-stream boom")
        )

    completions.create = _create  # type: ignore[method-assign]
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    app.dependency_overrides[get_openai_client] = lambda: fake
    try:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
    finally:
        app.dependency_overrides.pop(get_openai_client, None)

    assert response.status_code == 200
    events = _parse_sse(response.text)
    # The two produced tokens stream, then a terminal error event.
    assert events == [
        {"delta": "Par", "isFinished": False},
        {"delta": "tial", "isFinished": False},
        {
            "delta": "",
            "isFinished": True,
            "error": "Error while communicating with the OpenAI API",
        },
    ]

    # The partial content is recorded on the generation and the chat span, both
    # marked as errored — not left empty.
    names = fake_langfuse.observation_names()
    generation_update = fake_langfuse.spans[names.index("generation")].updates[0]
    assert generation_update["output"] == "Partial"
    assert generation_update["level"] == "ERROR"
    chat_span_update = fake_langfuse.spans[names.index("chat")].updates[0]
    assert chat_span_update["output"] == "Partial"
    assert chat_span_update["level"] == "ERROR"


def test_chat_rejects_invalid_request_body(client):
    # `stream` is required by the schema
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 422


def test_chat_non_streaming_creates_langfuse_generation(
    client, override_openai, fake_langfuse
):
    override_openai(stream_chunks=[*_answer("Hi back"), _usage_chunk()])

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 200
    # Outer "chat" span groups the (optional) retrieval with the run; the run is
    # an "agents" observation and the LLM call a "generation" nested inside it.
    assert fake_langfuse.observation_names() == ["chat", "agents", "generation"]
    chat_span, agents_span, generation = fake_langfuse.observations
    assert chat_span["name"] == "chat"
    assert chat_span["as_type"] == "span"
    assert chat_span["metadata"]["stream"] is False
    assert chat_span["metadata"]["rag"] is False
    assert agents_span["as_type"] == "agent"
    assert generation["as_type"] == "generation"
    assert generation["model"] == "gpt-oss-120b"
    assert generation["input"] == {"messages": [{"role": "user", "content": "Hi"}]}

    # Usage + output are recorded on the generation; the outer span records the
    # final answer.
    generation_update = fake_langfuse.spans[2].updates[0]
    assert generation_update["output"] == "Hi back"
    assert generation_update["usage_details"] == {"input": 5, "output": 7}
    assert fake_langfuse.spans[0].updates[0]["output"] == "Hi back"


def test_chat_non_streaming_handles_missing_usage(
    client, override_openai, fake_langfuse
):
    override_openai(stream_chunks=_answer("Hi back"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # No usage in the stream means the engine leaves the field off entirely
    # rather than reporting a zeroed one.
    generation_update = fake_langfuse.spans[2].updates[0]
    assert "usage_details" not in generation_update


def test_chat_streaming_records_accumulated_output_in_langfuse(
    client, override_openai, fake_langfuse
):
    override_openai(
        stream_chunks=[
            _chunk("Hello"),
            _chunk(", "),
            _chunk("world!"),
            _chunk(None, "stop"),
        ]
    )

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )
    assert response.status_code == 200

    assert fake_langfuse.observation_names() == ["chat", "agents", "generation"]
    chat_span = fake_langfuse.observations[0]
    assert chat_span["as_type"] == "span"
    assert chat_span["metadata"]["stream"] is True

    generation_update = fake_langfuse.spans[2].updates[0]
    assert generation_update["output"] == "Hello, world!"
    assert generation_update["metadata"]["finish_reason"] == "stop"
    assert "usage_details" not in generation_update
    assert fake_langfuse.spans[0].updates[0]["output"] == "Hello, world!"


class _FakeRAGPipeline:
    """A pipeline that is *available* — which is all /chat now asks of it.

    Retrieval itself is the `rag_search` tool's business (and the engine's, in
    `tests/app/agents_test.py`); here the pipeline only has to exist, or
    `enable_rag` is a 503.
    """

    def __init__(self):
        self.retrieve_calls: list[str] = []

    async def retrieve(self, *, query: str, **_kwargs):
        self.retrieve_calls.append(query)
        return {"context": f"CTX for {query}", "reference_chunks": [1, 2]}


class _FailingRAGPipeline:
    async def retrieve(self, *, query: str, **_kwargs):
        raise RuntimeError("connection to https://secret.internal:8080 refused")


@pytest.fixture
def override_rag():
    def _install(pipeline):
        app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    yield _install
    app.dependency_overrides.pop(get_rag_pipeline, None)


def test_chat_rag_disabled_by_default_leaves_messages_untouched(
    client, override_openai
):
    completions = override_openai(stream_chunks=_answer("ok"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_chat_enable_rag_returns_503_when_pipeline_unavailable(
    client, override_openai, override_rag
):
    override_openai(stream_chunks=_answer("ok"))
    override_rag(None)

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "enable_rag": True,
        },
    )

    assert response.status_code == 503
    assert "RAG is not enabled" in response.json()["detail"]


def test_chat_enable_rag_offers_the_search_tool_and_leaves_the_messages_alone(
    client, override_openai, override_rag
):
    completions = override_openai(stream_chunks=_answer("Grounded answer"))
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "What is photosynthesis?"}],
            "stream": False,
            "enable_rag": True,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"content": "Grounded answer", "finishReason": "stop"}
    # `enable_rag` now means "the model may search", not "the server searches
    # first": the tool definition goes upstream and the conversation is passed
    # through untouched.
    assert [tool["function"]["name"] for tool in completions.calls[0]["tools"]] == [
        "rag_search"
    ]
    assert completions.calls[0]["messages"] == [
        {"role": "user", "content": "What is photosynthesis?"}
    ]
    # Nothing was retrieved, because the model did not ask for anything.
    assert pipeline.retrieve_calls == []


def test_chat_enable_rag_passes_the_whole_history_through(
    client, override_openai, override_rag
):
    completions = override_openai(stream_chunks=_answer("Grounded answer"))
    override_rag(_FakeRAGPipeline())
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]

    response = client.post(
        "/chat",
        json={"messages": messages, "stream": False, "enable_rag": True},
    )

    assert response.status_code == 200
    # No turn is rewritten and no system message is inserted — the model reads
    # the conversation as the client wrote it and decides what to search for.
    assert completions.calls[0]["messages"] == messages


def test_chat_without_enable_rag_offers_no_tools(client, override_openai, override_rag):
    completions = override_openai(stream_chunks=_answer("ok"))
    override_rag(_FakeRAGPipeline())

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # The plain preset grants nothing, so the model is never even offered a
    # search — which is the difference `enable_rag` makes.
    assert "tools" not in completions.calls[0]


def test_chat_enable_rag_streams_only_the_answer_text(
    client, override_openai, override_rag
):
    completions = override_openai(
        stream_chunks=[_chunk("Answer"), _chunk(None, "stop")]
    )
    override_rag(_FakeRAGPipeline())

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Explain gravity"}],
            "stream": True,
            "enable_rag": True,
        },
    )

    assert response.status_code == 200
    # The legacy frames are unchanged by the preset behind them.
    assert _parse_sse(response.text) == [
        {"delta": "Answer", "isFinished": False},
        {"delta": "", "isFinished": True},
    ]
    assert [tool["function"]["name"] for tool in completions.calls[0]["tools"]] == [
        "rag_search"
    ]


def test_chat_enable_rag_returns_422_without_user_text(
    client, override_openai, override_rag
):
    override_openai(stream_chunks=_answer("ok"))
    override_rag(_FakeRAGPipeline())

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "assistant", "content": "prior turn"}],
            "stream": False,
            "enable_rag": True,
        },
    )

    assert response.status_code == 422
    assert "user message" in response.json()["detail"]


def test_chat_enable_rag_survives_a_failed_search(
    client, override_openai, override_rag
):
    completions = override_openai(stream_chunks=_answer("Answering from memory"))
    override_rag(_FailingRAGPipeline())

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Explain gravity"}],
            "stream": False,
            "enable_rag": True,
        },
    )

    # Retrieval is no longer a pre-step that can fail the whole turn: the model
    # simply never called the tool here, so a broken retriever costs nothing.
    # (A failed *call* is handed back to the model as a recoverable tool error —
    # `tests/app/agents_test.py` pins that.)
    assert response.status_code == 200
    assert response.json()["content"] == "Answering from memory"
    assert "secret.internal" not in response.text
    assert [tool["function"]["name"] for tool in completions.calls[0]["tools"]] == [
        "rag_search"
    ]


class _PropagateRecorder:
    """Stand-in for `propagate_attributes` that records how it was invoked.

    The router calls `propagate_attributes(...)` and uses the result as a
    context manager, so each call returns a no-op context.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return contextlib.nullcontext()


@pytest.fixture
def capture_propagate(monkeypatch):
    recorder = _PropagateRecorder()
    monkeypatch.setattr("app.routers.chat.propagate_attributes", recorder)
    return recorder


def test_chat_non_streaming_propagates_session_and_user(
    client, override_openai, capture_propagate
):
    override_openai(stream_chunks=_answer("ok"))

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "session": "sess-1",
            "user": "user-1",
        },
    )

    assert response.status_code == 200
    assert capture_propagate.calls == [{"session_id": "sess-1", "user_id": "user-1"}]


def test_chat_streaming_propagates_session_and_user(
    client, override_openai, capture_propagate
):
    override_openai(stream_chunks=[_chunk("Hi"), _chunk(None, "stop")])

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "session": "sess-2",
            "user": "user-2",
        },
    )

    assert response.status_code == 200
    # The trace context is entered lazily inside the streaming generator, so the
    # body must be consumed before `propagate_attributes` is invoked.
    _parse_sse(response.text)
    assert capture_propagate.calls == [{"session_id": "sess-2", "user_id": "user-2"}]


def test_chat_propagates_when_only_one_attribute_provided(
    client, override_openai, capture_propagate
):
    override_openai(stream_chunks=_answer("ok"))

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "session": "sess-3",
        },
    )

    assert response.status_code == 200
    # `user` defaults to None but the session still drives propagation.
    assert capture_propagate.calls == [{"session_id": "sess-3", "user_id": None}]


def test_chat_does_not_propagate_without_session_or_user(
    client, override_openai, capture_propagate
):
    override_openai(stream_chunks=_answer("ok"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 200
    assert capture_propagate.calls == []
