import contextlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b,custom-model"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_langfuse_client, get_openai_client, get_rag_pipeline
from app.main import app


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


def _chunk(content: str | None = None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def _completion(content: str, finish_reason: str | None = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
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


class _FakeCompletions:
    def __init__(self, *, stream_chunks=None, completion=None, exc=None):
        self._stream_chunks = stream_chunks
        self._completion = completion
        self._exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        if kwargs.get("stream"):
            return _FakeAsyncStream(self._stream_chunks or [])
        return self._completion


def _make_fake_openai(*, stream_chunks=None, completion=None, exc=None):
    completions = _FakeCompletions(
        stream_chunks=stream_chunks, completion=completion, exc=exc
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


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
    created: list = []

    def _install(**kwargs):
        fake, completions = _make_fake_openai(**kwargs)
        app.dependency_overrides[get_openai_client] = lambda: fake
        created.append(completions)
        return completions

    yield _install
    app.dependency_overrides.pop(get_openai_client, None)


def test_chat_non_streaming_returns_full_message(client, override_openai):
    completions = override_openai(completion=_completion("Hello, world!", "stop"))

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"content": "Hello, world!", "finishReason": "stop"}
    assert completions.calls[0]["stream"] is False
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_chat_non_streaming_uses_default_model_when_not_provided(
    client, override_openai
):
    completions = override_openai(completion=_completion("ok"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # Matches Settings.openai_default_model default
    assert completions.calls[0]["model"] == "gpt-oss-120b"


def test_chat_non_streaming_uses_request_model_when_provided(client, override_openai):
    completions = override_openai(completion=_completion("ok"))

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
    completions = override_openai(completion=_completion("ok"))

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


def test_chat_non_streaming_returns_502_when_no_choices(client, override_openai):
    override_openai(completion=SimpleNamespace(choices=[]))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 502
    assert "no choices" in response.json()["detail"]


def test_chat_non_streaming_handles_null_content(client, override_openai):
    override_openai(completion=_completion(None, "stop"))  # type: ignore[arg-type]

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
    assert "Connection timeout" in response.json()["detail"]


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


def test_chat_rejects_invalid_request_body(client):
    # `stream` is required by the schema
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 422


def _completion_with_usage(
    content: str, finish_reason: str | None = "stop", *, prompt=5, completion=7
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def test_chat_non_streaming_creates_langfuse_generation(
    client, override_openai, fake_langfuse
):
    override_openai(completion=_completion_with_usage("Hi back", "stop"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 200
    # Outer "chat" span groups the (optional) retrieval with the generation; the
    # LLM call is a nested "generation" observation.
    assert len(fake_langfuse.observations) == 2
    chat_span, generation = fake_langfuse.observations
    assert chat_span["name"] == "chat"
    assert chat_span["as_type"] == "span"
    assert chat_span["metadata"]["stream"] is False
    assert chat_span["metadata"]["rag"] is False
    assert generation["name"] == "generation"
    assert generation["as_type"] == "generation"
    assert generation["model"] == "gpt-oss-120b"
    assert generation["input"] == {"messages": [{"role": "user", "content": "Hi"}]}

    # Usage + output are recorded on the generation; the outer span records the
    # final answer.
    generation_update = fake_langfuse.spans[1].updates[0]
    assert generation_update["output"] == "Hi back"
    assert generation_update["usage_details"] == {"input": 5, "output": 7}
    assert fake_langfuse.spans[0].updates[0]["output"] == "Hi back"


def test_chat_non_streaming_handles_missing_usage(
    client, override_openai, fake_langfuse
):
    override_openai(completion=_completion("Hi back", "stop"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    # Usage lives on the nested generation (spans[1]); spans[0] is the chat span.
    generation_update = fake_langfuse.spans[1].updates[0]
    assert generation_update["usage_details"] is None


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

    assert len(fake_langfuse.observations) == 2
    chat_span, generation = fake_langfuse.observations
    assert chat_span["name"] == "chat"
    assert chat_span["as_type"] == "span"
    assert chat_span["metadata"]["stream"] is True
    assert generation["name"] == "generation"
    assert generation["as_type"] == "generation"

    generation_update = fake_langfuse.spans[1].updates[0]
    assert generation_update["output"] == "Hello, world!"
    assert generation_update["metadata"] == {"finish_reason": "stop"}
    assert generation_update["usage_details"] is None
    assert fake_langfuse.spans[0].updates[0]["output"] == "Hello, world!"


class _FakeRAGPipeline:
    def __init__(self):
        self.retrieve_calls: list[str] = []

    async def retrieve(self, *, query: str, **_kwargs):
        self.retrieve_calls.append(query)
        return {"context": f"CTX for {query}", "reference_chunks": [1, 2]}

    def compile_generator_prompt(self, *, context: str, query: str):
        system_message = {"role": "system", "content": "SYSTEM INSTRUCTIONS"}
        user_message = {"role": "user", "content": f"CTX<{context}> Q<{query}>"}
        return system_message, user_message, SimpleNamespace(name="rag-generator-user")


@pytest.fixture
def override_rag():
    def _install(pipeline):
        app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    yield _install
    app.dependency_overrides.pop(get_rag_pipeline, None)


def test_chat_rag_disabled_by_default_leaves_messages_untouched(
    client, override_openai
):
    completions = override_openai(completion=_completion("ok"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_chat_enable_rag_returns_503_when_pipeline_unavailable(
    client, override_openai, override_rag
):
    override_openai(completion=_completion("ok"))
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


def test_chat_enable_rag_augments_messages_with_retrieved_context(
    client, override_openai, override_rag
):
    completions = override_openai(completion=_completion("Grounded answer", "stop"))
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
    assert pipeline.retrieve_calls == ["What is photosynthesis?"]
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "SYSTEM INSTRUCTIONS"},
        {
            "role": "user",
            "content": "CTX<CTX for What is photosynthesis?> Q<What is photosynthesis?>",
        },
    ]


def test_chat_enable_rag_retains_conversation_history(
    client, override_openai, override_rag
):
    completions = override_openai(completion=_completion("Grounded answer", "stop"))
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
            "stream": False,
            "enable_rag": True,
        },
    )

    assert response.status_code == 200
    # Retrieval keyed off the latest user turn only.
    assert pipeline.retrieve_calls == ["second question"]
    # History preserved; RAG system prepended; only the latest user turn augmented.
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "SYSTEM INSTRUCTIONS"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {
            "role": "user",
            "content": "CTX<CTX for second question> Q<second question>",
        },
    ]


def test_chat_enable_rag_streaming_uses_retrieved_context(
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
    events = _parse_sse(response.text)
    assert events == [
        {"delta": "Answer", "isFinished": False},
        {"delta": "", "isFinished": True},
    ]
    assert completions.calls[0]["messages"][0] == {
        "role": "system",
        "content": "SYSTEM INSTRUCTIONS",
    }
    assert completions.calls[0]["messages"][1] == {
        "role": "user",
        "content": "CTX<CTX for Explain gravity> Q<Explain gravity>",
    }


def test_chat_enable_rag_returns_422_without_user_text(
    client, override_openai, override_rag
):
    override_openai(completion=_completion("ok"))
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
    override_openai(completion=_completion("ok"))

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
    override_openai(completion=_completion("ok"))

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
    override_openai(completion=_completion("ok"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    assert response.status_code == 200
    assert capture_propagate.calls == []
