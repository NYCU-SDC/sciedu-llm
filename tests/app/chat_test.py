import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_langfuse_client, get_openai_client
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


def test_chat_streaming_returns_502_on_openai_error(client, override_openai):
    override_openai(exc=RuntimeError("boom"))

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


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
    assert len(fake_langfuse.observations) == 1
    obs = fake_langfuse.observations[0]
    assert obs["name"] == "chat"
    assert obs["as_type"] == "generation"
    assert obs["model"] == "gpt-oss-120b"
    assert obs["input"] == {"messages": [{"role": "user", "content": "Hi"}]}
    assert obs["metadata"]["stream"] is False

    update = fake_langfuse.spans[0].updates[0]
    assert update["output"] == "Hi back"
    assert update["usage_details"] == {"input": 5, "output": 7}


def test_chat_non_streaming_handles_missing_usage(
    client, override_openai, fake_langfuse
):
    override_openai(completion=_completion("Hi back", "stop"))

    client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )

    update = fake_langfuse.spans[0].updates[0]
    assert update["usage_details"] is None


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

    assert len(fake_langfuse.observations) == 1
    obs = fake_langfuse.observations[0]
    assert obs["name"] == "chat"
    assert obs["as_type"] == "generation"
    assert obs["metadata"]["stream"] is True

    update = fake_langfuse.spans[0].updates[0]
    assert update["output"] == "Hello, world!"
    assert update["metadata"] == {"stream": True, "finish_reason": "stop"}
    assert update["usage_details"] is None
