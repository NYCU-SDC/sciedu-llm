import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ["OPENAI_API_KEY"] = "mock_key"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_langfuse_client, get_openai_client
from app.main import app


class _FakeChatPrompt:
    def __init__(self):
        self.last_kwargs: dict | None = None

    def compile(self, **kwargs):
        self.last_kwargs = kwargs
        return [
            {"role": "system", "content": "You are a chat-title generator."},
            {"role": "user", "content": kwargs["conversation"]},
        ]


class _FakeLangfuse:
    def __init__(self):
        self.prompt = _FakeChatPrompt()
        self.requested_types: list[str | None] = []

    def get_prompt(self, name: str, type: str | None = None):
        self.requested_types.append(type)
        return self.prompt

    @contextmanager
    def start_as_current_observation(self, **_kw):
        yield SimpleNamespace(update=lambda **_: None)

    def update_current_generation(self, **_kw):
        pass


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _make_openai(*contents: str):
    """Return a fake openai client; consecutive calls yield consecutive contents.

    The last content is reused if calls exceed the list length.
    """
    completions = [_completion(c) for c in contents]
    counter = {"i": 0}

    async def create(**_kwargs):
        i = min(counter["i"], len(completions) - 1)
        counter["i"] += 1
        return completions[i]

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))),
        _calls=counter,
    )


@pytest.fixture
def fake_langfuse():
    return _FakeLangfuse()


@pytest.fixture
def client(fake_langfuse):
    app.dependency_overrides[get_langfuse_client] = lambda: fake_langfuse
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_title_happy_path_chinese(client, fake_langfuse):
    fake_openai = _make_openai("量子糾纏的基本概念與應用")
    app.dependency_overrides[get_openai_client] = lambda: fake_openai

    response = client.post(
        "/chat/title",
        json={
            "messages": [
                {"role": "user", "content": "什麼是量子糾纏？"},
                {"role": "assistant", "content": "量子糾纏是…"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"title": "量子糾纏的基本概念與應用"}
    # Confirm we asked LangFuse for the chat-type prompt.
    assert fake_langfuse.requested_types == ["chat"]
    # Confirm the conversation variable was passed in.
    assert fake_langfuse.prompt.last_kwargs is not None
    assert "什麼是量子糾纏" in fake_langfuse.prompt.last_kwargs["conversation"]


def test_chat_title_strips_quotes_and_period(client):
    fake_openai = _make_openai('  "Intro to Quantum Entanglement."  ')
    app.dependency_overrides[get_openai_client] = lambda: fake_openai

    response = client.post(
        "/chat/title",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Intro to Quantum Entanglement"


def test_chat_title_retries_empty_then_succeeds(client):
    fake_openai = _make_openai("", "   ", "Recovered Title")
    app.dependency_overrides[get_openai_client] = lambda: fake_openai

    response = client.post(
        "/chat/title",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Recovered Title"
    assert fake_openai._calls["i"] == 3


def test_chat_title_all_empty_attempts_returns_422(client):
    fake_openai = _make_openai("", "", "")
    app.dependency_overrides[get_openai_client] = lambda: fake_openai

    response = client.post(
        "/chat/title",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 422
    assert fake_openai._calls["i"] == 3


def test_chat_title_empty_messages_rejected(client):
    response = client.post("/chat/title", json={"messages": []})
    assert response.status_code == 422


def test_chat_title_only_system_messages_rejected(client):
    app.dependency_overrides[get_openai_client] = lambda: _make_openai("anything")
    response = client.post(
        "/chat/title",
        json={"messages": [{"role": "system", "content": "you are…"}]},
    )
    assert response.status_code == 422
