import os
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"

import pytest

from app import dependencies
from app.dependencies import Settings, validate_allowed_models


class _FakeModelsPaginator:
    def __init__(self, ids):
        self._ids = list(ids)

    def __aiter__(self):
        async def _gen():
            for model_id in self._ids:
                yield SimpleNamespace(id=model_id)

        return _gen()


def _fake_client(ids=None, exc=None):
    # `models.list()` returns an async paginator, not a coroutine.
    def list_():
        if exc is not None:
            raise exc
        return _FakeModelsPaginator(ids or [])

    return SimpleNamespace(models=SimpleNamespace(list=list_))


def _install(monkeypatch, *, allowed, served=None, exc=None):
    settings = Settings(openai_api_key="mock_key", allowed_models=allowed)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(
        dependencies, "get_openai_client", lambda: _fake_client(served, exc)
    )
    return settings


@pytest.mark.asyncio
async def test_validate_allowed_models_returns_configured_list(monkeypatch):
    _install(monkeypatch, allowed="a,b", served=["a", "b", "c"])

    assert await validate_allowed_models() == ["a", "b"]


@pytest.mark.asyncio
async def test_validate_allowed_models_raises_when_empty(monkeypatch):
    _install(monkeypatch, allowed="", served=["a"])

    with pytest.raises(ValueError, match="No allowed models configured"):
        await validate_allowed_models()


@pytest.mark.asyncio
async def test_validate_allowed_models_warns_for_unknown(monkeypatch, caplog):
    _install(monkeypatch, allowed="a,ghost", served=["a"])

    with caplog.at_level("WARNING"):
        assert await validate_allowed_models() == ["a", "ghost"]

    assert "ghost" in caplog.text


@pytest.mark.asyncio
async def test_validate_allowed_models_tolerates_listing_failure(monkeypatch, caplog):
    _install(monkeypatch, allowed="a", exc=RuntimeError("boom"))

    with caplog.at_level("ERROR"):
        # Does not raise — the models endpoint check is best-effort.
        assert await validate_allowed_models() == ["a"]

    assert "Could not fetch the model list" in caplog.text
