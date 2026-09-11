import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from app import main


class _Registry:
    async def refresh(self):
        return SimpleNamespace(loaded=[], errors=[])


class _EvalRunner:
    def __init__(self, *_args):
        self.was_shutdown = False

    def shutdown(self):
        self.was_shutdown = True


class _Pipeline:
    corpus_dataset_names = ["corpus/bio"]

    def config_snapshot(self):
        return {}


@pytest.fixture(autouse=True)
def _startup_stack(monkeypatch):
    async def validate_allowed_models():
        return []

    async def ensure_default_presets(*_args):
        return []

    async def get_openai_client():
        return SimpleNamespace()

    monkeypatch.setattr(main, "validate_allowed_models", validate_allowed_models)
    monkeypatch.setattr(main, "ensure_default_presets", ensure_default_presets)
    monkeypatch.setattr(main, "get_openai_client", get_openai_client)
    monkeypatch.setattr(main, "get_langfuse_client", lambda: SimpleNamespace())
    monkeypatch.setattr(main, "PresetRegistry", lambda **_kwargs: _Registry())
    monkeypatch.setattr(main, "EvalRunner", _EvalRunner)


@pytest.mark.asyncio
async def test_lifespan_serves_with_rag_disabled_until_background_build_lands(
    monkeypatch,
):
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    pipeline = _Pipeline()

    async def build_rag_pipeline():
        build_started.set()
        await release_build.wait()
        return pipeline

    monkeypatch.setattr(main, "build_rag_pipeline", build_rag_pipeline)
    app = FastAPI()

    async with main.lifespan(app):
        # Entering the lifespan means startup has completed and requests may be
        # served, while the scheduled corpus build is still deliberately blocked.
        assert app.state.rag_pipeline is None
        assert app.state.rag_build_manager is None
        await asyncio.wait_for(build_started.wait(), timeout=1)
        assert app.state.rag_pipeline is None

        release_build.set()
        await asyncio.wait_for(app.state.rag_startup_task, timeout=1)

        assert app.state.rag_pipeline is pipeline
        assert app.state.rag_build_manager.pipeline is pipeline


@pytest.mark.asyncio
async def test_failed_background_build_leaves_rag_disabled(monkeypatch, caplog):
    async def build_rag_pipeline():
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(main, "build_rag_pipeline", build_rag_pipeline)
    app = FastAPI()

    with caplog.at_level("ERROR"):
        async with main.lifespan(app):
            await asyncio.wait_for(app.state.rag_startup_task, timeout=1)
            assert app.state.rag_pipeline is None
            assert app.state.rag_build_manager is None

    assert "Initial RAG index build failed; RAG remains disabled" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_cancels_the_initial_background_build(
    monkeypatch,
):
    build_started = asyncio.Event()

    async def build_rag_pipeline():
        build_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(main, "build_rag_pipeline", build_rag_pipeline)
    app = FastAPI()

    async with main.lifespan(app):
        await asyncio.wait_for(build_started.wait(), timeout=1)
        task = app.state.rag_startup_task
        assert not task.done()

    assert task.cancelled()
    assert app.state.rag_pipeline is None
