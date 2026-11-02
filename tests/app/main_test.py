from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from app import main
from tests.app.rag_cache_test import FakePipeline, Registry, preset


@pytest.mark.asyncio
async def test_lifespan_warms_preset_corpora_without_waiting_and_drains_on_shutdown(
    monkeypatch,
):
    registry = Registry(preset(), preset("same"), preset("other", ("corpus/chem",)))

    async def refresh():
        return SimpleNamespace(loaded=list(registry.snapshot()), errors=[])

    registry.refresh = refresh

    async def allowed():
        return []

    async def seed(*_args):
        return []

    async def openai():
        return SimpleNamespace()

    pipelines = []

    def pipeline(*_args, config):
        result = FakePipeline(config)
        result.release.clear()
        pipelines.append(result)
        return result

    runner = SimpleNamespace(shutdown=lambda: None)
    monkeypatch.setattr(main, "validate_allowed_models", allowed)
    monkeypatch.setattr(main, "ensure_default_presets", seed)
    monkeypatch.setattr(main, "get_openai_client", openai)
    monkeypatch.setattr(main, "get_langfuse_client", lambda: SimpleNamespace())
    monkeypatch.setattr(main, "PresetRegistry", lambda **_kwargs: registry)
    monkeypatch.setattr(main, "EvalRunner", lambda *_args: runner)
    monkeypatch.setattr(main, "RAGPipeline", pipeline)
    app = FastAPI()
    async with main.lifespan(app):
        assert len(pipelines) == 2
        assert not any(p.is_built for p in pipelines)
        manager = app.state.rag_cache.manager(await registry.get("a"))
        assert manager.state.is_building
    assert manager.state.status == "cancelled"
