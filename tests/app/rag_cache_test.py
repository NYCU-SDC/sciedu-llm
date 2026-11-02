"""Preset isolation and shared, settings-keyed builds without upstream calls."""

import asyncio

import pytest
from pydantic import ValidationError

from app.presets import Preset
from app.rag_builds import BuildStatus
from app.rag_cache import RagBuildCache
from rag.config import RAGConfig


def preset(name="a", corpus=("corpus/bio",), *, enabled=True):
    return Preset.model_validate(
        {
            "name": name,
            "tools": {
                "rag": {
                    "enable_tool": enabled,
                    "corpus_datasets": list(corpus),
                }
            },
        }
    )


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.is_built = False
        self.corpus_dataset_names = []
        self.calls = []
        self.release = asyncio.Event()
        self.release.set()
        self.failure = None

    def config_snapshot(self):
        return self.config.model_dump()

    def apply_overrides(self, overrides):
        self.config = RAGConfig.model_validate(self.config.model_dump() | overrides)

    async def build(self, names):
        self.calls.append(list(names))
        await self.release.wait()
        if self.failure:
            raise self.failure
        self.corpus_dataset_names = list(names)
        self.is_built = True


class Registry:
    def __init__(self, *presets):
        self.presets = {p.name: p for p in presets}

    async def get(self, name):
        from app.presets import PresetNotFoundError

        if name not in self.presets:
            raise PresetNotFoundError(name)
        return self.presets[name]

    def snapshot(self):
        return dict(self.presets)


@pytest.mark.asyncio
async def test_matching_presets_share_completed_and_inflight_builds():
    cache = RagBuildCache(FakePipeline)
    a = preset(corpus=("corpus/b", "corpus/a", "corpus/a"))
    b = preset("b", corpus=("corpus/a", "corpus/b"))
    manager = cache.ensure(a)
    assert manager is cache.ensure(b)
    await manager.wait()
    assert manager is cache.ensure(b)
    assert manager.pipeline.calls == [["corpus/a", "corpus/b"]]
    await cache.aclose()


@pytest.mark.asyncio
async def test_corpus_changes_and_disabled_presets_are_isolated():
    cache = RagBuildCache(FakePipeline)
    old = cache.ensure(preset())
    await old.wait()
    new = cache.ensure(preset(corpus=("corpus/chem",)))
    assert old is not new
    assert old.pipeline.corpus_dataset_names == ["corpus/bio"]
    assert not new.pipeline.is_built
    assert cache.ensure(preset(corpus=())) is None
    assert cache.ensure(preset(enabled=False)) is None
    await new.wait()
    assert new.pipeline.corpus_dataset_names == ["corpus/chem"]
    await cache.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_model", "another-model"),
        ("embedding_batch_size", 16),
        ("chunk_size", 600),
        ("chunk_overlap", 90),
    ],
)
async def test_build_setting_changes_select_a_new_entry_and_can_reuse_old(field, value):
    cache = RagBuildCache(FakePipeline)
    p = preset()
    old = cache.ensure(p)
    await old.wait()
    original = cache.config_snapshot()[field]
    cache.apply_overrides({field: value})
    new = cache.ensure(p)
    assert old is not new
    assert old.pipeline.config_snapshot()[field] == original
    await new.wait()
    cache.apply_overrides({field: original})
    assert cache.ensure(p) is old
    assert len(old.pipeline.calls) == 1
    await cache.aclose()


@pytest.mark.asyncio
async def test_retrieval_settings_apply_globally_without_rebuilding():
    cache = RagBuildCache(FakePipeline)
    a, b = preset(), preset("b", ("corpus/chem",))
    first, second = cache.ensure(a), cache.ensure(b)
    await asyncio.gather(first.wait(), second.wait())
    cache.apply_overrides({"final_k": 9, "rerank_model": "new-reranker"})
    assert cache.ensure(a) is first
    for manager in (first, second):
        assert manager.pipeline.config_snapshot()["final_k"] == 9
        assert manager.pipeline.config_snapshot()["rerank_model"] == "new-reranker"
        assert len(manager.pipeline.calls) == 1
    await cache.aclose()


@pytest.mark.asyncio
async def test_failed_new_settings_do_not_corrupt_cached_indexes_or_retry_on_poll():
    cache = RagBuildCache(FakePipeline)
    p = preset()
    old = cache.ensure(p)
    await old.wait()
    cache.apply_overrides({"chunk_size": 700})
    new = cache.manager(p)
    new.pipeline.failure = RuntimeError("embedding failed")
    cache.ensure(p)
    await new.wait()
    assert new.state.status is BuildStatus.FAILED
    assert cache.config_snapshot()["chunk_size"] == 700
    assert cache.ensure(p) is new
    assert len(new.pipeline.calls) == 1
    assert old.pipeline.is_built
    cache.apply_overrides({"chunk_size": old.pipeline.config.chunk_size})
    assert cache.ensure(p) is old
    await cache.aclose()


@pytest.mark.asyncio
async def test_cancellation_before_task_starts_settles_and_can_be_retried():
    cache = RagBuildCache(FakePipeline)
    manager = cache.ensure(preset())
    manager.cancel()
    await manager.wait()
    assert manager.state.status is BuildStatus.CANCELLED
    assert cache.ensure(preset()) is manager
    assert manager.pipeline.calls == []
    manager.start(["corpus/bio"])
    await manager.wait()
    assert manager.pipeline.is_built
    await cache.aclose()


def test_invalid_global_update_is_atomic():
    cache = RagBuildCache(FakePipeline)
    original = cache.config_snapshot()
    with pytest.raises(ValidationError):
        cache.apply_overrides({"chunk_overlap": 1000, "final_k": 42})
    assert cache.config_snapshot() == original


@pytest.mark.asyncio
async def test_pipeline_returns_only_the_presets_own_completed_index():
    cache = RagBuildCache(FakePipeline)
    a, b = preset(), preset("b", ("corpus/chem",))
    assert cache.pipeline(a) is None
    await cache.manager(a).wait()
    assert cache.pipeline(a).corpus_dataset_names == ["corpus/bio"]
    assert cache.pipeline(b) is None
    await cache.manager(b).wait()
    assert cache.pipeline(b).corpus_dataset_names == ["corpus/chem"]
    await cache.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,selection,expected_preset",
    [
        ("/agents", {"preset": "chem"}, "chem"),
        ("/agents", {}, "a"),
        ("/chat", {"enable_rag": True}, "chem"),
        ("/chat", {"enable_rag": False}, "plain"),
    ],
)
async def test_chat_routes_use_exactly_the_resolved_presets_index(
    monkeypatch, path, selection, expected_preset
):
    from types import SimpleNamespace
    from httpx import ASGITransport, AsyncClient
    from app.dependencies import (
        Settings,
        get_settings,
        get_rag_cache,
        get_preset_registry,
        get_openai_client,
        get_langfuse_client,
    )
    from app.main import app
    from app.agents.run import PresetRunError
    from app.routers import agents, chat

    cache = RagBuildCache(FakePipeline)
    registry = Registry(
        preset(), preset("chem", ("corpus/chem",)), preset("plain", ("corpus/plain",))
    )
    settings = Settings(
        openai_api_key="fake",
        agents_default_preset="a",
        chat_rag_preset_name="chem",
        chat_preset_name="plain",
    )

    def prepare(*, preset, rag_pipeline, **_kwargs):
        # Stop before generation, proving which index the real router supplies.
        assert rag_pipeline is cache.manager(preset).pipeline
        raise PresetRunError(
            418, f"{preset.name}:{rag_pipeline.corpus_dataset_names[0]}"
        )

    monkeypatch.setattr(agents, "prepare_preset_run", prepare)
    monkeypatch.setattr(chat, "prepare_preset_run", prepare)
    saved = dict(app.dependency_overrides)
    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_rag_cache: lambda: cache,
            get_preset_registry: lambda: registry,
            get_openai_client: lambda: SimpleNamespace(),
            get_langfuse_client: lambda: SimpleNamespace(),
        }
    )
    try:
        # Build all three real cache entries first; none can be chosen as a fallback.
        cache.warm(registry.snapshot().values())
        for p in registry.snapshot().values():
            await cache.manager(p).wait()
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post(
                path,
                json={
                    "messages": [{"role": "user", "content": "question"}],
                    "stream": False,
                    **selection,
                },
            )
        assert response.status_code == 418
        expected = (await registry.get(expected_preset)).tools.rag.corpus_datasets[0]
        assert response.json()["detail"] == f"{expected_preset}:{expected}"
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)
        await cache.aclose()


@pytest.mark.asyncio
async def test_cache_with_real_indexes_keeps_chapter_resolution_separate():
    from rag import RAGPipeline
    from tests.rag.test_build_logging import _fake_langfuse, _fake_openai

    config = RAGConfig(chunk_size=20, chunk_overlap=0)
    cache = RagBuildCache(
        lambda cfg: RAGPipeline(_fake_openai(), _fake_langfuse(), config=cfg),
        config=config,
    )
    a, b = preset(corpus=("corpus/a",)), preset("b", ("corpus/b",))
    first, second = cache.ensure(a), cache.ensure(b)
    await asyncio.gather(first.wait(), second.wait())
    assert first.pipeline.is_built and second.pipeline.is_built
    assert first.pipeline.resolve_chunks("ch1", 0, 10000)
    assert second.pipeline.resolve_chunks("ch2", 0, 10000)
    assert first.pipeline.resolve_chunks("ch2", 0, 10000) == []
    assert second.pipeline.resolve_chunks("ch1", 0, 10000) == []
    cache.apply_overrides({"final_k": 12})
    assert config.final_k != 12
    assert first.pipeline.config_snapshot()["final_k"] == 12
    await cache.aclose()


@pytest.mark.parametrize("names", [[""], [" "], ["corpus/bio", "\t"]])
def test_blank_dataset_names_are_rejected(names):
    with pytest.raises(ValidationError, match="non-empty"):
        preset(corpus=names)
