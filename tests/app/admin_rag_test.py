"""Global RAG configuration and preset-scoped shared build controls."""

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.dependencies import get_preset_registry, get_rag_cache, get_settings
from app.main import app
from app.rag_cache import RagBuildCache
from rag.config import RAGConfig
from tests.app.rag_cache_test import FakePipeline, Registry, preset

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def stack():
    from types import SimpleNamespace

    cache = RagBuildCache(FakePipeline)
    registry = Registry(
        preset(),
        preset("b"),
        preset("chem", ("corpus/chem",)),
        preset("empty", ()),
        preset("off", enabled=False),
    )
    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_rag_cache] = lambda: cache
    app.dependency_overrides[get_preset_registry] = lambda: registry
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        agents_default_preset="a"
    )
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        yield client, cache, registry
    await cache.aclose()
    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved)


async def test_get_reports_selected_preset_and_shares_its_build(stack):
    client, cache, registry = stack
    first = (await client.get("/admin/rag/config?preset=a")).json()
    assert first["preset"] == "a"
    assert first["corpus_datasets"] == ["corpus/bio"]
    assert first["build"]["status"] == "building"
    a = cache.manager(await registry.get("a"))
    await a.wait()
    second = (await client.get("/admin/rag/config?preset=b")).json()
    assert second["is_built"]
    assert second["build"]["status"] == "completed"
    assert cache.manager(await registry.get("b")) is a
    assert len(a.pipeline.calls) == 1


async def test_patch_build_settings_warms_distinct_presets_and_retrieval_is_immediate(
    stack,
):
    client, cache, registry = stack
    await client.get("/admin/rag/config")
    old = cache.manager(await registry.get("a"))
    await old.wait()
    response = await client.patch(
        "/admin/rag/config", json={"chunk_size": 800, "final_k": 9}
    )
    assert response.status_code == 200
    assert response.json()["build_started"]
    assert old.pipeline.config.chunk_size != 800
    assert old.pipeline.config.final_k == 9
    new = cache.manager(await registry.get("a"))
    chem = cache.manager(await registry.get("chem"))
    await asyncio.gather(new.wait(), chem.wait())
    assert cache.manager(await registry.get("b")) is new
    assert new.pipeline.corpus_dataset_names == ["corpus/bio"]
    assert chem.pipeline.corpus_dataset_names == ["corpus/chem"]


async def test_patch_without_rebuild_stages_settings_until_use(stack):
    client, cache, registry = stack
    response = await client.patch(
        "/admin/rag/config", json={"chunk_size": 800, "rebuild": False}
    )
    assert not response.json()["build_started"]
    manager = cache.manager(await registry.get("a"))
    assert manager.pipeline.calls == []
    assert manager.state.status == "idle"
    await client.get("/admin/rag/config")
    await manager.wait()
    assert len(manager.pipeline.calls) == 1


async def test_global_corpus_selection_is_rejected_and_invalid_patch_is_atomic(stack):
    client, cache, _ = stack
    original = cache.config_snapshot()
    for payload in (
        {"corpus_datasets": ["leak"]},
        {"final_k": 0},
        {"chunk_overlap": 10000, "final_k": 42},
    ):
        assert (
            await client.patch("/admin/rag/config", json=payload)
        ).status_code == 422
    assert cache.config_snapshot() == original


async def test_empty_and_disabled_presets_still_allow_global_configuration(stack):
    client, _, _ = stack
    for name in ("empty", "off"):
        response = await client.get(f"/admin/rag/config?preset={name}")
        assert response.status_code == 200
        assert not response.json()["is_built"]
        assert (
            await client.post(f"/admin/rag/rebuild?preset={name}")
        ).status_code == 400
        assert (
            await client.post(f"/admin/rag/rebuild/cancel?preset={name}")
        ).status_code == 409
    assert (await client.get("/admin/rag/config?preset=missing")).status_code == 404


async def test_cancel_is_shared_but_other_corpora_keep_building_and_retry_works(stack):
    client, cache, registry = stack
    a, chem = (
        cache.manager(await registry.get("a")),
        cache.manager(await registry.get("chem")),
    )
    a.pipeline.release.clear()
    chem.pipeline.release.clear()
    await client.post("/admin/rag/rebuild?preset=a")
    await client.post("/admin/rag/rebuild?preset=chem")
    assert (await client.post("/admin/rag/rebuild?preset=b")).status_code == 409
    assert (await client.post("/admin/rag/rebuild/cancel?preset=b")).status_code == 200
    await a.wait()
    assert a.state.status == "cancelled"
    assert chem.state.status == "building"
    a.pipeline.release.set()
    assert (await client.post("/admin/rag/rebuild?preset=a")).status_code == 202
    await a.wait()
    assert a.pipeline.is_built
    chem.pipeline.release.set()
    await chem.wait()


async def test_failed_rebuild_preserves_the_shared_serving_index(stack):
    client, cache, registry = stack
    await client.get("/admin/rag/config")
    manager = cache.manager(await registry.get("a"))
    await manager.wait()
    manager.pipeline.failure = RuntimeError("upstream refused")
    await client.post("/admin/rag/rebuild")
    await manager.wait()
    response = (await client.get("/admin/rag/config?preset=b")).json()
    assert response["is_built"]
    assert response["build"]["status"] == "failed"
    assert response["build"]["error"] == "upstream refused"


async def test_reset_restores_global_settings_and_reuses_cached_index(stack):
    client, cache, registry = stack
    await client.get("/admin/rag/config")
    original = cache.manager(await registry.get("a"))
    await original.wait()
    await client.patch("/admin/rag/config", json={"chunk_size": 800, "final_k": 8})
    response = await client.post("/admin/rag/reset")
    assert response.status_code == 200
    assert response.json()["config"]["chunk_size"] == RAGConfig().chunk_size
    assert cache.manager(await registry.get("a")) is original
    assert original.pipeline.config.final_k == RAGConfig().final_k


async def test_global_settings_do_not_select_a_preset_or_start_a_build(stack):
    client, cache, registry = stack
    response = await client.get("/admin/rag/settings")
    assert response.status_code == 200
    assert response.json() == cache.config_snapshot()
    assert "preset" not in response.json()
    assert "corpus_datasets" not in response.json()
    assert cache.manager(await registry.get("a")).state.status == "idle"


async def test_global_settings_remain_editable_without_any_presets(stack):
    client, cache, registry = stack
    registry.presets.clear()
    assert (await client.get("/admin/rag/settings")).status_code == 200
    response = await client.patch("/admin/rag/settings", json={"final_k": 9})
    assert response.status_code == 200
    assert response.json()["config"]["final_k"] == 9
    assert response.json()["build_started"] is False
    response = await client.post("/admin/rag/settings/reset")
    assert response.status_code == 200
    assert cache.config_snapshot()["final_k"] == RAGConfig().final_k


async def test_global_settings_validate_atomically_and_reject_corpus_updates(stack):
    client, cache, _ = stack
    before = cache.config_snapshot()
    for payload in (
        {"corpus_datasets": ["corpus/other"]},
        {"chunk_overlap": 10000, "final_k": 25},
    ):
        response = await client.patch("/admin/rag/settings", json=payload)
        assert response.status_code == 422
        assert cache.config_snapshot() == before


async def test_global_build_settings_warm_each_distinct_preset_corpus(stack):
    client, cache, registry = stack
    response = await client.patch("/admin/rag/settings", json={"chunk_size": 800})
    assert response.status_code == 200
    assert response.json()["build_started"]
    first = cache.manager(await registry.get("a"))
    second = cache.manager(await registry.get("chem"))
    assert cache.manager(await registry.get("b")) is first
    await asyncio.gather(first.wait(), second.wait())
    assert first.pipeline.corpus_dataset_names == ["corpus/bio"]
    assert second.pipeline.corpus_dataset_names == ["corpus/chem"]
