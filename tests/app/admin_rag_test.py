"""The `/admin/rag` surface.

These are async and drive the app through `ASGITransport` rather than
`TestClient`, because rebuilds are now background tasks: `TestClient` opens a
fresh event loop per request and tears it down with the response, which would
throw away the very task being asserted on. One loop per test keeps a scheduled
build alive across the requests that inspect and cancel it.
"""

import asyncio
import os

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b"
# Pin the retrieval knobs so reset-to-env assertions are deterministic regardless
# of any repo-local .env (os.environ takes precedence over load_dotenv).
os.environ["RAG_FINAL_K"] = "5"
os.environ["RAG_CHUNK_SIZE"] = "500"

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.dependencies import get_rag_pipeline
from app.main import app
from rag.config import RAGConfig

pytestmark = pytest.mark.asyncio


class _FakeRAGPipeline:
    """Minimal stand-in mirroring the pipeline surface the admin router uses.

    `hold` gates `build()` so a test can keep one running while it asserts on
    what the API says about it; `fail_with` makes it blow up instead.
    """

    def __init__(self, *, hold: bool = False, fail_with: Exception | None = None):
        self._values = RAGConfig().model_dump()
        self.is_built = True
        self.corpus_dataset_names = ["biology"]
        self.build_calls: list = []
        self.retrieve_calls: list = []
        self.cancelled = 0
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self._fail_with = fail_with

    def config_snapshot(self) -> dict:
        return dict(self._values)

    def apply_overrides(self, overrides: dict) -> None:
        self._values.update(overrides)

    async def build(self, corpus_dataset_names, **kwargs) -> None:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self._fail_with is not None:
            raise self._fail_with
        self.build_calls.append(list(corpus_dataset_names))
        self.corpus_dataset_names = list(corpus_dataset_names)
        self.is_built = True

    async def retrieve(self, *, query: str, **kwargs):
        self.retrieve_calls.append((query, kwargs))
        return {"context": "", "reference_chunks": []}


@pytest.fixture
def override_rag():
    def _install(pipeline):
        app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
        # A manager left on app.state by an earlier test wraps a dead pipeline;
        # the dependency re-wraps when the pipeline changes, but clearing it
        # keeps a *None* pipeline from finding a stale one.
        app.state.rag_build_manager = None
        return pipeline

    yield _install
    app.dependency_overrides.pop(get_rag_pipeline, None)
    app.state.rag_build_manager = None


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _settle():
    """Let a scheduled build task run to completion."""
    for _ in range(20):
        await asyncio.sleep(0)


async def test_get_rag_config_returns_current_values(client, override_rag):
    override_rag(_FakeRAGPipeline())

    response = await client.get("/admin/rag/config")

    assert response.status_code == 200
    body = response.json()
    assert body["final_k"] == 5
    assert body["chunk_size"] == 500
    assert body["is_built"] is True
    assert body["corpus_datasets"] == ["biology"]
    # Nothing has been built *through the admin API* in this process.
    assert body["build"]["status"] == "idle"
    assert body["build"]["cancel_requested"] is False


async def test_patch_schedules_a_build_by_default(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    response = await client.patch("/admin/rag/config", json={"final_k": 8})

    assert response.status_code == 200
    body = response.json()
    assert body["build_started"] is True
    assert body["config"]["final_k"] == 8
    # The response goes out while the build is still running.
    assert body["config"]["build"]["status"] == "building"

    await _settle()
    assert pipeline.build_calls == [["biology"]]
    later = await client.get("/admin/rag/config")
    assert later.json()["build"]["status"] == "completed"


async def test_patch_rebuild_false_applies_without_building(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    response = await client.patch(
        "/admin/rag/config", json={"chunk_size": 400, "rebuild": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["build_started"] is False
    assert body["config"]["chunk_size"] == 400
    assert body["config"]["build"]["status"] == "idle"

    await _settle()
    assert pipeline.build_calls == []


async def test_patch_corpus_datasets_reindexes_and_forces_a_build(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    # rebuild=False is ignored for a corpus change — it must re-index to take effect.
    response = await client.patch(
        "/admin/rag/config",
        json={"corpus_datasets": ["chemistry", "physics"], "rebuild": False},
    )

    assert response.status_code == 200
    assert response.json()["build_started"] is True

    await _settle()
    assert pipeline.build_calls == [["chemistry", "physics"]]
    later = await client.get("/admin/rag/config")
    assert later.json()["corpus_datasets"] == ["chemistry", "physics"]


async def test_patch_empty_corpus_datasets_returns_400(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    response = await client.patch("/admin/rag/config", json={"corpus_datasets": []})

    assert response.status_code == 400
    await _settle()
    assert pipeline.build_calls == []


async def test_patch_rejects_out_of_range_value(client, override_rag):
    override_rag(_FakeRAGPipeline())

    response = await client.patch("/admin/rag/config", json={"final_k": 0})

    assert response.status_code == 422


async def test_admin_endpoints_return_503_when_rag_disabled(client, override_rag):
    override_rag(None)

    assert (await client.get("/admin/rag/config")).status_code == 503
    assert (
        await client.patch("/admin/rag/config", json={"final_k": 8})
    ).status_code == 503
    assert (await client.post("/admin/rag/rebuild")).status_code == 503
    assert (await client.post("/admin/rag/rebuild/cancel")).status_code == 503
    assert (await client.post("/admin/rag/reset")).status_code == 503


async def test_rebuild_answers_202_and_builds_in_the_background(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    response = await client.post("/admin/rag/rebuild")

    assert response.status_code == 202
    assert response.json()["build"]["status"] == "building"

    await _settle()
    assert pipeline.build_calls == [["biology"]]


async def test_a_second_build_while_one_runs_is_a_conflict(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline(hold=True))

    assert (await client.post("/admin/rag/rebuild")).status_code == 202
    await _settle()

    conflict = await client.post("/admin/rag/rebuild")
    assert conflict.status_code == 409

    pipeline.release.set()
    await _settle()
    assert pipeline.build_calls == [["biology"]]


async def test_cancel_stops_the_running_build(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline(hold=True))

    await client.post("/admin/rag/rebuild")
    await _settle()

    response = await client.post("/admin/rag/rebuild/cancel")

    assert response.status_code == 200
    # Accepted, but the task has not unwound yet.
    assert response.json()["build"]["cancel_requested"] is True

    await _settle()
    body = (await client.get("/admin/rag/config")).json()
    assert body["build"]["status"] == "cancelled"
    assert pipeline.cancelled == 1
    # Nothing was installed, so the indexes that were answering still are.
    assert pipeline.build_calls == []
    assert body["is_built"] is True

    # And the pipeline is free to build again.
    pipeline.release.set()
    assert (await client.post("/admin/rag/rebuild")).status_code == 202


async def test_cancelling_reverts_the_build_time_settings_it_would_have_baked(
    client, override_rag
):
    """A killed build must not leave the API describing an index that never was.

    The change is applied before the rebuild (the build needs the new value to
    build with), so cancelling has to put it back — otherwise `GET /config`
    reports a chunk size the serving index does not have, which is what the
    console shows as "the settings currently in use".
    """
    override_rag(_FakeRAGPipeline(hold=True))

    await client.patch("/admin/rag/config", json={"chunk_size": 400, "final_k": 9})
    await _settle()

    # While it builds, the pipeline really is configured for the new value.
    during = (await client.get("/admin/rag/config")).json()
    assert during["chunk_size"] == 400

    await client.post("/admin/rag/rebuild/cancel")
    await _settle()

    after = (await client.get("/admin/rag/config")).json()
    assert after["build"]["status"] == "cancelled"
    assert after["chunk_size"] == 500
    # Retrieval knobs are not baked into anything — they applied at once and stay.
    assert after["final_k"] == 9


async def test_a_failed_build_also_reverts_its_build_time_settings(
    client, override_rag
):
    override_rag(_FakeRAGPipeline(fail_with=RuntimeError("upstream refused")))

    await client.patch("/admin/rag/config", json={"chunk_overlap": 250})
    await _settle()

    after = (await client.get("/admin/rag/config")).json()
    assert after["build"]["status"] == "failed"
    assert after["chunk_overlap"] == 100


async def test_a_completed_build_keeps_its_build_time_settings(client, override_rag):
    override_rag(_FakeRAGPipeline())

    await client.patch("/admin/rag/config", json={"chunk_size": 400})
    await _settle()

    after = (await client.get("/admin/rag/config")).json()
    assert after["build"]["status"] == "completed"
    assert after["chunk_size"] == 400


async def test_cancel_with_nothing_running_is_a_conflict(client, override_rag):
    override_rag(_FakeRAGPipeline())

    response = await client.post("/admin/rag/rebuild/cancel")

    assert response.status_code == 409


async def test_a_failed_build_is_reported_on_the_config(client, override_rag):
    override_rag(_FakeRAGPipeline(fail_with=RuntimeError("upstream refused")))

    # Starting one still succeeds — the failure happens after the response.
    assert (await client.post("/admin/rag/rebuild")).status_code == 202
    await _settle()

    build = (await client.get("/admin/rag/config")).json()["build"]
    assert build["status"] == "failed"
    assert "upstream refused" in build["error"]


async def test_reset_restores_env_defaults_and_builds(client, override_rag):
    pipeline = override_rag(_FakeRAGPipeline())

    # Drift a live knob away from the env default (without rebuilding), then reset.
    await client.patch("/admin/rag/config", json={"final_k": 8, "rebuild": False})
    assert pipeline.config_snapshot()["final_k"] == 8
    assert pipeline.build_calls == []

    response = await client.post("/admin/rag/reset")

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["final_k"] == 5
    # Reset always rebuilds.
    assert body["build_started"] is True

    await _settle()
    assert pipeline.build_calls == [["biology"]]
