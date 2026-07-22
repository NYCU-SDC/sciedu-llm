import os

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b"
# Pin the retrieval knobs so reset-to-env assertions are deterministic regardless
# of any repo-local .env (os.environ takes precedence over load_dotenv).
os.environ["RAG_FINAL_K"] = "5"
os.environ["RAG_CHUNK_SIZE"] = "500"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_rag_pipeline
from app.main import app
from rag.config import RAGConfig


class _FakeRAGPipeline:
    """Minimal stand-in mirroring the pipeline surface the admin router uses."""

    def __init__(self):
        self._values = RAGConfig().model_dump()
        self.is_built = True
        self.corpus_dataset_names = ["biology"]
        self.rebuild_calls: list = []
        self.build_calls: list = []
        self.retrieve_calls: list = []

    def config_snapshot(self) -> dict:
        return dict(self._values)

    def apply_overrides(self, overrides: dict) -> None:
        self._values.update(overrides)

    async def rebuild(self) -> None:
        self.rebuild_calls.append(True)

    async def build(self, corpus_dataset_names, **kwargs) -> None:
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

    yield _install
    app.dependency_overrides.pop(get_rag_pipeline, None)


@pytest.fixture
def client():
    return TestClient(app)


def test_get_rag_config_returns_current_values(client, override_rag):
    override_rag(_FakeRAGPipeline())

    response = client.get("/admin/rag/config")

    assert response.status_code == 200
    body = response.json()
    assert body["final_k"] == 5
    assert body["chunk_size"] == 500
    assert body["is_built"] is True
    assert body["corpus_datasets"] == ["biology"]


def test_patch_rebuilds_by_default(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.patch("/admin/rag/config", json={"final_k": 8})

    assert response.status_code == 200
    body = response.json()
    assert body["rebuilt"] is True
    assert body["config"]["final_k"] == 8
    assert len(pipeline.rebuild_calls) == 1


def test_patch_rebuild_false_applies_without_rebuild(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.patch(
        "/admin/rag/config", json={"chunk_size": 400, "rebuild": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rebuilt"] is False
    assert body["config"]["chunk_size"] == 400
    assert pipeline.rebuild_calls == []


def test_patch_corpus_datasets_reindexes_and_forces_rebuild(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    # rebuild=False is ignored for a corpus change — it must re-index to take effect.
    response = client.patch(
        "/admin/rag/config",
        json={"corpus_datasets": ["chemistry", "physics"], "rebuild": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rebuilt"] is True
    assert body["config"]["corpus_datasets"] == ["chemistry", "physics"]
    assert pipeline.build_calls == [["chemistry", "physics"]]
    # build() re-indexes from the new corpus; rebuild() (old corpus) is not called.
    assert pipeline.rebuild_calls == []


def test_patch_empty_corpus_datasets_returns_400(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.patch("/admin/rag/config", json={"corpus_datasets": []})

    assert response.status_code == 400
    assert pipeline.build_calls == []


def test_patch_rejects_out_of_range_value(client, override_rag):
    override_rag(_FakeRAGPipeline())

    response = client.patch("/admin/rag/config", json={"final_k": 0})

    assert response.status_code == 422


def test_admin_endpoints_return_503_when_rag_disabled(client, override_rag):
    override_rag(None)

    assert client.get("/admin/rag/config").status_code == 503
    assert client.patch("/admin/rag/config", json={"final_k": 8}).status_code == 503
    assert client.post("/admin/rag/rebuild").status_code == 503
    assert client.post("/admin/rag/reset").status_code == 503


def test_rebuild_invokes_pipeline_build(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    response = client.post("/admin/rag/rebuild")

    assert response.status_code == 200
    assert len(pipeline.rebuild_calls) == 1


def test_reset_restores_env_defaults(client, override_rag):
    pipeline = _FakeRAGPipeline()
    override_rag(pipeline)

    # Drift a live knob away from the env default (without rebuilding), then reset.
    client.patch("/admin/rag/config", json={"final_k": 8, "rebuild": False})
    assert pipeline.config_snapshot()["final_k"] == 8
    assert pipeline.rebuild_calls == []

    response = client.post("/admin/rag/reset")

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["final_k"] == 5
    # Reset always rebuilds.
    assert body["rebuilt"] is True
    assert len(pipeline.rebuild_calls) == 1
