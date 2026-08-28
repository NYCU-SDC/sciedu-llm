import os
import urllib.parse
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b"

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_eval_runner, get_langfuse_client
from app.main import app
from judge import RunNotCancellableError, RunState, RunStatus
from rag.config import get_rag_config

_VALID_BODY = {
    "eval_model": "eval-model",
    "judge_model": "judge-model",
    "corpus_datasets": ["corpus/bio"],
    "question_datasets": ["questions/bio"],
    "judge_prompts": ["judge/faithfulness"],
}


class _FakeRunner:
    """Stand-in for `EvalRunner` recording what the router asked it to do.

    `start` builds a real `RunState` (a plain dataclass) so the response
    serialisation under test is the production one — only the scheduling is
    faked away.
    """

    def __init__(self):
        self.states: list[RunState] = []
        self.start_kwargs: list[dict] = []
        self.cancelled: list[str] = []

    def start(self, **kwargs) -> RunState:
        self.start_kwargs.append(kwargs)
        state = RunState(
            run_id=f"run-{len(self.states) + 1}",
            eval_model=kwargs["eval_model"],
            judge_model=kwargs["judge_model"],
            corpus_datasets=list(kwargs["corpus"]),
            question_datasets=list(kwargs["questions"]),
            k=kwargs["k"],
            embedding_model=kwargs["embedding_model"],
            rerank_model=kwargs["rerank_model"],
            chunk_size=kwargs["chunk_size"],
            chunk_overlap=kwargs["chunk_overlap"],
            judge_prompts=list(kwargs["judge_prompts"]),
            max_concurrency=kwargs["max_concurrency"],
            started_at=datetime.now(UTC),
        )
        self.states.append(state)
        return state

    def add(self, run_id: str, *, status: RunStatus, started_at: datetime) -> RunState:
        state = RunState(
            run_id=run_id,
            eval_model="eval-model",
            judge_model="judge-model",
            corpus_datasets=["corpus/bio"],
            question_datasets=["questions/bio"],
            k=5,
            embedding_model="bge-m3",
            rerank_model="reranker",
            chunk_size=500,
            chunk_overlap=100,
            judge_prompts=["judge/faithfulness"],
            max_concurrency=8,
            started_at=started_at,
            status=status,
        )
        self.states.append(state)
        return state

    def list(self) -> list[RunState]:
        return sorted(self.states, key=lambda s: s.started_at, reverse=True)

    def get(self, run_id: str) -> RunState | None:
        return next((s for s in self.states if s.run_id == run_id), None)

    def cancel(self, run_id: str) -> RunState | None:
        state = self.get(run_id)
        if state is None:
            return None
        if state.is_terminal:
            raise RunNotCancellableError(run_id, state.status)
        self.cancelled.append(run_id)
        return state


@pytest.fixture
def runner():
    fake = _FakeRunner()
    app.dependency_overrides[get_eval_runner] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_eval_runner, None)


@pytest.fixture
def client():
    return TestClient(app)


def test_create_run_returns_202_and_fills_config_defaults(client, runner):
    response = client.post("/admin/evals/runs", json=_VALID_BODY)

    assert response.status_code == 202
    body = response.json()
    rag_config = get_rag_config()
    # Fields the client left out are resolved from the server's RAG config.
    assert body["embedding_model"] == rag_config.embedding_model
    assert body["rerank_model"] == rag_config.rerank_model
    assert body["chunk_size"] == rag_config.chunk_size
    assert body["chunk_overlap"] == rag_config.chunk_overlap
    assert body["max_concurrency"] == rag_config.max_concurrency
    assert body["k"] == 5
    assert body["run_id"] == "run-1"
    # The router schedules and answers — it must not have awaited the run.
    assert body["status"] == RunStatus.PENDING.value
    assert body["finished_at"] is None
    assert runner.start_kwargs[0]["corpus"] == ["corpus/bio"]
    assert runner.start_kwargs[0]["questions"] == ["questions/bio"]


def test_create_run_honours_explicit_overrides(client, runner):
    response = client.post(
        "/admin/evals/runs",
        json={
            **_VALID_BODY,
            "k": 12,
            "embedding_model": "custom-embed",
            "rerank_model": "custom-rerank",
            "chunk_size": 300,
            "chunk_overlap": 20,
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["k"] == 12
    assert body["embedding_model"] == "custom-embed"
    assert body["rerank_model"] == "custom-rerank"
    assert body["chunk_size"] == 300
    assert body["chunk_overlap"] == 20


@pytest.mark.parametrize(
    "patch",
    [
        {"corpus_datasets": []},
        {"question_datasets": []},
        {"judge_prompts": []},
        {"eval_model": ""},
        {"k": 0},
        {"k": 21},
    ],
)
def test_create_run_rejects_invalid_payloads(client, runner, patch):
    response = client.post("/admin/evals/runs", json={**_VALID_BODY, **patch})

    assert response.status_code == 422
    assert runner.start_kwargs == []


def test_create_run_rejects_overlap_at_least_chunk_size(client, runner):
    response = client.post(
        "/admin/evals/runs",
        json={**_VALID_BODY, "chunk_size": 200, "chunk_overlap": 200},
    )

    assert response.status_code == 422
    assert runner.start_kwargs == []


def test_create_run_rejects_overlap_beyond_the_configured_chunk_size(client, runner):
    # Only chunk_overlap is pinned, so the conflict is invisible to the schema —
    # the router has to re-check once the RAG config default fills chunk_size in.
    response = client.post(
        "/admin/evals/runs",
        json={**_VALID_BODY, "chunk_overlap": get_rag_config().chunk_size + 1},
    )

    assert response.status_code == 422
    assert "chunk_overlap" in response.json()["detail"]
    assert runner.start_kwargs == []


def test_list_runs_is_newest_first(client, runner):
    now = datetime.now(UTC)
    runner.add("run-old", status=RunStatus.COMPLETED, started_at=now - timedelta(1))
    runner.add("run-new", status=RunStatus.JUDGING, started_at=now)

    response = client.get("/admin/evals/runs")

    assert response.status_code == 200
    assert [r["run_id"] for r in response.json()] == ["run-new", "run-old"]


def test_get_run_returns_the_run(client, runner):
    runner.add("run-x", status=RunStatus.JUDGING, started_at=datetime.now(UTC))

    response = client.get("/admin/evals/runs/run-x")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-x"
    assert response.json()["status"] == "judging"


def test_get_unknown_run_returns_404(client, runner):
    assert client.get("/admin/evals/runs/run-nope").status_code == 404


def test_cancel_run_returns_200(client, runner):
    runner.add("run-x", status=RunStatus.JUDGING, started_at=datetime.now(UTC))

    response = client.post("/admin/evals/runs/run-x/cancel")

    assert response.status_code == 200
    assert runner.cancelled == ["run-x"]


def test_cancel_unknown_run_returns_404(client, runner):
    assert client.post("/admin/evals/runs/run-nope/cancel").status_code == 404


def test_cancel_finished_run_returns_409(client, runner):
    runner.add("run-done", status=RunStatus.COMPLETED, started_at=datetime.now(UTC))

    response = client.post("/admin/evals/runs/run-done/cancel")

    assert response.status_code == 409
    assert "completed" in response.json()["detail"]
    assert runner.cancelled == []


@pytest.fixture
def langfuse():
    def _install(get_runs):
        app.dependency_overrides[get_langfuse_client] = lambda: SimpleNamespace(
            api=SimpleNamespace(datasets=SimpleNamespace(get_runs=get_runs))
        )

    yield _install
    app.dependency_overrides.pop(get_langfuse_client, None)


def test_history_reads_back_langfuse_experiment_runs(client, langfuse):
    created = datetime.now(UTC)

    def get_runs(*, dataset_name, page, limit):  # noqa: ARG001
        # The name arrives percent-encoded, because the generated `api.*`
        # clients drop path parameters into the URL raw. Langfuse itself answers
        # with the decoded name, so the fake does too.
        assert dataset_name == "questions%2Fbio"
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    name="eval-model 2026-08-28",
                    dataset_name=urllib.parse.unquote(dataset_name),
                    created_at=created,
                    description="judge=judge-model",
                )
            ],
            meta=SimpleNamespace(total_pages=1),
        )

    langfuse(get_runs)

    response = client.get("/admin/evals/history?question_dataset=questions/bio")

    assert response.status_code == 200
    (entry,) = response.json()
    assert entry["dataset_name"] == "questions/bio"
    assert entry["run_name"] == "eval-model 2026-08-28"
    assert entry["description"] == "judge=judge-model"
    assert datetime.fromisoformat(entry["created_at"]) == created


def test_history_requires_the_dataset_query_param(client, langfuse):
    langfuse(lambda **_: None)

    assert client.get("/admin/evals/history").status_code == 422


def test_history_returns_502_on_upstream_failure(client, langfuse):
    def boom(**_kwargs):
        raise RuntimeError("langfuse exploded")

    langfuse(boom)

    response = client.get("/admin/evals/history?question_dataset=questions/bio")

    assert response.status_code == 502
    assert "langfuse exploded" in response.json()["detail"]
