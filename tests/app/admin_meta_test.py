import os
from datetime import UTC, datetime
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["ALLOWED_MODELS"] = "gpt-oss-120b"

import httpx
import pytest
from fastapi.testclient import TestClient

from app import listings
from app.agents.tools import registered_tool_names
from app.dependencies import (
    Settings,
    get_langfuse_client,
    get_openai_client,
    get_settings,
)
from app.main import app
from app.presets import Preset
from rag.config import get_rag_config


def _page(names: list[str], total_pages: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(name=n) for n in names],
        meta=SimpleNamespace(total_pages=total_pages),
    )


def _paginated(pages: dict[int, SimpleNamespace]):
    def _list(*, page, limit):  # noqa: ARG001
        return pages[page]

    return _list


def _boom(*, page, limit):  # noqa: ARG001
    raise RuntimeError("langfuse exploded")


def _langfuse(*, prompts=None, datasets=None) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(
            prompts=SimpleNamespace(list=prompts),
            datasets=SimpleNamespace(list=datasets),
        )
    )


def _openai(model_ids: list[str] | None = None, *, exc: Exception | None = None):
    async def _stream():
        if exc is not None:
            raise exc
        for model_id in model_ids or []:
            yield SimpleNamespace(id=model_id)

    return SimpleNamespace(models=SimpleNamespace(list=lambda: _stream()))


# --------------------------------------------------------------------------- #
# app.listings
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_judge_prompt_names_filters_by_folder_and_paginates():
    langfuse = _langfuse(
        prompts=_paginated(
            {
                1: _page(
                    ["judge/zeta", "extract-score", "judge/alpha", "rag-generator"],
                    total_pages=2,
                ),
                2: _page(["judge/mu", "judge/beta"], total_pages=2),
            }
        )
    )

    assert await listings.list_judge_prompt_names(langfuse) == [
        ("alpha", "judge/alpha"),
        ("beta", "judge/beta"),
        ("mu", "judge/mu"),
        ("zeta", "judge/zeta"),
    ]


@pytest.mark.asyncio
async def test_list_judge_prompt_names_raises_on_failure():
    # An empty list would read as "this project has no judge prompts" — the
    # caller has to be able to tell that apart from an unreachable Langfuse.
    with pytest.raises(RuntimeError, match="langfuse exploded"):
        await listings.list_judge_prompt_names(_langfuse(prompts=_boom))


@pytest.mark.asyncio
async def test_list_dataset_names_splits_by_folder():
    langfuse = _langfuse(
        datasets=_paginated(
            {
                1: _page(
                    ["corpus/ver3/biology", "questions/zeta", "scratch/notes"],
                    total_pages=2,
                ),
                2: _page(["questions/alpha", "corpus/ver1/chem"], total_pages=2),
            }
        )
    )

    corpus, questions = await listings.list_dataset_names(
        langfuse, corpus_folder="corpus", questions_folder="questions"
    )

    assert corpus == [
        ("ver1/chem", "corpus/ver1/chem"),
        ("ver3/biology", "corpus/ver3/biology"),
    ]
    assert questions == [
        ("alpha", "questions/alpha"),
        ("zeta", "questions/zeta"),
    ]


@pytest.mark.asyncio
async def test_list_dataset_names_raises_on_failure():
    with pytest.raises(RuntimeError, match="langfuse exploded"):
        await listings.list_dataset_names(
            _langfuse(datasets=_boom), corpus_folder="corpus", questions_folder="q"
        )


@pytest.mark.asyncio
async def test_list_model_ids_returns_sorted_ids():
    # `openai.models.list()` returns an AsyncPaginator (not a coroutine);
    # the implementation must iterate it with `async for`.
    ids = await listings.list_model_ids(_openai(["zeta-7b", "bge-m3", "alpha-1"]))

    assert ids == ["alpha-1", "bge-m3", "zeta-7b"]


@pytest.mark.asyncio
async def test_list_model_ids_raises_on_failure():
    with pytest.raises(RuntimeError, match="upstream 503"):
        await listings.list_model_ids(_openai(exc=RuntimeError("upstream 503")))


# --- list_experiment_runs ---------------------------------------------------
# Langfuse v4 keeps judge runs as *experiments*; a v4 deployment in
# `events_only` mode answers the older dataset-runs endpoint with a 404 saying
# so. The listing reads experiments and keeps the old endpoint as a fallback, so
# both shapes are covered here.

SINCE = datetime(2026, 6, 1, tzinfo=UTC)


def _runs_page(names, total_pages):
    return SimpleNamespace(
        data=[SimpleNamespace(name=n, created_at=c) for n, c in names],
        meta=SimpleNamespace(total_pages=total_pages),
    )


def _dataset_runs_langfuse():
    """A Langfuse with no reachable REST transport, so only the old path exists."""

    def get_runs(*, dataset_name, page, limit):  # noqa: ARG001
        # Percent-encoded: the generated `api.*` clients interpolate path
        # parameters raw, so a foldered name has to arrive already quoted or it
        # addresses `/datasets/questions/biology/runs` and 404s.
        assert dataset_name == "questions%2Fbiology"
        return {
            1: _runs_page([("old", 1), ("new", 3)], total_pages=2),
            2: _runs_page([("mid", 2)], total_pages=2),
        }[page]

    return SimpleNamespace(
        api=SimpleNamespace(datasets=SimpleNamespace(get_runs=get_runs))
    )


def _experiments_langfuse(handler):
    """A Langfuse whose SDK internals expose a base URL, plus a dataset id lookup."""
    wrapper = SimpleNamespace(
        get_base_url=lambda: "https://langfuse.test/",
        get_headers=lambda: {"Authorization": "Basic pk:sk"},
    )
    api = SimpleNamespace(
        datasets=SimpleNamespace(
            get=lambda dataset_name: SimpleNamespace(id="ds-42"),
            get_runs=handler,
        )
    )
    api._client_wrapper = wrapper
    return SimpleNamespace(api=api)


def _experiment(name, start_time):
    return {
        "id": f"exp-{name}",
        "name": name,
        "description": f"judge run {name}",
        "startTime": start_time,
        "endTime": start_time,
        "itemCount": 3,
        "datasetId": "ds-42",
    }


@pytest.mark.asyncio
async def test_list_experiment_runs_reads_experiments_and_follows_the_cursor():
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/experiments"
        params = dict(request.url.params)
        seen.append(params)
        assert params["datasetId"] == "ds-42"
        assert params["fromStartTime"] == SINCE.isoformat()
        assert request.headers["Authorization"] == "Basic pk:sk"
        if "cursor" not in params:
            return httpx.Response(
                200,
                json={
                    "data": [
                        _experiment("old", "2026-06-02T10:00:00Z"),
                        _experiment("new", "2026-08-20T10:00:00Z"),
                    ],
                    "meta": {"cursor": "page-2"},
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [_experiment("mid", "2026-07-05T10:00:00Z")],
                "meta": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        runs = await listings.list_experiment_runs(
            _experiments_langfuse(None), "questions/biology", since=SINCE, http=http
        )

    assert [run.name for run in runs] == ["new", "mid", "old"]
    assert [run.dataset_name for run in runs] == ["questions/biology"] * 3
    assert runs[0].created_at == datetime(2026, 8, 20, 10, tzinfo=UTC)
    assert runs[0].description == "judge run new"
    # Two pages, and the second one carried the cursor from the first.
    assert len(seen) == 2 and seen[1]["cursor"] == "page-2"


@pytest.mark.asyncio
async def test_list_experiment_runs_falls_back_when_experiments_are_404():
    """An older deployment has no /experiments; the deprecated endpoint answers."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    def get_runs(*, dataset_name, page, limit):  # noqa: ARG001
        assert dataset_name == "questions%2Fbiology"
        return _runs_page([("only", 1)], total_pages=1)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        runs = await listings.list_experiment_runs(
            _experiments_langfuse(get_runs), "questions/biology", since=SINCE, http=http
        )

    assert [run.name for run in runs] == ["only"]


@pytest.mark.asyncio
async def test_list_experiment_runs_propagates_other_experiment_failures():
    """Only a 404 means "wrong endpoint" — a 500 is an outage and must surface."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "langfuse exploded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await listings.list_experiment_runs(
                _experiments_langfuse(None),
                "questions/biology",
                since=SINCE,
                http=http,
            )


@pytest.mark.asyncio
async def test_list_experiment_runs_sorted_newest_first_on_the_old_endpoint():
    runs = await listings.list_experiment_runs(
        _dataset_runs_langfuse(), "questions/biology", since=SINCE
    )

    assert [r.name for r in runs] == ["new", "mid", "old"]


# --------------------------------------------------------------------------- #
# routers/admin/meta.py
# --------------------------------------------------------------------------- #


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def overrides():
    """Install dependency overrides and tear them all down afterwards."""
    installed: list = []

    def _install(dependency, value):
        app.dependency_overrides[dependency] = lambda: value
        installed.append(dependency)

    _install(
        get_settings,
        Settings(
            openai_api_key="mock_key",
            openai_default_model="default-model",
            allowed_models="alpha-1,bge-m3",
            corpus_dataset_folder="corpus",
            questions_dataset_folder="questions",
        ),
    )
    yield _install
    for dependency in installed:
        app.dependency_overrides.pop(dependency, None)


def test_get_models_reports_full_listing_and_defaults(client, overrides):
    overrides(get_openai_client, _openai(["zeta-7b", "bge-m3", "alpha-1"]))

    response = client.get("/admin/models")

    assert response.status_code == 200
    body = response.json()
    # Unfiltered: ALLOWED_MODELS governs /chat, not what an admin may evaluate with.
    assert body["models"] == ["alpha-1", "bge-m3", "zeta-7b"]
    assert body["allowed_models"] == ["alpha-1", "bge-m3"]
    rag_config = get_rag_config()
    assert body["defaults"] == {
        "eval_model": "default-model",
        "judge_model": "default-model",
        "embedding_model": rag_config.embedding_model,
        "rerank_model": rag_config.rerank_model,
    }


def test_get_models_returns_502_on_upstream_failure(client, overrides):
    overrides(get_openai_client, _openai(exc=RuntimeError("upstream 503")))

    response = client.get("/admin/models")

    assert response.status_code == 502
    assert "upstream 503" in response.json()["detail"]


def test_get_datasets_groups_by_folder(client, overrides):
    overrides(
        get_langfuse_client,
        _langfuse(
            datasets=_paginated(
                {1: _page(["corpus/ver3/bio", "questions/bio", "other"], 1)}
            )
        ),
    )

    response = client.get("/admin/datasets")

    assert response.status_code == 200
    assert response.json() == {
        "corpus": [{"name": "corpus/ver3/bio", "label": "ver3/bio"}],
        "questions": [{"name": "questions/bio", "label": "bio"}],
    }


def test_get_datasets_returns_502_on_upstream_failure(client, overrides):
    overrides(get_langfuse_client, _langfuse(datasets=_boom))

    assert client.get("/admin/datasets").status_code == 502


def test_get_judge_prompts_strips_folder_prefix(client, overrides):
    overrides(
        get_langfuse_client,
        _langfuse(prompts=_paginated({1: _page(["judge/beta", "unrelated"], 1)})),
    )

    response = client.get("/admin/judge-prompts")

    assert response.status_code == 200
    assert response.json() == [{"name": "judge/beta", "label": "beta"}]


def test_get_judge_prompts_returns_502_on_upstream_failure(client, overrides):
    overrides(get_langfuse_client, _langfuse(prompts=_boom))

    assert client.get("/admin/judge-prompts").status_code == 502


def test_get_tools_reports_the_registry(client):
    # No upstream at all: the tool registry is code, so this endpoint cannot 502
    # and needs no dependency overrides.
    response = client.get("/admin/tools")

    assert response.status_code == 200
    body = response.json()
    assert [tool["name"] for tool in body] == registered_tool_names()
    by_name = {tool["name"]: tool for tool in body}
    assert by_name["rag_search"]["requires_rag"] is True
    assert by_name["rag_search"]["internal"] is False
    # The summon mechanism is plumbing a frontend hides, and needs no corpus.
    assert by_name["summon_subagent"]["internal"] is True
    assert by_name["summon_subagent"]["requires_rag"] is False
    assert all(tool["description"] for tool in body)


def test_get_tools_lists_exactly_what_a_preset_may_name(client):
    # The preset validator checks `tools` against this same registry, so a name
    # offered here is a name a preset can be saved with.
    names = [tool["name"] for tool in client.get("/admin/tools").json()]

    assert (
        Preset.model_validate(
            {
                "name": "every-tool",
                "orchestrator": "assistant",
                "characters": [
                    {"id": "assistant", "display_name": "助教", "tools": names},
                    {
                        "id": "student",
                        "display_name": "學生",
                        "prompt_name": "agents/student",
                    },
                ],
            }
        )
        .characters[0]
        .tools
        == names
    )
