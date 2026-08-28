import os
from types import SimpleNamespace

os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["OPENAI_DEFAULT_MODEL"] = "gpt-oss-120b"

import pytest

from app import dependencies
from app.dependencies import Settings, build_rag_pipeline, validate_allowed_models


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


# --- build_rag_pipeline -----------------------------------------------------
# RAG_CORPUS_DATASETS is optional: an unset one means "index whatever corpus
# datasets this Langfuse project has", so a deployment does not keep the same
# list in two places. Discovery failing is not a startup failure — it is RAG
# staying off, exactly as an empty configuration has always been.


class _FakePipeline:
    def __init__(self, *_args):
        self.built: list[str] | None = None

    async def build(self, names):
        self.built = list(names)


@pytest.fixture
def rag_stack(monkeypatch):
    """Install the pieces `build_rag_pipeline` reaches for, and report the calls."""
    built: list[_FakePipeline] = []
    listed: list[dict] = []

    def _install(*, corpus_datasets="", discovered=None, exc=None):
        settings = Settings(
            openai_api_key="mock_key",
            rag_corpus_datasets=corpus_datasets,
            corpus_dataset_folder="corpus",
            questions_dataset_folder="questions",
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
        monkeypatch.setattr(
            dependencies, "get_openai_client", lambda: SimpleNamespace()
        )
        monkeypatch.setattr(
            dependencies, "get_langfuse_client", lambda: SimpleNamespace()
        )

        def _make(*args):
            pipeline = _FakePipeline(*args)
            built.append(pipeline)
            return pipeline

        monkeypatch.setattr(dependencies, "RAGPipeline", _make)

        async def _list_dataset_names(_langfuse, *, corpus_folder, questions_folder):
            listed.append(
                {"corpus_folder": corpus_folder, "questions_folder": questions_folder}
            )
            if exc is not None:
                raise exc
            return [(name.split("/", 1)[-1], name) for name in discovered or []], []

        monkeypatch.setattr(
            dependencies.listings, "list_dataset_names", _list_dataset_names
        )
        return SimpleNamespace(built=built, listed=listed)

    return _install


@pytest.mark.asyncio
async def test_build_rag_pipeline_indexes_the_configured_datasets(rag_stack):
    calls = rag_stack(
        corpus_datasets="corpus/bio, corpus/chem", discovered=["corpus/x"]
    )

    pipeline = await build_rag_pipeline()

    assert pipeline is not None
    assert pipeline.built == ["corpus/bio", "corpus/chem"]
    # An explicit list pins the set — discovery is not even attempted.
    assert calls.listed == []


@pytest.mark.asyncio
async def test_build_rag_pipeline_discovers_the_corpus_when_unconfigured(rag_stack):
    calls = rag_stack(discovered=["corpus/ver3/bio", "corpus/chem"])

    pipeline = await build_rag_pipeline()

    assert pipeline is not None
    assert pipeline.built == ["corpus/ver3/bio", "corpus/chem"]
    assert calls.listed == [
        {"corpus_folder": "corpus", "questions_folder": "questions"}
    ]


@pytest.mark.asyncio
async def test_build_rag_pipeline_disables_rag_when_nothing_is_discovered(rag_stack):
    calls = rag_stack(discovered=[])

    assert await build_rag_pipeline() is None
    assert calls.built == []


@pytest.mark.asyncio
async def test_build_rag_pipeline_disables_rag_when_discovery_fails(rag_stack, caplog):
    calls = rag_stack(exc=RuntimeError("langfuse is down"))

    with caplog.at_level("WARNING"):
        assert await build_rag_pipeline() is None

    # RAG off and a warning — not a failed startup.
    assert calls.built == []
    assert "discover a RAG corpus" in caplog.text
