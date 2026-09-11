"""Progress and lifecycle reporting for index builds."""

import logging
import re
import threading
from types import SimpleNamespace

import pytest

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline

CHAPTERS = {
    "corpus/a": [("ch1", "光反應發生在類囊體膜上。" * 4)],
    "corpus/b": [("ch2", "暗反應固定二氧化碳。" * 4)],
}


def _fake_langfuse():
    def get_dataset(name):
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    id=f"{name}-{index}",
                    metadata={"chapter": chapter},
                    input={"content": content},
                )
                for index, (chapter, content) in enumerate(CHAPTERS[name])
            ]
        )

    return SimpleNamespace(get_dataset=get_dataset)


def _fake_openai():
    async def create(*, model, input):  # noqa: A002 — mirrors the SDK's kwarg
        texts = input if isinstance(input, list) else [input]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in texts]
        )

    return SimpleNamespace(
        base_url="http://upstream.test/v1",
        api_key="fake",
        embeddings=SimpleNamespace(create=create),
    )


def _pipeline() -> RAGPipeline:
    # One chunk per batch, so the embedding pass really is several calls.
    config = RAGConfig(chunk_size=20, chunk_overlap=0, embedding_batch_size=1)
    return RAGPipeline(_fake_openai(), _fake_langfuse(), config=config)


@pytest.mark.asyncio
async def test_build_uses_tqdm_and_logs_its_settings_and_totals(caplog, capsys):
    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await _pipeline().build(["corpus/a", "corpus/b"])

    text = caplog.text
    progress = capsys.readouterr().err

    # What it is about to do, in enough detail to tell two rebuilds apart.
    assert "Index build starting" in text
    assert "datasets=corpus/a, corpus/b" in text
    assert "chunk_size=20" in text

    # Both asynchronous phases expose tqdm progress without replacing lifecycle
    # and summary messages from the logging module.
    assert "Fetching RAG datasets" in progress
    assert "Embedding RAG chunks" in progress

    # And the totals, so a finished build is unambiguous in the log.
    assert re.search(
        r"Index build finished \| \d+ chunk\(s\), 2 chapter\(s\) "
        r"from 2 dataset\(s\) in \d+s",
        text,
    )


@pytest.mark.asyncio
async def test_build_fetches_datasets_in_parallel():
    barrier = threading.Barrier(len(CHAPTERS))

    def get_dataset(name):
        # A sequential implementation times out here on its first fetch. Both
        # calls must be in worker threads together before either can return.
        barrier.wait(timeout=2)
        return _fake_langfuse().get_dataset(name)

    pipeline = _pipeline()
    pipeline._langfuse = SimpleNamespace(get_dataset=get_dataset)

    await pipeline.build(["corpus/a", "corpus/b"])

    # gather returns results in input order even if requests finish out of order,
    # so parallel fetching must not make stable chunk IDs nondeterministic.
    chapter_one = pipeline.resolve_chunks("ch1", 0, 10_000)
    chapter_two = pipeline.resolve_chunks("ch2", 0, 10_000)
    assert max(chapter_one) < min(chapter_two)
