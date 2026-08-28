"""What a build says about itself while it runs.

A real re-index is minutes of embedding calls with no upstream progress signal,
so the service log is the only place an operator can watch one. These assert the
lines that make that possible — the settings a build started with, each dataset
as it is collected, how many embedding batches have come back, and the totals at
the end — because "the build is logged" is a feature, not an implementation
detail, and silence would look identical to a hang.
"""

import logging
import re
from types import SimpleNamespace

import pytest

from rag import pipeline as pipeline_module
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
async def test_build_logs_its_settings_datasets_progress_and_totals(
    caplog, monkeypatch
):
    # Progress is throttled to one line every few seconds in production; at zero
    # every batch reports, which is what makes the cadence assertable at all.
    monkeypatch.setattr(pipeline_module, "_PROGRESS_INTERVAL_SECONDS", 0.0)

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await _pipeline().build(["corpus/a", "corpus/b"])

    text = caplog.text

    # What it is about to do, in enough detail to tell two rebuilds apart.
    assert "Index build starting" in text
    assert "datasets=corpus/a, corpus/b" in text
    assert "chunk_size=20" in text

    # Each dataset as it lands — the slow part before embedding even starts.
    assert "Collected dataset 1/2 'corpus/a'" in text
    assert "Collected dataset 2/2 'corpus/b'" in text

    # Periodic progress, ending on a line that accounts for every batch.
    progress = re.findall(r"Embedding (\d+)/(\d+) batches \((\d+)/(\d+) chunks", text)
    assert progress, f"no embedding progress was logged:\n{text}"
    done, total, chunks, total_chunks = progress[-1]
    assert done == total and chunks == total_chunks
    assert [int(entry[0]) for entry in progress] == list(range(1, int(total) + 1))

    # And the totals, so a finished build is unambiguous in the log.
    assert re.search(
        rf"Index build finished \| {total_chunks} chunk\(s\), 2 chapter\(s\) "
        r"from 2 dataset\(s\) in \d+s",
        text,
    )


@pytest.mark.asyncio
async def test_embedding_progress_is_throttled_to_one_line_per_interval(
    caplog, monkeypatch
):
    """The default interval is long, so a quick build logs progress once: at the end.

    Without this the log would carry a line per batch — hundreds of them for a
    real corpus, which is how a useful signal becomes noise nobody reads.
    """
    monkeypatch.setattr(pipeline_module, "_PROGRESS_INTERVAL_SECONDS", 3600.0)

    with caplog.at_level(logging.INFO, logger="rag.pipeline"):
        await _pipeline().build(["corpus/a", "corpus/b"])

    progress = re.findall(r"Embedding (\d+)/(\d+) batches", caplog.text)
    assert len(progress) == 1
    done, total = progress[0]
    assert done == total
