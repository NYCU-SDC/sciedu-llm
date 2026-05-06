"""End-to-end RAG smoke test.

Builds the pipeline from the three biology corpus datasets on Langfuse, runs a
generation against a sample question, and prints the answer alongside the
retrieved reference chunks.

Reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_DEFAULT_MODEL`, and the usual
Langfuse credentials from the environment (or `.env`).

Usage:
    uv run python scripts/rag_smoke.py
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from observability import init_langfuse_client
from rag import RAGPipeline

load_dotenv()

QUERY = (
    "為了維護台灣的生物多樣性，政府設立了國家公園與生態保護區。"
    "請說明設立這些保護區的主要目的，"
    "以及這與「擴大農業耕地」在土地利用概念上有何本質上的不同？"
)

CORPUS_DATASETS = [
    "corpus-biology-10",
]


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-oss-120b")

    openai_client = AsyncOpenAI()  # picks up OPENAI_API_KEY / OPENAI_BASE_URL
    langfuse_client = init_langfuse_client()

    pipeline = RAGPipeline(openai_client, langfuse_client)

    print(f"Building RAG pipeline from corpora: {CORPUS_DATASETS}")
    build_start = time.perf_counter()
    await pipeline.build(CORPUS_DATASETS)
    chunker = pipeline._chunker  # noqa: SLF001 — smoke script uses internals to display chunks
    assert chunker is not None
    chunks = chunker.chunks
    chapters = sorted({chunk.chapter for chunk in chunks})
    print(
        f"Built in {time.perf_counter() - build_start:.1f}s — "
        f"{len(chunks)} chunks across {len(chapters)} chapter(s): {chapters}"
    )

    print(f"\nQuery:\n{QUERY}\n")
    print(f"Generating with model: {model}")

    gen_start = time.perf_counter()
    result = await pipeline.generate(query=QUERY, model=model)
    print(f"Generation took {time.perf_counter() - gen_start:.1f}s")

    print("\n=== ANSWER ===")
    print(result["output_text"])

    print("\n=== REFERENCE CHUNKS ===")
    for chunk_id in result["reference_chunks"]:
        chunk = chunks[chunk_id]
        print(
            f"\n--- chunk {chunk_id} | {chunk.chapter} | "
            f"[{chunk.start}, {chunk.end}) ---"
        )
        print(chunk.text)

    langfuse_client.flush()


if __name__ == "__main__":
    asyncio.run(main())
