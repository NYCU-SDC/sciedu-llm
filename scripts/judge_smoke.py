"""End-to-end judge smoke test.

Builds the pipeline from the biology corpus dataset on Langfuse, then runs the
judge module against the `questions-biology` question dataset and prints the
per-dataset summary plus the Langfuse dataset run URL.

Reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_DEFAULT_MODEL`,
`JUDGE_MODEL` (optional, defaults to `OPENAI_DEFAULT_MODEL`), and the usual
Langfuse credentials from the environment (or `.env`).

Usage:
    uv run python scripts/judge_smoke.py
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from judge import Judge
from observability import init_langfuse_client
from rag import RAGPipeline

load_dotenv()

CORPUS_DATASETS = [
    "corpus-biology-10",
]
QUESTION_DATASETS = [
    "questions-biology",
]
K = 5


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    eval_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-oss-120b")
    judge_model = os.getenv("JUDGE_MODEL", eval_model)

    openai_client = AsyncOpenAI()  # picks up OPENAI_API_KEY / OPENAI_BASE_URL
    langfuse_client = init_langfuse_client()

    pipeline = RAGPipeline(openai_client, langfuse_client)

    print(f"Building RAG pipeline from corpora: {CORPUS_DATASETS}")
    build_start = time.perf_counter()
    await pipeline.build(CORPUS_DATASETS)
    print(f"Built in {time.perf_counter() - build_start:.1f}s")

    judge = Judge(
        pipeline=pipeline,
        openai=openai_client,
        langfuse=langfuse_client,
        judge_model=judge_model,
        eval_model=eval_model,
        k=K,
    )

    print(
        f"\nRunning judge on {QUESTION_DATASETS} | "
        f"eval={eval_model} judge={judge_model} k={K}"
    )
    print(f"Session ID: {judge.session_id}")

    run_start = time.perf_counter()
    results = await judge.run(QUESTION_DATASETS, CORPUS_DATASETS)
    print(f"\nJudge run took {time.perf_counter() - run_start:.1f}s")

    for experiment in results:
        print("\n" + experiment.format())

    langfuse_client.flush()


if __name__ == "__main__":
    asyncio.run(main())
