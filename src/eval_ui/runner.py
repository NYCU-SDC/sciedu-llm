"""Process-global registry for background evaluation runs.

A `Start` click on the Gradio UI calls `EvalRunner.start(...)`, which constructs
a `RunState`, schedules `_execute(...)` on the event loop via
`asyncio.create_task`, and returns immediately. The handler does NOT await the
task — closing the browser tab severs the WebSocket but the task is owned by
Gradio's persistent uvicorn loop and keeps running until completion.

Strong reference invariant: `state._task` keeps the task alive for the GC; do
not "clean up" the apparently-unused field.

Caveats:
- State is in-memory only. If the Gradio process dies, run history is lost
  and any in-flight Langfuse experiment may be left half-populated.
- v1 does not support cancellation; runs always finish or fail naturally.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from langfuse import Langfuse
from openai import AsyncOpenAI

from judge import Judge
from rag import RAGPipeline

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    PENDING = "pending"
    BUILDING = "building"
    JUDGING = "judging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunState:
    run_id: str
    eval_model: str
    judge_model: str
    corpus_datasets: list[str]
    question_datasets: list[str]
    k: int
    embedding_model: str
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    judge_prompts: list[str]
    max_concurrency: int
    started_at: datetime
    status: RunStatus = RunStatus.PENDING
    session_id: str | None = None
    finished_at: datetime | None = None
    error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class EvalRunner:
    def __init__(self, openai: AsyncOpenAI, langfuse: Langfuse) -> None:
        self._openai = openai
        self._langfuse = langfuse
        self._runs: dict[str, RunState] = {}

    def start(
        self,
        *,
        eval_model: str,
        judge_model: str,
        corpus: Sequence[str],
        questions: Sequence[str],
        k: int,
        embedding_model: str,
        rerank_model: str,
        chunk_size: int,
        chunk_overlap: int,
        judge_prompts: Sequence[str],
        max_concurrency: int,
    ) -> RunState:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        state = RunState(
            run_id=run_id,
            eval_model=eval_model,
            judge_model=judge_model,
            corpus_datasets=list(corpus),
            question_datasets=list(questions),
            k=k,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            judge_prompts=list(judge_prompts),
            max_concurrency=max_concurrency,
            started_at=datetime.now(UTC),
        )
        self._runs[run_id] = state
        state._task = asyncio.create_task(self._execute(state), name=run_id)
        return state

    def list(self) -> list[RunState]:
        return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    async def _execute(self, state: RunState) -> None:
        try:
            logger.info(
                "run %s starting | eval=%s judge=%s k=%d corpus=%s questions=%s",
                state.run_id,
                state.eval_model,
                state.judge_model,
                state.k,
                state.corpus_datasets,
                state.question_datasets,
            )
            state.status = RunStatus.BUILDING
            pipeline = RAGPipeline(
                self._openai,
                self._langfuse,
                embedding_model=state.embedding_model,
                rerank_model=state.rerank_model,
            )
            await pipeline.build(
                state.corpus_datasets,
                chunk_size=state.chunk_size,
                chunk_overlap=state.chunk_overlap,
                max_concurrency=state.max_concurrency,
            )

            judge = Judge(
                pipeline=pipeline,
                openai=self._openai,
                langfuse=self._langfuse,
                judge_model=state.judge_model,
                eval_model=state.eval_model,
                judge_prompts=state.judge_prompts,
                k=state.k,
            )
            state.session_id = judge.session_id
            state.status = RunStatus.JUDGING
            await judge.run(
                state.question_datasets, state.corpus_datasets, state.max_concurrency
            )
            state.status = RunStatus.COMPLETED
            logger.info("run %s completed", state.run_id)
        except Exception as exc:
            state.error = repr(exc)
            state.status = RunStatus.FAILED
            logger.exception("run %s failed", state.run_id)
        finally:
            state.finished_at = datetime.now(UTC)
