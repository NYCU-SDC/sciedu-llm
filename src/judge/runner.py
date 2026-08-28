"""Process-global registry for background evaluation runs.

`EvalRunner.start(...)` constructs a `RunState`, schedules `_execute(...)` on the
running event loop via `asyncio.create_task`, and returns immediately. Callers
(the `/admin/evals` API) do NOT await the task — the HTTP response goes out while
the run keeps going on the app's uvicorn loop until it finishes, fails, or is
cancelled.

Strong reference invariant: `state._task` keeps the task alive for the GC; do
not "clean up" the apparently-unused field.

Caveats:
- State is in-memory only. If the app process dies, run history is lost and any
  in-flight Langfuse experiment may be left half-populated.
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

from judge.judge import Judge
from rag import RAGPipeline

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    PENDING = "pending"
    BUILDING = "building"
    JUDGING = "judging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses a run can never leave. Cancelling one of these is a conflict.
TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class RunNotCancellableError(RuntimeError):
    """Raised by :meth:`EvalRunner.cancel` for a run that already finished."""

    def __init__(self, run_id: str, status: RunStatus) -> None:
        super().__init__(f"Run {run_id} already finished with status '{status}'.")
        self.run_id = run_id
        self.status = status


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

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def duration_seconds(self) -> float:
        """Wall-clock seconds elapsed; for a live run, up to *now*."""
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()


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

    def cancel(self, run_id: str) -> RunState | None:
        """Request cancellation of an in-flight run.

        Returns ``None`` for an unknown ``run_id`` (the caller answers 404) and
        raises :class:`RunNotCancellableError` when the run already reached a
        terminal status (the caller answers 409). Otherwise the underlying task
        is cancelled and the state is returned — the status flips to
        ``cancelled`` once :meth:`_execute` unwinds, so the returned state may
        still read as running for a moment.
        """
        state = self._runs.get(run_id)
        if state is None:
            return None
        if state.is_terminal:
            raise RunNotCancellableError(run_id, state.status)
        if state._task is not None:
            state._task.cancel()
        logger.info("run %s cancellation requested", run_id)
        return state

    def shutdown(self) -> None:
        """Cancel every non-terminal run. Called from the app lifespan teardown.

        Fire-and-forget: this is synchronous, so the tasks are only *asked* to
        stop. The loop is about to go away with the process, so there is nothing
        useful to await.
        """
        for state in self._runs.values():
            if state.is_terminal or state._task is None or state._task.done():
                continue
            state._task.cancel()
            logger.info("run %s cancelled by shutdown", state.run_id)

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
        # Must precede the generic handler: `CancelledError` is a BaseException,
        # but an `except Exception` above it would still be wrong to reach for.
        except asyncio.CancelledError:
            state.status = RunStatus.CANCELLED
            logger.info("run %s cancelled", state.run_id)
            # Re-raise so the task is marked cancelled rather than completed —
            # swallowing it would lie to anyone inspecting the task.
            raise
        except Exception as exc:
            state.error = repr(exc)
            state.status = RunStatus.FAILED
            logger.exception("run %s failed", state.run_id)
        finally:
            state.finished_at = datetime.now(UTC)
