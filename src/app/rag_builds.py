"""Background, cancellable index builds for the live RAG pipeline.

Re-indexing embeds the whole corpus, which for a real one is minutes of upstream
calls. Doing that *inside* the request that asked for it makes a rebuild
something an operator can start and then only wait out: the HTTP call sits open
for the duration, the console can do nothing but hold the tab, and there is no
way to say "stop, I picked the wrong corpus".

So `RagBuildManager` runs at most one build at a time as a task on the app's
event loop — the same fire-and-forget shape `judge.EvalRunner` uses for eval runs
— reports what that build is doing, and can be told to cancel it. The admin
routes then answer immediately and the console polls `GET /admin/rag/config`.

Cancelling is safe because `RAGPipeline.build` only swaps its indexes in on the
last line: a cancelled or failed build leaves the previous indexes serving
queries, exactly as they were. Build settings stay fixed for this cache entry.

Strong reference invariant: `self._task` is what keeps the running build alive
for the GC — do not "clean up" the apparently-unused field.

State is in-memory only: a restart forgets the last build's outcome. What this
carries is job-level — running, cancelled, failed, how long — while the work-level
detail (dataset fetches and embedding batches) goes to tqdm alongside the service
log from `rag.pipeline`, which is where a long build is watched.
"""

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

from rag import RAGPipeline

logger = logging.getLogger(__name__)


class BuildStatus(StrEnum):
    #: No build has been scheduled for this cache entry.
    IDLE = "idle"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BuildInProgressError(RuntimeError):
    """Raised by :meth:`RagBuildManager.start` while a build is already running."""


class NoBuildInProgressError(RuntimeError):
    """Raised by :meth:`RagBuildManager.cancel` when nothing is running."""


@dataclass(frozen=True)
class BuildState:
    """A snapshot of the last (or current) build. Immutable on purpose."""

    status: BuildStatus = BuildStatus.IDLE
    corpus_datasets: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    #: A cancel has been asked for but the build has not unwound yet. The status
    #: stays `building` until it does, because until then it really is.
    cancel_requested: bool = False

    @property
    def is_building(self) -> bool:
        return self.status is BuildStatus.BUILDING

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock seconds the build took; for a live one, up to *now*."""
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()


class RagBuildManager:
    """Owns the one in-flight index build for a pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline
        self._state = BuildState()
        self._task: asyncio.Task[None] | None = None

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    @property
    def state(self) -> BuildState:
        return self._state

    def start(self, corpus_datasets: list[str] | None = None) -> BuildState:
        """Schedule a rebuild and return immediately.

        ``corpus_datasets`` re-indexes from that set (and becomes the corpus the
        pipeline is built from); omitting it rebuilds from the current corpus.
        Raises :class:`BuildInProgressError` when one is already running — the
        caller answers 409 — and ``ValueError`` when there is no corpus to build
        from.
        """
        if self._task is not None and not self._task.done():
            raise BuildInProgressError(
                "An index build is already running. Cancel it before starting another."
            )

        names = (
            list(corpus_datasets)
            if corpus_datasets is not None
            else self._pipeline.corpus_dataset_names
        )
        if not names:
            raise ValueError("No corpus datasets configured to build from.")

        snapshot = self._pipeline.config_snapshot()
        self._state = BuildState(
            status=BuildStatus.BUILDING,
            corpus_datasets=names,
            started_at=datetime.now(UTC),
        )
        self._task = asyncio.create_task(self._run(names), name="rag-build")
        self._task.add_done_callback(self._on_done)
        logger.info(
            "RAG index build requested | corpus=%s chunk_size=%s chunk_overlap=%s "
            "embedding_model=%s",
            ", ".join(names),
            snapshot.get("chunk_size"),
            snapshot.get("chunk_overlap"),
            snapshot.get("embedding_model"),
        )
        return self._state

    def cancel(self) -> BuildState:
        """Ask the running build to stop.

        Raises :class:`NoBuildInProgressError` when nothing is running (the
        caller answers 409). The returned state still reads as ``building`` with
        ``cancel_requested`` set: the task has only been asked, and it stops at
        its next await — which is never long, but is not now.
        """
        if self._task is None or self._task.done():
            raise NoBuildInProgressError("No index build is currently running.")

        self._state = replace(self._state, cancel_requested=True)
        self._task.cancel()
        logger.info("RAG index build cancellation requested")
        return self._state

    def shutdown(self) -> None:
        """Cancel a running build. Called from the app lifespan teardown.

        Call wait() afterwards to drain the cancelled task.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            logger.info("RAG index build cancelled by shutdown")

    def _on_done(self, task: asyncio.Task[None]) -> None:
        # Cancellation can happen before _run's first instruction executes.
        if task is self._task and task.cancelled() and self._state.is_building:
            self._settle(BuildStatus.CANCELLED)

    async def wait(self) -> None:
        """Drain the current job, including cancellation during shutdown."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, names: list[str]) -> None:
        try:
            await self._pipeline.build(names)
        except asyncio.CancelledError:
            self._settle(BuildStatus.CANCELLED)
            logger.info("RAG index build cancelled — the previous indexes stay in use")
            raise
        except Exception as exc:
            logger.exception("RAG index build failed")
            self._settle(BuildStatus.FAILED, error=str(exc))
        else:
            self._settle(BuildStatus.COMPLETED)
            logger.info(
                "RAG index build finished in %.1fs",
                self._state.duration_seconds or 0.0,
            )

    def _settle(self, status: BuildStatus, *, error: str | None = None) -> None:
        self._state = replace(
            self._state,
            status=status,
            finished_at=datetime.now(UTC),
            error=error,
            cancel_requested=False,
        )
