"""Start and track RAG evaluation runs.

`POST /admin/evals/runs` schedules the run on the app's event loop and answers
202 immediately — the run outlives the request. Run state lives in the process
(see :class:`judge.EvalRunner`), so `GET /runs` is lost on restart; `GET
/history` reads the durable record back out of Langfuse instead.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from app import listings
from app.dependencies import (
    eval_runner_dependency,
    langfuse_dependency,
    settings_dependency,
)
from app.schema.admin.evals import (
    EVAL_RUN_RESPONSES,
    CHUNK_OVERLAP_DETAIL,
    EvalHistoryEntry,
    EvalRunCreate,
    EvalRunResponse,
)
from app.schema.admin.meta import UPSTREAM_RESPONSES
from judge import RunNotCancellableError
from rag.config import get_rag_config

router = APIRouter(prefix="/evals", tags=["Admin"])

logger = logging.getLogger(__name__)


@router.post(
    "/runs",
    status_code=202,
    response_model=EvalRunResponse,
    summary="Start a background evaluation run",
    description=(
        "Schedules the run and returns immediately with its `run_id`. Unset "
        "model / chunking fields fall back to the server's `RAG_*` configuration. "
        "Poll `GET /admin/evals/runs/{run_id}` for progress."
    ),
)
async def create_eval_run(payload: EvalRunCreate, runner: eval_runner_dependency):
    rag_config = get_rag_config()
    embedding_model = payload.embedding_model or rag_config.embedding_model
    rerank_model = payload.rerank_model or rag_config.rerank_model
    chunk_size = (
        payload.chunk_size if payload.chunk_size is not None else rag_config.chunk_size
    )
    chunk_overlap = (
        payload.chunk_overlap
        if payload.chunk_overlap is not None
        else rag_config.chunk_overlap
    )
    # The schema can only compare the two when the client pinned both; once the
    # server's defaults are mixed in, the pair has to be re-checked.
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{CHUNK_OVERLAP_DETAIL} "
                f"(resolved chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
            ),
        )

    state = runner.start(
        eval_model=payload.eval_model,
        judge_model=payload.judge_model,
        corpus=payload.corpus_datasets,
        questions=payload.question_datasets,
        k=payload.k,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        judge_prompts=payload.judge_prompts,
        max_concurrency=rag_config.max_concurrency,
    )
    logger.info("Started eval run %s", state.run_id)
    return EvalRunResponse.from_state(state)


@router.get(
    "/runs",
    response_model=list[EvalRunResponse],
    summary="List the evaluation runs this process knows about",
    description="Newest first. In-memory only — cleared when the server restarts.",
)
async def list_eval_runs(runner: eval_runner_dependency):
    return [EvalRunResponse.from_state(state) for state in runner.list()]


@router.get(
    "/runs/{run_id}",
    response_model=EvalRunResponse,
    summary="Get a single evaluation run",
    responses=EVAL_RUN_RESPONSES,
)
async def get_eval_run(run_id: str, runner: eval_runner_dependency):
    state = runner.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return EvalRunResponse.from_state(state)


@router.post(
    "/runs/{run_id}/cancel",
    response_model=EvalRunResponse,
    summary="Cancel an in-flight evaluation run",
    description=(
        "Cancels the underlying task. The returned status may still read as "
        "running — it settles on `cancelled` once the task unwinds. Any Langfuse "
        "experiment the run had started is left partially populated."
    ),
    responses={
        **EVAL_RUN_RESPONSES,
        409: {
            "description": "Conflict - the run already reached a terminal status",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Run run-x already finished with status 'completed'."
                    }
                }
            },
        },
    },
)
async def cancel_eval_run(run_id: str, runner: eval_runner_dependency):
    try:
        state = runner.cancel(run_id)
    except RunNotCancellableError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return EvalRunResponse.from_state(state)


@router.get(
    "/history",
    response_model=list[EvalHistoryEntry],
    summary="List past experiment runs recorded in Langfuse",
    description=(
        "Reads the judge runs Langfuse recorded against a question dataset, "
        "newest first. Unlike `GET /runs` this survives restarts — it is "
        "Langfuse's own record of every judge run that has executed against the "
        "dataset. Langfuse v4 keeps these as *experiments* and requires a start "
        "time, so this is a window: the last `EVALS_HISTORY_LOOKBACK_DAYS` days "
        "(90 by default), not all of history."
    ),
    responses=UPSTREAM_RESPONSES,
)
async def list_eval_history(
    langfuse: langfuse_dependency,
    settings: settings_dependency,
    question_dataset: str = Query(
        description="Canonical Langfuse question dataset name, folder prefix included."
    ),
):
    since = datetime.now(UTC) - timedelta(days=settings.evals_history_lookback_days)
    try:
        runs = await listings.list_experiment_runs(
            langfuse, question_dataset, since=since
        )
    except Exception as e:
        logger.exception("Failed to list Langfuse experiment runs")
        raise HTTPException(
            status_code=502, detail=f"Failed to list Langfuse experiment runs: {e}"
        ) from e

    return [
        EvalHistoryEntry(
            dataset_name=run.dataset_name,
            run_name=run.name,
            created_at=run.created_at,
            description=run.description,
        )
        for run in runs
    ]
