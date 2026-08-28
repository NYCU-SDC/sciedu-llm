"""Read and retune the live RAG pipeline.

Re-indexing is slow enough that it is not done inside the request that asks for
it: every route here that needs a rebuild *schedules* one on
`app.rag_builds.RagBuildManager` and answers straight away, so an admin UI can
show the build running, keep working, and stop it. Poll `GET /rag/config` and
watch `build.status` to follow one.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.dependencies import rag_build_manager_dependency
from app.rag_builds import (
    BuildInProgressError,
    BuildState,
    NoBuildInProgressError,
    RagBuildManager,
)
from app.schema.admin.rag import (
    ADMIN_RAG_RESPONSES,
    RAGBuildState,
    RAGConfigResponse,
    RAGConfigUpdate,
    RAGConfigUpdateResponse,
)
from rag.config import RAGConfig

# Mounted under the `/admin` aggregator in `app.routers.admin`, so the paths
# below resolve to `/admin/rag/...`.
router = APIRouter(prefix="/rag", tags=["Admin"])

logger = logging.getLogger(__name__)


def _require_manager(manager: RagBuildManager | None) -> RagBuildManager:
    """Return the build manager, or raise 503 when RAG is disabled.

    The manager exists exactly when a pipeline does, so this is the one gate
    every route here needs — each of them reaches the pipeline through it.
    """
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it.",
        )
    return manager


def _build_state(state: BuildState) -> RAGBuildState:
    return RAGBuildState(
        status=state.status.value,
        corpus_datasets=state.corpus_datasets,
        started_at=state.started_at,
        finished_at=state.finished_at,
        duration_seconds=state.duration_seconds,
        error=state.error,
        cancel_requested=state.cancel_requested,
    )


def _snapshot_response(manager: RagBuildManager) -> RAGConfigResponse:
    pipeline = manager.pipeline
    return RAGConfigResponse(
        **pipeline.config_snapshot(),
        is_built=pipeline.is_built,
        corpus_datasets=pipeline.corpus_dataset_names,
        build=_build_state(manager.state),
    )


def _start_build(
    manager: RagBuildManager, corpus_datasets: list[str] | None = None
) -> None:
    """Schedule a rebuild, translating the manager's refusals into HTTP.

    A build already running is a 409 rather than a queued second build: two
    concurrent re-indexes of the same pipeline would race to install their
    indexes, and the operator asking for one almost certainly wants the newer
    settings, not both.
    """
    try:
        manager.start(corpus_datasets)
    except BuildInProgressError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/config",
    response_model=RAGConfigResponse,
    summary="Get the current RAG pipeline configuration",
    description=(
        "Includes `build`, the state of the background index build. Poll this "
        "while `build.status` is `building` to follow a rebuild."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def get_rag_config(rag_builds: rag_build_manager_dependency):
    return _snapshot_response(_require_manager(rag_builds))


@router.patch(
    "/config",
    response_model=RAGConfigUpdateResponse,
    summary="Override RAG pipeline configuration",
    description=(
        "Partially override the RAG config. Retrieval knobs apply to the next "
        "query immediately. A rebuild is *scheduled* after applying the changes "
        "by default (so build-time fields take effect) and runs in the "
        "background; pass `rebuild=false` to skip it. Pass `corpus_datasets` to "
        "re-index from a different set of Langfuse corpus datasets (always "
        "rebuilds). Answers 409 if a build is already running."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def update_rag_config(
    update: RAGConfigUpdate, rag_builds: rag_build_manager_dependency
):
    manager = _require_manager(rag_builds)
    overrides = update.model_dump(
        exclude_unset=True, exclude_none=True, exclude={"rebuild", "corpus_datasets"}
    )

    if update.corpus_datasets is not None and not update.corpus_datasets:
        raise HTTPException(
            status_code=400,
            detail="corpus_datasets must contain at least one dataset name.",
        )

    manager.pipeline.apply_overrides(overrides)

    build_started = False
    if update.corpus_datasets is not None:
        # A corpus change only takes effect once re-indexed, so always rebuild.
        _start_build(manager, update.corpus_datasets)
        build_started = True
    elif update.rebuild:
        _start_build(manager)
        build_started = True

    return RAGConfigUpdateResponse(
        config=_snapshot_response(manager), build_started=build_started
    )


@router.post(
    "/rebuild",
    response_model=RAGConfigResponse,
    status_code=202,
    summary="Start a rebuild of the RAG indexes",
    description=(
        "Schedule a rebuild of the BM25 + dense indexes from the configured "
        "corpus datasets using the current config — e.g. to re-index after the "
        "corpus changed in Langfuse. Returns as soon as the build is scheduled; "
        "follow it via `build` on this response or `GET /admin/rag/config`. "
        "Config changes go through PATCH /admin/rag/config."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def rebuild_rag(rag_builds: rag_build_manager_dependency):
    manager = _require_manager(rag_builds)
    _start_build(manager)
    return _snapshot_response(manager)


@router.post(
    "/rebuild/cancel",
    response_model=RAGConfigResponse,
    summary="Stop the running index build",
    description=(
        "Cancel the in-flight rebuild. The indexes that were serving queries "
        "before it started keep serving them — a build only installs its result "
        "once it has finished — and the build-time settings it was going to bake "
        "(chunking, embedding model, batch size) are reverted to the ones the "
        "live indexes actually have, so this endpoint keeps describing what is "
        "answering queries. Answers 409 when no build is running. The build stops "
        "at its next await, so the returned `build.status` may still read "
        "`building` with `build.cancel_requested` set; poll `GET /rag/config` for "
        "the settled state."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def cancel_rag_rebuild(rag_builds: rag_build_manager_dependency):
    manager = _require_manager(rag_builds)
    try:
        manager.cancel()
    except NoBuildInProgressError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _snapshot_response(manager)


@router.post(
    "/reset",
    response_model=RAGConfigUpdateResponse,
    summary="Reset RAG configuration to environment defaults",
    description=(
        "Discard all runtime overrides, restoring the values derived from the "
        "RAG_* environment variables, and schedule a rebuild of the indexes."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def reset_rag_config(rag_builds: rag_build_manager_dependency):
    manager = _require_manager(rag_builds)
    pipeline = manager.pipeline

    # A fresh RAGConfig() re-reads the env defaults (bypassing the cached
    # get_rag_config singleton). Apply only the fields that differ from the
    # current effective values, then always rebuild.
    env_defaults = RAGConfig().model_dump()
    current = pipeline.config_snapshot()
    overrides = {
        key: value for key, value in env_defaults.items() if current.get(key) != value
    }

    if overrides:
        pipeline.apply_overrides(overrides)
    _start_build(manager)

    return RAGConfigUpdateResponse(
        config=_snapshot_response(manager), build_started=True
    )
