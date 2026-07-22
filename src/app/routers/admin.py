import logging

from fastapi import APIRouter, HTTPException

from app.dependencies import rag_pipeline_dependency
from app.schema.admin import (
    ADMIN_RAG_RESPONSES,
    RAGConfigResponse,
    RAGConfigUpdate,
    RAGConfigUpdateResponse,
)
from rag import RAGPipeline
from rag.config import RAGConfig

router = APIRouter(prefix="/admin", tags=["Admin"])

logger = logging.getLogger(__name__)


def _require_pipeline(rag_pipeline: RAGPipeline | None) -> RAGPipeline:
    """Return the live pipeline or raise 503 when RAG is disabled."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it.",
        )
    return rag_pipeline


def _snapshot_response(pipeline: RAGPipeline) -> RAGConfigResponse:
    return RAGConfigResponse(
        **pipeline.config_snapshot(),
        is_built=pipeline.is_built,
        corpus_datasets=pipeline.corpus_dataset_names,
    )


async def _rebuild(
    pipeline: RAGPipeline, corpus_datasets: list[str] | None = None
) -> None:
    """Rebuild the indexes, surfacing failures as a 502.

    When ``corpus_datasets`` is given, re-index from those datasets (updating the
    corpus the pipeline is built from); otherwise rebuild from the current corpus.
    """
    try:
        if corpus_datasets is not None:
            await pipeline.build(corpus_datasets)
        else:
            await pipeline.rebuild()
    except Exception as e:
        logger.exception("RAG rebuild failed")
        raise HTTPException(
            status_code=502, detail=f"Error during RAG rebuild: {str(e)}"
        ) from e


@router.get(
    "/rag/config",
    response_model=RAGConfigResponse,
    summary="Get the current RAG pipeline configuration",
    responses=ADMIN_RAG_RESPONSES,
)
async def get_rag_config(rag_pipeline: rag_pipeline_dependency):
    pipeline = _require_pipeline(rag_pipeline)
    return _snapshot_response(pipeline)


@router.patch(
    "/rag/config",
    response_model=RAGConfigUpdateResponse,
    summary="Override RAG pipeline configuration",
    description=(
        "Partially override the RAG config. Retrieval knobs apply to the next "
        "query immediately. The indexes are rebuilt after applying the changes "
        "by default (so build-time fields take effect); pass `rebuild=false` to "
        "skip the rebuild. Pass `corpus_datasets` to re-index from a different set "
        "of Langfuse corpus datasets (always rebuilds)."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def update_rag_config(
    update: RAGConfigUpdate, rag_pipeline: rag_pipeline_dependency
):
    pipeline = _require_pipeline(rag_pipeline)
    overrides = update.model_dump(
        exclude_unset=True, exclude_none=True, exclude={"rebuild", "corpus_datasets"}
    )

    pipeline.apply_overrides(overrides)

    rebuilt = update.rebuild
    if update.corpus_datasets is not None:
        if not update.corpus_datasets:
            raise HTTPException(
                status_code=400,
                detail="corpus_datasets must contain at least one dataset name.",
            )
        # A corpus change only takes effect once re-indexed, so always rebuild.
        await _rebuild(pipeline, corpus_datasets=update.corpus_datasets)
        rebuilt = True
    elif update.rebuild:
        await _rebuild(pipeline)

    return RAGConfigUpdateResponse(config=_snapshot_response(pipeline), rebuilt=rebuilt)


@router.post(
    "/rag/rebuild",
    response_model=RAGConfigResponse,
    summary="Force a rebuild of the RAG indexes",
    description=(
        "Rebuild the BM25 + dense indexes from the configured corpus datasets "
        "using the current config — e.g. to re-index after the corpus changed in "
        "Langfuse. Config changes go through PATCH /admin/rag/config."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def rebuild_rag(rag_pipeline: rag_pipeline_dependency):
    pipeline = _require_pipeline(rag_pipeline)
    await _rebuild(pipeline)
    return _snapshot_response(pipeline)


@router.post(
    "/rag/reset",
    response_model=RAGConfigUpdateResponse,
    summary="Reset RAG configuration to environment defaults",
    description=(
        "Discard all runtime overrides, restoring the values derived from the "
        "RAG_* environment variables, and rebuild the indexes."
    ),
    responses=ADMIN_RAG_RESPONSES,
)
async def reset_rag_config(rag_pipeline: rag_pipeline_dependency):
    pipeline = _require_pipeline(rag_pipeline)

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
    await _rebuild(pipeline)

    return RAGConfigUpdateResponse(config=_snapshot_response(pipeline), rebuilt=True)
