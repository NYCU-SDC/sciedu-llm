from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RAGConfigValues(BaseModel):
    """The tunable RAG pipeline config values (env-derived defaults + overrides)."""

    embedding_model: str
    rerank_model: str
    embedding_batch_size: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    generator_system_prompt_name: str
    generator_user_prompt_name: str
    bm25_top_n: int = Field(gt=0)
    dense_top_n: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    rerank_pool_size: int = Field(gt=0)
    final_k: int = Field(gt=0)


class RAGBuildState(BaseModel):
    """What the background index build is doing (or last did).

    Rebuilds do not happen inside the request that asks for one — they are
    scheduled and this is how a client follows them: poll `GET /admin/rag/config`
    while `status` is `building`. Deliberately job-level: no percentage, because
    what a client can act on is "still going / finished / failed", and
    `duration_seconds` says how long it has been going. Batch-by-batch progress
    is logged by the service instead (`rag.pipeline`, at INFO).
    """

    status: Literal["idle", "building", "completed", "failed", "cancelled"] = Field(
        description=(
            "`idle` means no build has been started through the admin API this "
            "process — the indexes may still be built, from startup."
        )
    )
    corpus_datasets: list[str] = Field(
        description="Corpus datasets this build is (or was) indexing."
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(
        default=None,
        description="Elapsed seconds; counts up while a build is running.",
    )
    error: str | None = Field(
        default=None, description="Why the build failed, when `status` is `failed`."
    )
    cancel_requested: bool = Field(
        default=False,
        description=(
            "A cancel has been accepted but the build has not unwound yet, so "
            "`status` is still `building`."
        ),
    )


class RAGConfigResponse(RAGConfigValues):
    """Current effective config plus pipeline status."""

    is_built: bool = Field(
        description="Whether the BM25 + dense indexes are currently built."
    )
    corpus_datasets: list[str] = Field(
        description="Langfuse corpus dataset names the current indexes were built from."
    )
    build: RAGBuildState = Field(description="The state of the background index build.")


class RAGConfigUpdate(BaseModel):
    """Partial override of the RAG config. Only the provided fields are changed.

    The indexes are rebuilt after applying the changes by default; set
    ``rebuild=false`` to apply without rebuilding (build-time changes then take
    effect only on the next rebuild).

    Build-time changes are applied to the pipeline before the rebuild that makes
    them real, and are rolled back if that rebuild is cancelled or fails — so a
    later ``GET /config`` describes the indexes that are actually serving.
    """

    rebuild: bool = Field(
        default=True,
        description=(
            "Whether to rebuild the indexes after applying the changes. Defaults "
            "to true. Set false to skip the rebuild. The rebuild runs in the "
            "background — this request returns as soon as it is scheduled."
        ),
    )

    corpus_datasets: Optional[list[str]] = Field(
        default=None,
        description=(
            "Override which Langfuse corpus datasets the indexes are built from. "
            "When provided (non-empty), the indexes are always rebuilt from the new "
            "corpus regardless of the `rebuild` flag, so the change takes effect."
        ),
    )

    embedding_model: Optional[str] = None
    rerank_model: Optional[str] = None
    embedding_batch_size: Optional[int] = Field(default=None, gt=0)
    max_concurrency: Optional[int] = Field(default=None, gt=0)
    chunk_size: Optional[int] = Field(default=None, gt=0)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)
    generator_system_prompt_name: Optional[str] = None
    generator_user_prompt_name: Optional[str] = None
    bm25_top_n: Optional[int] = Field(default=None, gt=0)
    dense_top_n: Optional[int] = Field(default=None, gt=0)
    rrf_k: Optional[int] = Field(default=None, gt=0)
    rerank_pool_size: Optional[int] = Field(default=None, gt=0)
    final_k: Optional[int] = Field(default=None, gt=0)

    model_config = {
        "json_schema_extra": {"example": {"final_k": 8, "rerank_pool_size": 40}}
    }


class RAGConfigUpdateResponse(BaseModel):
    """Result of an override — the new effective config and whether a build started."""

    config: RAGConfigResponse
    build_started: bool = Field(
        description=(
            "True when the change required an index rebuild and one was "
            "scheduled. Follow it via `config.build`."
        )
    )


ADMIN_RAG_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {
        "description": (
            "Conflict - a build is already running, or a cancel was asked for "
            "with nothing running"
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": "An index build is already running. Cancel it before starting another."
                }
            }
        },
    },
    503: {
        "description": "Service Unavailable - RAG is not enabled on this server",
        "content": {
            "application/json": {
                "example": {
                    "detail": "RAG is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it."
                }
            }
        },
    },
}
