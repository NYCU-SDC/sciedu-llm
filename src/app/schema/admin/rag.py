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
    """Status of the cached build shared by matching presets."""

    status: Literal["idle", "building", "completed", "failed", "cancelled"] = Field(
        description="idle means no build has been scheduled for this cache entry."
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

    preset: str = Field(
        description="Preset whose corpus and index status are reported."
    )
    is_built: bool = Field(
        description="Whether the BM25 + dense indexes are currently built."
    )
    corpus_datasets: list[str] = Field(
        description="Corpus dataset names configured in the selected preset."
    )
    build: RAGBuildState = Field(description="The state of the background index build.")


class RAGConfigUpdate(BaseModel):
    """Partial override of server-wide RAG settings.

    Changed build settings select new cache entries; failed or cancelled builds
    never change settings or indexes stored under another cache key.
    """

    rebuild: bool = Field(
        default=True,
        description=(
            "Whether to rebuild the indexes after applying the changes. Defaults "
            "to true. Set false to skip the rebuild. The rebuild runs in the "
            "background — this request returns as soon as it is scheduled."
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
        "extra": "forbid",
        "json_schema_extra": {"example": {"final_k": 8, "rerank_pool_size": 40}},
    }


class RAGConfigUpdateResponse(BaseModel):
    """Result of an override — the new effective config and whether a build started."""

    config: RAGConfigResponse
    build_started: bool = Field(
        description=(
            "True when at least one missing preset index build was scheduled. "
            "config.build describes only the selected preset."
        )
    )


ADMIN_RAG_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Enable RAG and select a corpus before rebuilding."},
    404: {"description": "The selected preset does not exist."},
    409: {
        "description": "The shared entry is already building, or no build can be cancelled."
    },
}
