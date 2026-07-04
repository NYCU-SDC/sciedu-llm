from typing import Any, Optional

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


class RAGConfigResponse(RAGConfigValues):
    """Current effective config plus pipeline status."""

    is_built: bool = Field(
        description="Whether the BM25 + dense indexes are currently built."
    )
    corpus_datasets: list[str] = Field(
        description="Langfuse corpus dataset names the current indexes were built from."
    )


class RAGConfigUpdate(BaseModel):
    """Partial override of the RAG config. Only the provided fields are changed.

    The indexes are rebuilt after applying the changes by default; set
    ``rebuild=false`` to apply without rebuilding (build-time changes then take
    effect only on the next rebuild).
    """

    rebuild: bool = Field(
        default=True,
        description=(
            "Whether to rebuild the indexes after applying the changes. Defaults "
            "to true. Set false to skip the rebuild."
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
    """Result of an override — the new effective config and whether a rebuild ran."""

    config: RAGConfigResponse
    rebuilt: bool = Field(
        description="True when the change required and triggered an index rebuild."
    )


ADMIN_RAG_RESPONSES: dict[int | str, dict[str, Any]] = {
    502: {
        "description": "Bad Gateway - Index rebuild failed",
        "content": {
            "application/json": {
                "example": {"detail": "Error during RAG rebuild: <reason>"}
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
