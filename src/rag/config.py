from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """Tunable defaults for the RAG pipeline. Override via RAG_* env vars or `.env`."""

    embedding_model: str = "bge-m3"
    rerank_model: str = "BGE-Reranker-V2-M3"
    embedding_batch_size: int = Field(default=64, gt=0)
    max_concurrency: int = Field(default=64, gt=0)
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    generator_system_prompt_name: str = "rag-generator-system"
    generator_user_prompt_name: str = "rag-generator-user"

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore"
    )


@cache
def get_rag_config() -> RAGConfig:
    return RAGConfig()
