from functools import cache
from typing import Annotated

from fastapi import Depends, Request
from langfuse import Langfuse
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from observability import init_langfuse_client
from rag import RAGPipeline


class Settings(BaseSettings):
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = Field(default=...)
    openai_default_model: str = "gpt-oss-120b"

    chat_title_prompt_name: str = "app/chat-title-generator"
    chat_title_max_attempts: int = 3

    # Comma-separated Langfuse corpus dataset names to index for RAG-enabled chat.
    # Read from RAG_CORPUS_DATASETS. Leave empty to disable RAG (the /chat
    # `enable_rag` flag then returns 503 until at least one dataset is configured).
    rag_corpus_datasets: str = ""

    # Load env variables from .env for development, CI/CD deployments should rely on automated injection
    # Note that env variables always take precedence over values in .env.
    # `extra="ignore"` because .env is shared with other modules (langfuse/rag/judge/eval_ui).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def rag_corpus_dataset_names(self) -> list[str]:
        return [name.strip() for name in self.rag_corpus_datasets.split(",") if name.strip()]


@cache
def get_settings():
    settings = Settings()
    return settings


settings_dependency = Annotated[Settings, Depends(get_settings)]


@cache
def get_openai_client():
    settings = get_settings()
    client = AsyncOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    return client


openai_dependency = Annotated[AsyncOpenAI, Depends(get_openai_client)]


@cache
def get_langfuse_client() -> Langfuse:
    return init_langfuse_client()


langfuse_dependency = Annotated[Langfuse, Depends(get_langfuse_client)]


async def build_rag_pipeline() -> RAGPipeline | None:
    """Build the RAG pipeline from the configured corpus datasets at startup.

    Returns ``None`` when no corpus datasets are configured, leaving RAG disabled.
    Called once from the app lifespan; the built pipeline is stashed on
    ``app.state`` and served via ``get_rag_pipeline``.
    """
    settings = get_settings()
    names = settings.rag_corpus_dataset_names
    if not names:
        return None
    pipeline = RAGPipeline(get_openai_client(), get_langfuse_client())
    await pipeline.build(names)
    return pipeline


def get_rag_pipeline(request: Request) -> RAGPipeline | None:
    return getattr(request.app.state, "rag_pipeline", None)


rag_pipeline_dependency = Annotated[RAGPipeline | None, Depends(get_rag_pipeline)]
