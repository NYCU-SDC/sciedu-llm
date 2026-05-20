from functools import cache
from typing import Annotated

from fastapi import Depends
from langfuse import Langfuse
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from observability import init_langfuse_client


class Settings(BaseSettings):
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = Field(default=...)
    openai_default_model: str = "gpt-oss-120b"

    chat_title_prompt_name: str = "app/chat-title-generator"
    chat_title_max_attempts: int = 3

    # Load env variables from .env for development, CI/CD deployments should rely on automated injection
    # Note that env variables always take precedence over values in .env.
    # `extra="ignore"` because .env is shared with other modules (langfuse/rag/judge/eval_ui).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


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
