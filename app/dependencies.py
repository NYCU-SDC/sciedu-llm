from functools import cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = Field(default=...)
    openai_default_model: str = "gpt-oss-120b"

    # Load env variables from .env for development, CI/CD deployments should rely on automated injection
    # Note that env variables always take precedence over values in .env
    model_config = SettingsConfigDict(env_file=".env")


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
