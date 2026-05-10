from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalUIConfig(BaseSettings):
    """Tunable defaults for the eval UI. Override via EVAL_UI_* env vars or `.env`."""

    port: int = Field(default=7860, ge=1, le=65535)
    corpus_dataset_prefix: str = "corpus-"
    questions_dataset_prefix: str = "questions-"

    model_config = SettingsConfigDict(
        env_prefix="EVAL_UI_", env_file=".env", extra="ignore"
    )


@cache
def get_eval_ui_config() -> EvalUIConfig:
    return EvalUIConfig()
