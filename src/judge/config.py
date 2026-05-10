from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeConfig(BaseSettings):
    """Tunable defaults for the LLM judge. Override via JUDGE_* env vars or `.env`."""

    prompt_prefix: str = "judge-"
    extract_prompt_name: str = "extract-score-from-judgement"
    max_extract_retries: int = Field(default=10, ge=1)
    failed_score: float = -1.0

    model_config = SettingsConfigDict(
        env_prefix="JUDGE_", env_file=".env", extra="ignore"
    )


@cache
def get_judge_config() -> JudgeConfig:
    return JudgeConfig()
