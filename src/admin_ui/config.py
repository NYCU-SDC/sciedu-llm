from functools import cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdminUIConfig(BaseSettings):
    """Tunable defaults for the admin UI. Override via ADMIN_UI_* env vars or `.env`."""

    # 7861 avoids clashing with the eval UI (7860) when both run locally.
    port: int = Field(default=7861, ge=1, le=65535)
    # Base URL of the running FastAPI server (matches `poe dev`'s --port 8080).
    # No trailing `/admin` — the client appends the admin paths.
    api_base_url: str = "http://localhost:8080"

    model_config = SettingsConfigDict(
        env_prefix="ADMIN_UI_", env_file=".env", extra="ignore"
    )


@cache
def get_admin_ui_config() -> AdminUIConfig:
    return AdminUIConfig()
