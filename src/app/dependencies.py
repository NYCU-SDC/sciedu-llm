import logging
from functools import cache
from typing import Annotated

from fastapi import Depends, Request
from langfuse import Langfuse
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import listings
from app.presets import PresetRegistry
from judge import EvalRunner
from observability import init_langfuse_client
from rag import RAGPipeline

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = Field(default=...)
    openai_default_model: str = "gpt-oss-120b"

    # Comma-separated list of model ids the /chat endpoint is permitted to serve.
    # Read from ALLOWED_MODELS. Must be non-empty — the app refuses to start
    # otherwise (see `validate_allowed_models`). Requests asking for a model
    # outside this list are rejected with a 400.
    allowed_models: str = ""

    chat_title_prompt_name: str = "app/chat-title-generator"
    chat_title_max_attempts: int = 3

    # Agentic (/agents) config. Who may speak, what they may call and how many
    # steps they get is no longer configured here — it lives in a *preset* (see
    # `app.presets`), so a new behaviour is a config change rather than a deploy.
    #
    # How long a single tool may go without producing anything before it is
    # abandoned and reported to the model as a timeout.
    agents_tool_timeout_seconds: float = 60.0

    # Langfuse dataset holding the preset documents. Each item's `input` is one
    # preset; entries shadow the code-defined defaults of the same name. The
    # defaults are seeded into it at startup when missing, and a missing or
    # broken dataset is survivable — the code defaults stay in service.
    presets_dataset_name: str = "config/presets"
    # How long a loaded preset map is served before the dataset is re-read.
    presets_cache_ttl_seconds: float = 300.0
    # The preset /agents runs when the request omits `preset`.
    agents_default_preset: str = "default-agents"
    # The presets /chat is implemented on top of: `enable_rag` picks between them.
    chat_preset_name: str = "default-chat-plain"
    chat_rag_preset_name: str = "default-chat"

    # Comma-separated Langfuse corpus dataset names to index for RAG-enabled chat.
    # Read from RAG_CORPUS_DATASETS. Optional: when empty, every dataset under
    # the corpus folder is discovered from Langfuse at startup instead (see
    # `build_rag_pipeline`). RAG is disabled only when neither yields a dataset.
    rag_corpus_datasets: str = ""

    # Langfuse dataset folders the /admin listing endpoints filter on. A dataset
    # named "corpus/ver3/biology" is a corpus dataset; "questions/biology" is a
    # question dataset. Folders are a Langfuse naming convention, not an API
    # concept, so the prefixes are configuration rather than constants.
    corpus_dataset_folder: str = "corpus"
    questions_dataset_folder: str = "questions"

    # Load env variables from .env for development, CI/CD deployments should rely on automated injection
    # Note that env variables always take precedence over values in .env.
    # `extra="ignore"` because .env is shared with other modules (langfuse/rag/judge).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def rag_corpus_dataset_names(self) -> list[str]:
        return [
            name.strip() for name in self.rag_corpus_datasets.split(",") if name.strip()
        ]

    @property
    def allowed_model_names(self) -> list[str]:
        return [name.strip() for name in self.allowed_models.split(",") if name.strip()]


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


async def validate_allowed_models() -> list[str]:
    """Validate the configured ALLOWED_MODELS at startup.

    Ensures at least one model is configured (raising ``ValueError`` otherwise) and
    warns for any allowed model that the upstream OpenAI-compatible server does not
    advertise via its ``/models`` endpoint. A failed listing only logs — the models
    endpoint is best-effort and should not block startup. Returns the validated
    list of allowed model names. Called once from the app lifespan.
    """
    settings = get_settings()
    allowed = settings.allowed_model_names
    if not allowed:
        raise ValueError(
            "No allowed models configured. Set ALLOWED_MODELS to a comma-separated "
            "list of model ids the /chat endpoint is permitted to serve."
        )

    client = get_openai_client()
    try:
        served = {model.id async for model in client.models.list()}
    except Exception:
        logger.exception(
            "Could not fetch the model list from %s to validate ALLOWED_MODELS; "
            "skipping the availability check",
            settings.openai_base_url,
        )
        return allowed

    unknown = [name for name in allowed if name not in served]
    if unknown:
        logger.warning(
            "Allowed models not advertised by the OpenAI models endpoint (%s): %s",
            settings.openai_base_url,
            ", ".join(unknown),
        )

    return allowed


async def build_rag_pipeline() -> RAGPipeline | None:
    """Build the RAG pipeline from the corpus datasets at startup.

    ``RAG_CORPUS_DATASETS`` pins the set when it is configured. When it is not,
    every dataset under the corpus folder is discovered from Langfuse, so a
    deployment that has seeded a corpus gets retrieval without a second place to
    keep the list in sync. Returns ``None`` — RAG disabled — when neither yields
    a dataset, including when discovery itself fails: an unreachable Langfuse
    must leave the server up and answering ``enable_rag`` requests with the
    documented 503, not stuck at boot.

    Called once from the app lifespan; the built pipeline is stashed on
    ``app.state`` and served via ``get_rag_pipeline``.
    """
    settings = get_settings()
    names = settings.rag_corpus_dataset_names or await _discover_corpus_datasets(
        settings
    )
    if not names:
        return None
    pipeline = RAGPipeline(get_openai_client(), get_langfuse_client())
    await pipeline.build(names)
    return pipeline


async def _discover_corpus_datasets(settings: Settings) -> list[str]:
    """Every Langfuse dataset under the corpus folder, or ``[]`` if none/failed."""
    try:
        corpus, _questions = await listings.list_dataset_names(
            get_langfuse_client(),
            corpus_folder=settings.corpus_dataset_folder,
            questions_folder=settings.questions_dataset_folder,
        )
    except Exception:
        logger.warning(
            "Could not list Langfuse datasets to discover a RAG corpus; RAG stays "
            "disabled. Set RAG_CORPUS_DATASETS to configure it explicitly.",
            exc_info=True,
        )
        return []
    names = [name for _label, name in corpus]
    if names:
        logger.info(
            "Discovered %d corpus dataset(s) under '%s/': %s",
            len(names),
            settings.corpus_dataset_folder,
            ", ".join(names),
        )
    return names


def get_rag_pipeline(request: Request) -> RAGPipeline | None:
    return getattr(request.app.state, "rag_pipeline", None)


rag_pipeline_dependency = Annotated[RAGPipeline | None, Depends(get_rag_pipeline)]


def get_preset_registry(
    request: Request, langfuse: langfuse_dependency, settings: settings_dependency
) -> PresetRegistry:
    """Serve the process-wide preset registry, building it on first use.

    The lifespan normally warms one up so the dataset is already loaded before
    the first request; the lazy branch is what keeps tests (and any app started
    without the lifespan) working — construction never touches the network and
    the builtins are in service immediately.
    """
    registry = getattr(request.app.state, "preset_registry", None)
    if registry is None:
        registry = PresetRegistry(langfuse=langfuse, settings=settings)
        request.app.state.preset_registry = registry
    return registry


preset_registry_dependency = Annotated[PresetRegistry, Depends(get_preset_registry)]


def get_eval_runner(request: Request) -> EvalRunner:
    """Serve the process-wide eval runner stashed on ``app.state`` at startup.

    Unlike the RAG pipeline this is never ``None`` — the runner is cheap to
    construct and holds no upstream state until a run is actually started.
    """
    return request.app.state.eval_runner


eval_runner_dependency = Annotated[EvalRunner, Depends(get_eval_runner)]
