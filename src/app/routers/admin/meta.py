"""Read-only listings an admin UI needs to populate its pickers.

Every endpoint here proxies a third-party listing. When the upstream fails we
answer 502 rather than an empty list — an admin must be able to tell "Langfuse
is unreachable" from "this project has no datasets".
"""

import logging

from fastapi import APIRouter, HTTPException

from app import listings
from app.agents.tools import get_tool, registered_tool_names
from app.dependencies import langfuse_dependency, openai_dependency, settings_dependency
from app.schema.admin.meta import (
    UPSTREAM_RESPONSES,
    DatasetsResponse,
    ModelDefaults,
    ModelInfo,
    ModelsResponse,
    NamedResource,
    ToolInfo,
)
from rag.config import get_rag_config

router = APIRouter(tags=["Admin"])

logger = logging.getLogger(__name__)


def _bad_gateway(message: str, exc: Exception) -> HTTPException:
    logger.exception(message)
    return HTTPException(status_code=502, detail=f"{message}: {exc}")


def _resources(pairs: list[tuple[str, str]]) -> list[NamedResource]:
    return [NamedResource(name=name, label=label) for label, name in pairs]


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List the models the upstream server advertises",
    description=(
        "Returns every model id and model mode from the configured "
        "OpenAI-compatible server, "
        "unfiltered — a non-empty `ALLOWED_MODELS` only constrains `/chat`, not "
        "which model an admin may evaluate or embed with, so it is reported "
        "alongside rather than applied. An empty list means all models are allowed."
    ),
    responses=UPSTREAM_RESPONSES,
)
async def list_models(openai: openai_dependency, settings: settings_dependency):
    try:
        models = await listings.list_models(openai)
    except Exception as e:
        raise _bad_gateway("Failed to list models", e) from e

    rag_config = get_rag_config()
    return ModelsResponse(
        models=[
            ModelInfo(id=model.id, model_mode=model.model_mode) for model in models
        ],
        allowed_models=settings.allowed_model_names,
        defaults=ModelDefaults(
            eval_model=settings.openai_default_model,
            judge_model=settings.openai_default_model,
            embedding_model=rag_config.embedding_model,
            rerank_model=rag_config.rerank_model,
        ),
    )


@router.get(
    "/datasets",
    response_model=DatasetsResponse,
    summary="List the Langfuse corpus and question datasets",
    responses=UPSTREAM_RESPONSES,
)
async def list_datasets(langfuse: langfuse_dependency, settings: settings_dependency):
    try:
        corpus, questions = await listings.list_dataset_names(
            langfuse,
            corpus_folder=settings.corpus_dataset_folder,
            questions_folder=settings.questions_dataset_folder,
        )
    except Exception as e:
        raise _bad_gateway("Failed to list Langfuse datasets", e) from e

    return DatasetsResponse(corpus=_resources(corpus), questions=_resources(questions))


@router.get(
    "/tools",
    response_model=list[ToolInfo],
    summary="List the server-side tool registry",
    description=(
        "The server-side tool registry, in registry order. Presets expose these "
        "through typed tool sections rather than arbitrary name lists. Nothing "
        "upstream is called, so this cannot 502."
    ),
)
async def list_tools():
    return [
        ToolInfo(
            name=spec.name,
            description=spec.description,
            internal=spec.internal,
            requires_rag=spec.requires_rag,
        )
        for name in registered_tool_names()
        if (spec := get_tool(name)) is not None
    ]


@router.get(
    "/prompts",
    response_model=list[NamedResource],
    summary="List all Langfuse prompts",
    responses=UPSTREAM_RESPONSES,
)
async def list_prompts(langfuse: langfuse_dependency):
    try:
        prompts = await listings.list_prompt_names(langfuse)
    except Exception as e:
        raise _bad_gateway("Failed to list Langfuse prompts", e) from e

    return _resources(prompts)


@router.get(
    "/judge-prompts",
    response_model=list[NamedResource],
    summary="List the Langfuse prompts under the judge folder",
    responses=UPSTREAM_RESPONSES,
)
async def list_judge_prompts(langfuse: langfuse_dependency):
    try:
        prompts = await listings.list_judge_prompt_names(langfuse)
    except Exception as e:
        raise _bad_gateway("Failed to list Langfuse prompts", e) from e

    return _resources(prompts)
