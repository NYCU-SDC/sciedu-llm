"""Global RAG settings and preset-scoped index build controls."""

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.dependencies import (
    preset_registry_dependency,
    rag_cache_dependency,
    settings_dependency,
)
from app.presets import Preset, PresetNotFoundError, PresetRegistry
from app.rag_builds import (
    BuildInProgressError,
    BuildState,
    NoBuildInProgressError,
)
from app.rag_cache import RagBuildCache
from app.schema.admin.rag import (
    ADMIN_RAG_RESPONSES,
    RAGBuildState,
    RAGConfigResponse,
    RAGConfigUpdate,
    RAGConfigUpdateResponse,
    RAGConfigValues,
    RAGSettingsUpdateResponse,
)
from rag.config import RAGConfig

router = APIRouter(prefix="/rag", tags=["Admin"], responses=ADMIN_RAG_RESPONSES)


@router.get("/settings", response_model=RAGConfigValues)
async def get_rag_settings(cache: rag_cache_dependency):
    """Read server-wide settings without selecting a preset or starting a build."""
    return cache.config_snapshot()


@router.patch("/settings", response_model=RAGSettingsUpdateResponse)
async def update_rag_settings(
    update: RAGConfigUpdate,
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
):
    """Apply global settings independently of preset availability."""
    overrides = update.model_dump(
        exclude_unset=True, exclude_none=True, exclude={"rebuild"}
    )
    try:
        cache.apply_overrides(overrides)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    started = cache.warm(registry.snapshot().values()) if update.rebuild else False
    return RAGSettingsUpdateResponse(
        config=cache.config_snapshot(), build_started=started
    )


@router.post("/settings/reset", response_model=RAGSettingsUpdateResponse)
async def reset_rag_settings(
    cache: rag_cache_dependency, registry: preset_registry_dependency
):
    """Reset server-wide settings and warm missing indexes for enabled presets."""
    cache.apply_overrides(RAGConfig().model_dump())
    started = cache.warm(registry.snapshot().values())
    return RAGSettingsUpdateResponse(
        config=cache.config_snapshot(), build_started=started
    )


async def _preset(registry: PresetRegistry, name: str) -> Preset:
    try:
        return await registry.get(name)
    except PresetNotFoundError as exc:
        raise HTTPException(404, f"Unknown preset '{name}'.") from exc


def _snapshot(cache: RagBuildCache, preset: Preset) -> RAGConfigResponse:
    manager = cache.manager(preset)
    state = manager.state if manager else BuildState()
    return RAGConfigResponse(
        **cache.config_snapshot(),
        preset=preset.name,
        is_built=manager.pipeline.is_built if manager else False,
        corpus_datasets=preset.tools.rag.corpus_datasets,
        build=RAGBuildState(
            status=state.status.value,
            corpus_datasets=state.corpus_datasets,
            started_at=state.started_at,
            finished_at=state.finished_at,
            duration_seconds=state.duration_seconds,
            error=state.error,
            cancel_requested=state.cancel_requested,
        ),
    )


@router.get("/config", response_model=RAGConfigResponse)
async def get_rag_config(
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
    settings: settings_dependency,
    preset: str | None = None,
):
    """Global settings, plus index status for the selected preset."""
    selected = await _preset(registry, preset or settings.agents_default_preset)
    cache.ensure(selected)
    return _snapshot(cache, selected)


@router.patch("/config", response_model=RAGConfigUpdateResponse)
async def update_rag_config(
    update: RAGConfigUpdate,
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
    settings: settings_dependency,
    preset: str | None = None,
):
    """Update global settings. Build changes warm all enabled preset corpora.

    Corpus selection is edited through the preset document, never this endpoint.
    With rebuild=false, missing indexes are built on their next use instead.
    """
    selected = await _preset(registry, preset or settings.agents_default_preset)
    overrides = update.model_dump(
        exclude_unset=True, exclude_none=True, exclude={"rebuild"}
    )
    try:
        cache.apply_overrides(overrides)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    started = cache.warm(registry.snapshot().values()) if update.rebuild else False
    return RAGConfigUpdateResponse(
        config=_snapshot(cache, selected), build_started=started
    )


@router.post("/rebuild", response_model=RAGConfigResponse, status_code=202)
async def rebuild_rag(
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
    settings: settings_dependency,
    preset: str | None = None,
):
    """Force a fresh build of this preset's shared cache entry."""
    selected = await _preset(registry, preset or settings.agents_default_preset)
    manager = cache.manager(selected)
    if manager is None:
        raise HTTPException(400, "Enable RAG and select corpus datasets in the preset.")
    try:
        manager.start(sorted(set(selected.tools.rag.corpus_datasets)))
    except BuildInProgressError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _snapshot(cache, selected)


@router.post("/rebuild/cancel", response_model=RAGConfigResponse)
async def cancel_rag_rebuild(
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
    settings: settings_dependency,
    preset: str | None = None,
):
    """Cancel this shared build; other corpus/settings keys are unaffected."""
    selected = await _preset(registry, preset or settings.agents_default_preset)
    manager = cache.manager(selected)
    if manager is None:
        raise HTTPException(409, "No index build is currently running.")
    try:
        manager.cancel()
    except NoBuildInProgressError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _snapshot(cache, selected)


@router.post("/reset", response_model=RAGConfigUpdateResponse)
async def reset_rag_config(
    cache: rag_cache_dependency,
    registry: preset_registry_dependency,
    settings: settings_dependency,
    preset: str | None = None,
):
    """Restore global environment settings and warm missing preset indexes."""
    selected = await _preset(registry, preset or settings.agents_default_preset)
    cache.apply_overrides(RAGConfig().model_dump())
    started = cache.warm(registry.snapshot().values())
    return RAGConfigUpdateResponse(
        config=_snapshot(cache, selected), build_started=started
    )
