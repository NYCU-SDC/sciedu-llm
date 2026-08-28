"""Author the presets the agentic endpoint serves.

A preset lives in one of two places: in code (``DEFAULT_PRESETS``, always
served, never deletable) or as an item in the Langfuse dataset named by
``settings.presets_dataset_name``. The two overlap by design — the code defaults
are seeded into the dataset at startup when missing — but the code copy is what
survives a Langfuse outage. This router is the write half of the dataset side:
it validates a document with the *same* schema the loader uses, stores it as a
dataset item keyed by the preset name, and refreshes the registry so the change
is live before the response is sent, rather than up to a TTL later.

Reading the last load result: there is deliberately no ``GET /load-report``.
``POST /refresh`` is cheap, idempotent, and returns the same report, so a panel
that wants to show "which items failed to validate" calls that instead of
maintaining a second, possibly staler view of the same state.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from langfuse import Langfuse
from langfuse.api.commons.errors.not_found_error import NotFoundError

from app.dependencies import (
    langfuse_dependency,
    preset_registry_dependency,
    settings_dependency,
)
from app.presets import (
    DEFAULT_PRESETS,
    DEFAULT_PRESETS_DATASET_NAME,
    PRESETS_DATASET_DESCRIPTION,
    Preset,
    PresetLoadReport,
    PresetRegistry,
)
from app.schema.admin.presets import (
    ADMIN_PRESET_RESPONSES,
    PRESET_DELETE_RESPONSES,
    PresetDetail,
    PresetLoadReportResponse,
    PresetSummary,
)

# Mounted under the `/admin` aggregator in `app.routers.admin`, so the paths
# below resolve to `/admin/presets/...`.
router = APIRouter(prefix="/presets", tags=["Admin"])

logger = logging.getLogger(__name__)


def _dataset_name(settings: Any) -> str:
    return getattr(settings, "presets_dataset_name", DEFAULT_PRESETS_DATASET_NAME)


def _report(report: PresetLoadReport) -> PresetLoadReportResponse:
    return PresetLoadReportResponse(
        loaded=report.loaded, errors=report.errors, fetched_at=report.fetched_at
    )


def _flags(name: str, served: Preset) -> tuple[bool, bool]:
    """``(builtin, shadowed_builtin)`` for one entry of the serving map.

    "Builtin" on the wire means *code-defined default* (``DEFAULT_PRESETS``).

    Identity, not equality: the registry rebuilds its map as
    ``dict(DEFAULT_PRESETS) | loaded``, so the code default object survives
    verbatim unless a dataset item replaced it. A dataset item whose content
    happens to match the default is still dataset-defined — and still deletable
    — so it must read as shadowing. Startup seeding writes exactly such an item,
    so a freshly seeded deployment reports its defaults as shadowed.
    """
    default = DEFAULT_PRESETS.get(name)
    if default is None:
        return False, False
    return True, served is not default


def _detail(name: str, served: Preset) -> PresetDetail:
    builtin, shadowed = _flags(name, served)
    return PresetDetail.build(served, builtin=builtin, shadowed_builtin=shadowed)


def _require(registry: PresetRegistry, name: str) -> Preset:
    served = registry.snapshot().get(name)
    if served is None:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{name}'.")
    return served


def _bad_gateway(message: str, exc: Exception) -> HTTPException:
    logger.exception(message)
    return HTTPException(status_code=502, detail=f"{message}: {exc}")


async def _find_item(langfuse: Langfuse, dataset_name: str, preset_name: str) -> Any:
    """The dataset item holding ``preset_name``, or ``None``.

    Items written here are keyed by preset name, but an item hand-created in the
    Langfuse UI can carry any id, so fall back to matching the document's own
    ``name``. Returns ``None`` when the dataset itself does not exist.
    """
    try:
        dataset = await asyncio.to_thread(langfuse.get_dataset, dataset_name)
    except NotFoundError:
        return None
    items = list(getattr(dataset, "items", None) or [])
    for item in items:
        if str(getattr(item, "id", "")) == preset_name:
            return item
    for item in items:
        document = getattr(item, "input", None)
        if isinstance(document, dict) and document.get("name") == preset_name:
            return item
    return None


@router.get(
    "",
    response_model=list[PresetSummary],
    summary="List every preset currently being served",
    description=(
        "Code-defined defaults and dataset presets together, sorted by name. "
        "Reads the registry's current serving map without forcing a fetch — call "
        "`POST /admin/presets/refresh` first to be sure it is up to date."
    ),
)
async def list_presets(registry: preset_registry_dependency):
    summaries = []
    for name, served in sorted(registry.snapshot().items()):
        builtin, shadowed = _flags(name, served)
        summaries.append(
            PresetSummary(
                name=name,
                description=served.description,
                builtin=builtin,
                shadowed_builtin=shadowed,
            )
        )
    return summaries


@router.get(
    "/{name}",
    response_model=PresetDetail,
    summary="Get one preset's full document",
    responses=ADMIN_PRESET_RESPONSES,
)
async def get_preset(name: str, registry: preset_registry_dependency):
    return _detail(name, _require(registry, name))


@router.put(
    "/{name}",
    response_model=PresetDetail,
    summary="Create or replace a dataset preset",
    description=(
        "Validates the document with the same schema the loader applies to "
        "dataset items (invalid documents are rejected with 422 before anything "
        "is written), stores it as the dataset item `{name}`, then refreshes the "
        "registry so the preset is live in this response. A document whose "
        "`name` differs from the path is rejected — the two are the same "
        "identity. Writing a preset named after a code default shadows that "
        "default."
    ),
    responses=ADMIN_PRESET_RESPONSES,
)
async def upsert_preset(
    name: str,
    preset: Preset,
    registry: preset_registry_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
):
    if preset.name != name:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Preset document name '{preset.name}' does not match the path "
                f"name '{name}'."
            ),
        )

    dataset_name = _dataset_name(settings)
    document = preset.model_dump()
    try:
        await _ensure_dataset(langfuse, dataset_name)
        # `create_dataset_item` upserts on a reused id, so this is the same call
        # for a create and for an edit.
        await asyncio.to_thread(
            lambda: langfuse.create_dataset_item(
                dataset_name=dataset_name, id=name, input=document
            )
        )
    except Exception as e:
        raise _bad_gateway(f"Failed to write preset '{name}'", e) from e

    await registry.refresh()
    return _detail(name, _require(registry, name))


async def _ensure_dataset(langfuse: Langfuse, dataset_name: str) -> None:
    """Create the preset dataset if this deployment has never had one."""
    try:
        await asyncio.to_thread(langfuse.get_dataset, dataset_name)
    except NotFoundError:
        await asyncio.to_thread(
            lambda: langfuse.create_dataset(
                name=dataset_name, description=PRESETS_DATASET_DESCRIPTION
            )
        )


@router.delete(
    "/{name}",
    status_code=204,
    summary="Delete a dataset preset",
    description=(
        "Removes the dataset item and refreshes the registry. Deleting a preset "
        "that was shadowing a code default does not remove the name — the "
        "code-defined preset is served again from the next request. A code "
        "default with no dataset item of its own cannot be deleted (409)."
    ),
    responses=PRESET_DELETE_RESPONSES,
)
async def delete_preset(
    name: str,
    registry: preset_registry_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
):
    dataset_name = _dataset_name(settings)
    try:
        item = await _find_item(langfuse, dataset_name, name)
    except Exception as e:
        raise _bad_gateway(
            f"Failed to read the preset dataset '{dataset_name}'", e
        ) from e

    if item is None:
        if name in DEFAULT_PRESETS:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Preset '{name}' is built in and cannot be deleted. Override "
                    "it by PUTting a preset of the same name instead."
                ),
            )
        raise HTTPException(status_code=404, detail=f"Unknown preset '{name}'.")

    item_id = item.id
    # No percent-encoding needed, unlike `list_experiment_runs`: this path
    # parameter is an item id — either a preset name (slug-shaped, no slashes)
    # or a Langfuse-generated cuid — never a foldered dataset name.
    try:
        await asyncio.to_thread(lambda: langfuse.api.dataset_items.delete(id=item_id))
    except Exception as e:
        raise _bad_gateway(f"Failed to delete preset '{name}'", e) from e

    await registry.refresh()
    return Response(status_code=204)


@router.post(
    "/refresh",
    response_model=PresetLoadReportResponse,
    summary="Reload the presets from Langfuse now",
    description=(
        "Ignores the cache TTL and re-reads the dataset, returning what is now "
        "being served and, per dataset item id, why any item was skipped. A "
        "failed fetch is not an error here: the previously loaded presets stay "
        "in service and `fetched_at` comes back null."
    ),
)
async def refresh_presets(registry: preset_registry_dependency):
    return _report(await registry.refresh())
