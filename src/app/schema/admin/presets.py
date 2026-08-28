"""Response models for `/admin/presets`.

The request body for an upsert is :class:`app.presets.Preset` itself — the raw
preset document, validated by FastAPI exactly as the registry validates dataset
items, so an admin authoring a preset gets the same errors the loader would
report rather than a second, looser schema.

The responses keep the document and the flags in separate places on purpose:
``definition`` is the preset as stored, so a UI can round-trip it straight back
into ``PUT`` without stripping server-added bookkeeping fields (``Preset``
forbids extras, so a flattened payload would fail to re-submit).
"""

from typing import Any

from pydantic import BaseModel, Field

from app.presets import Preset


class PresetSummary(BaseModel):
    """One preset in the serving map, with where it came from.

    The two flags answer "is this code-defined or dataset-defined", which is
    also what says whether it can be deleted:

    * ``builtin`` — a code-defined *default* preset of this name exists
      (``app.presets.DEFAULT_PRESETS``). It is always available, even with
      Langfuse down.
    * ``shadowed_builtin`` — that code-defined default exists *and* a dataset
      item of the same name is what is currently being served. Deleting the
      dataset item puts the code default back in service. Startup seeding
      writes the defaults into the dataset, so on a seeded deployment they
      normally read as shadowed.

    So a preset is dataset-defined (and therefore deletable) when
    ``not builtin or shadowed_builtin``.
    """

    name: str
    description: str = ""
    builtin: bool = Field(
        description=(
            "A code-defined default preset of this name exists, so the name "
            "cannot be removed."
        )
    )
    shadowed_builtin: bool = Field(
        description=(
            "A dataset item is currently overriding the code-defined default of "
            "this name. Deleting the item restores the code-defined preset."
        )
    )


class PresetDetail(PresetSummary):
    """A preset's full document alongside its provenance flags."""

    definition: Preset = Field(
        description=(
            "The preset document as served. Submit it unchanged to "
            "PUT /admin/presets/{name} to round-trip an edit."
        )
    )

    @classmethod
    def build(
        cls, preset: Preset, *, builtin: bool, shadowed_builtin: bool
    ) -> "PresetDetail":
        return cls(
            name=preset.name,
            description=preset.description,
            builtin=builtin,
            shadowed_builtin=shadowed_builtin,
            definition=preset,
        )


class PresetLoadReportResponse(BaseModel):
    """The outcome of a registry refresh.

    ``loaded`` is every preset now being served, code defaults included — it answers
    "what can I run right now", not "what did the dataset contain". ``errors``
    is keyed by dataset *item id*: those items were skipped, and everything else
    stayed in service. ``fetched_at`` is ``None`` when the dataset fetch itself
    failed and the previously loaded map is still being served.
    """

    loaded: list[str]
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Dataset item id -> why that item was skipped.",
    )
    fetched_at: float | None = Field(
        default=None,
        description=(
            "Unix timestamp of the successful fetch, or null when the fetch "
            "failed and the previous presets are still in service."
        ),
    )


ADMIN_PRESET_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "description": "Not Found - no preset of that name is being served",
        "content": {
            "application/json": {"example": {"detail": "Unknown preset 'socratic'."}}
        },
    },
    502: {
        "description": "Bad Gateway - writing to the Langfuse preset dataset failed",
        "content": {
            "application/json": {
                "example": {"detail": "Failed to write preset 'socratic': <reason>"}
            }
        },
    },
}

PRESET_DELETE_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ADMIN_PRESET_RESPONSES,
    409: {
        "description": "Conflict - code-defined default presets are permanent",
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "Preset 'default-chat' is built in and cannot be "
                        "deleted. Override it by PUTting a preset of the same "
                        "name instead."
                    )
                }
            }
        },
    },
}
