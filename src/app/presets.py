"""Named run configurations — the only thing a client picks when it calls the
agentic endpoint.

A preset says *everything* about a run that the server is willing to decide:
which model, which tools the teacher may use, how many steps they get, and how
those tools behave. The stored shape is tool-based rather than cast-based: the
main agent is always the teacher, while enabling the subagent tool optionally
adds one or more forced personas. Clients no longer ship tool lists or step
budgets, so adding a new behaviour is a config change (a Langfuse dataset item)
rather than a deploy.

Two sources, one serving map:

* ``DEFAULT_PRESETS`` — the code-defined defaults (``default-agents`` /
  ``default-chat`` / ``default-chat-plain``), always available, so a dead or
  empty Langfuse still serves both endpoints.
* A Langfuse dataset (``settings.presets_dataset_name``) whose items each hold
  one preset document. Only ``ACTIVE`` items are loaded; archived entries are
  ignored. Active dataset entries *shadow* code defaults of the same name, which
  is how a default gets tuned in production without a release.

The two are joined up at startup by ``ensure_default_presets``: any default with
no dataset item of its own is written into the dataset once, so an operator can
see and edit it. Existing items are never overwritten — "create if missing, then
use whatever is there".

Everything a preset can get wrong is caught at load time by the validators on
``Preset``: a bad dataset item is skipped with its error recorded in
``PresetRegistry.load_errors`` rather than taken into service, so a typo in the
dataset can never take the endpoint down.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from langfuse.api import DatasetStatus
from langfuse.api.commons.errors.not_found_error import NotFoundError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agents.engine import MAX_STEPS_CAP
from app.agents.tools import (
    RAG_SEARCH_TOOL,
    SUMMON_SUBAGENT_TOOL,
    registered_tool_names,
)

logger = logging.getLogger(__name__)

# Preset and character ids end up in URLs, SSE payloads, and Langfuse metadata,
# so they are kept to a boring, lowercase, slug-shaped alphabet.
PRESET_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
MAX_SUBAGENT_PERSONAS = 8

# Fallbacks used when `Settings` does not carry the preset knobs (older config,
# or a fake in tests). The real defaults live in `app.dependencies.Settings`.
DEFAULT_PRESETS_DATASET_NAME = "config/presets"
DEFAULT_PRESETS_CACHE_TTL_SECONDS = 300.0

# Written on the dataset when this deployment has never had one.
PRESETS_DATASET_DESCRIPTION = (
    "/agents 使用的執行設定。每個預設值各有一個項目；項目 id 是預設值名稱，"
    "項目的 input 則是預設值文件。"
)

# After a failed fetch, wait this long before trying again. Without it, a
# Langfuse outage turns every single request into its own failing round trip.
_FAILED_RETRY_SECONDS = 30.0


def is_active_preset_item(item: Any) -> bool:
    """Whether a Langfuse dataset item may participate in preset loading.

    Current Langfuse responses always carry ``status``. Treating a missing
    attribute as active keeps older SDK stand-ins and previously compatible
    clients working, while every explicit non-active status is skipped.
    """
    status = getattr(item, "status", DatasetStatus.ACTIVE)
    value = getattr(status, "value", status)
    return value == DatasetStatus.ACTIVE.value


class PresetCharacter(BaseModel):
    """One runtime speaker derived from a preset's tool configuration.

    ``prompt_name`` means two different things by position, because the two roles
    are prompted differently:

    * on the **orchestrator** — a Langfuse *text* prompt, compiled with no
      variables and prepended to the conversation as a system message;
    * on a **summoned** character — a Langfuse *chat* prompt compiled with
      ``task=`` (the summoner's brief), which becomes its whole context.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=PRESET_ID_PATTERN)
    display_name: str
    role: str = "assistant"
    prompt_name: str | None = None
    tools: list[str] = Field(default_factory=list)
    # The budget this character gets when summoned; ignored for the
    # orchestrator, which uses the preset's `max_steps`.
    max_steps: int = Field(default=3, ge=1, le=MAX_STEPS_CAP)


class RagToolConfig(BaseModel):
    """How the teacher may use retrieval.

    ``enable_tool`` exposes ``rag_search`` to the model. ``force`` instead runs
    retrieval before the model answers; it implies enablement in the stored
    document, even though no callable tool is exposed at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    enable_tool: bool = False
    force: bool = False

    @model_validator(mode="after")
    def _force_requires_enablement(self) -> "RagToolConfig":
        if self.force and not self.enable_tool:
            raise ValueError("force=true requires enable_tool=true")
        return self


class SubagentPersona(BaseModel):
    """One selectable identity available through ``summon_subagent``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=PRESET_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=80)
    prompt_name: str = Field(min_length=1)
    max_steps: int = Field(default=3, ge=1, le=MAX_STEPS_CAP)


class SubagentToolConfig(BaseModel):
    """How the teacher may delegate one task to a subagent.

    With character forcing off, the subagent is a normal assistant that receives
    the teacher's task directly. Turning it on makes ``personas`` selectable by
    id and compiles the chosen persona's Langfuse chat prompt. ``prompt_name`` and
    ``max_steps`` retain the original single-student document shape so existing
    presets continue to round-trip unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    enable_tool: bool = False
    character_forcing: bool = False
    prompt_name: str | None = None
    max_steps: int = Field(default=3, ge=1, le=MAX_STEPS_CAP)
    personas: list[SubagentPersona] = Field(
        default_factory=list,
        max_length=MAX_SUBAGENT_PERSONAS,
        # Keep legacy/default documents byte-shaped as before until they
        # actually opt into multiple personas.
        exclude_if=lambda personas: not personas,
    )

    @model_validator(mode="after")
    def _forcing_requires_an_enabled_tool_and_prompt(self) -> "SubagentToolConfig":
        if self.character_forcing and not self.enable_tool:
            raise ValueError("character_forcing=true requires enable_tool=true")
        if self.character_forcing and not self.personas and not self.prompt_name:
            raise ValueError(
                "character_forcing=true requires personas or the legacy prompt_name"
            )
        ids = [persona.id for persona in self.personas]
        duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if duplicates:
            raise ValueError(f"duplicate persona ids: {', '.join(duplicates)}")
        if "teacher" in ids:
            raise ValueError("persona id 'teacher' is reserved for the main agent")
        return self


class PresetTools(BaseModel):
    """The configurable tool sections stored in one preset document."""

    model_config = ConfigDict(extra="forbid")

    rag: RagToolConfig = Field(default_factory=RagToolConfig)
    subagents: SubagentToolConfig = Field(default_factory=SubagentToolConfig)


class Preset(BaseModel):
    """A complete, validated run configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=PRESET_ID_PATTERN)
    description: str = ""
    # `None` defers to `settings.openai_default_model`, resolved per run so a
    # deployment's default model change does not need every preset edited.
    model: str | None = None
    max_steps: int = Field(default=8, ge=1, le=MAX_STEPS_CAP)
    tool_choice: Literal["auto", "none", "required"] = "auto"
    teacher_prompt_name: str | None = None
    tools: PresetTools

    @model_validator(mode="before")
    @classmethod
    def _upgrade_cast_documents(cls, value: Any) -> Any:
        """Accept the pre-tool-section document shape during migration.

        Existing Langfuse items must keep serving across the rollout. Reads turn
        their old cast/tool lists into the new canonical model; ``model_dump`` and
        every admin write emit only the new shape.
        """
        if not isinstance(value, dict) or "characters" not in value:
            return value

        document = dict(value)
        raw_characters = document.pop("characters", [])
        characters = [
            item.model_dump() if isinstance(item, BaseModel) else item
            for item in raw_characters
            if isinstance(item, (dict, BaseModel))
        ]
        orchestrator_id = document.pop("orchestrator", "assistant")
        rag_mode = document.pop("rag_mode", "off")
        ids = [item.get("id") for item in characters]
        duplicates = sorted(
            {str(id_) for id_ in ids if id_ is not None and ids.count(id_) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate character ids: {', '.join(duplicates)}")
        main = next(
            (item for item in characters if item.get("id") == orchestrator_id), None
        )
        if main is None:
            raise ValueError(
                f"orchestrator '{orchestrator_id}' is not one of the characters: "
                f"{', '.join(str(id_) for id_ in ids)}"
            )
        others = [item for item in characters if item is not main]
        subagent = others[0] if others else {}
        main_tools = list(main.get("tools") or [])
        all_tools = [tool for item in characters for tool in (item.get("tools") or [])]
        unknown = sorted(set(all_tools) - set(registered_tool_names()))
        if unknown:
            raise ValueError(
                f"legacy preset requests unknown tool(s): {', '.join(unknown)}"
            )
        for item in others:
            if SUMMON_SUBAGENT_TOOL in (item.get("tools") or []):
                raise ValueError(
                    f"'{SUMMON_SUBAGENT_TOOL}' is only allowed on the orchestrator"
                )
        if SUMMON_SUBAGENT_TOOL in main_tools and not subagent:
            raise ValueError(f"'{SUMMON_SUBAGENT_TOOL}' needs a second character")
        missing_prompts = [
            str(item.get("id")) for item in others if not item.get("prompt_name")
        ]
        if missing_prompts:
            raise ValueError(
                "summoned character needs a prompt_name: " + ", ".join(missing_prompts)
            )
        if rag_mode == "forced":
            if len(characters) != 1:
                raise ValueError("rag_mode='forced' supports exactly one character")
            if all_tools:
                raise ValueError("rag_mode='forced' does not allow any tools")
            if main.get("prompt_name") is not None:
                raise ValueError("forced RAG character must not set prompt_name")

        subagents_enabled = SUMMON_SUBAGENT_TOOL in main_tools and bool(subagent)
        document["teacher_prompt_name"] = main.get("prompt_name")
        subagent_document: dict[str, Any] = {
            "enable_tool": subagents_enabled,
            "character_forcing": subagents_enabled
            and bool(subagent.get("prompt_name")),
            "prompt_name": subagent.get("prompt_name"),
            "max_steps": subagent.get("max_steps", 3),
        }
        if len(others) > 1:
            subagent_document["personas"] = [
                {
                    "id": item.get("id"),
                    "display_name": item.get("display_name", str(item.get("id"))),
                    "prompt_name": item.get("prompt_name"),
                    "max_steps": item.get("max_steps", 3),
                }
                for item in others
            ]
        document["tools"] = {
            "rag": {
                "enable_tool": rag_mode == "forced" or RAG_SEARCH_TOOL in all_tools,
                "force": rag_mode == "forced",
            },
            "subagents": subagent_document,
        }
        return document

    @model_validator(mode="after")
    def _validate_tool_combination(self) -> "Preset":
        # Forced RAG supplies the teacher's system instructions itself. A second
        # teacher prompt would compete with it, while a student prompt is scoped
        # to a later delegated run and remains valid.
        if self.tools.rag.force and self.teacher_prompt_name is not None:
            raise ValueError(
                "forced RAG supplies the teacher system prompt, so "
                "teacher_prompt_name must be null"
            )
        return self

    @property
    def rag_mode(self) -> Literal["off", "forced"]:
        """Compatibility view consumed by the existing execution pipeline."""
        return "forced" if self.tools.rag.force else "off"

    @property
    def orchestrator(self) -> str:
        """The main agent is fixed as the teacher for every preset."""
        return "teacher"

    @property
    def characters(self) -> list[PresetCharacter]:
        """Derive the runtime cast from the tool-based stored document."""
        teacher_tools: list[str] = []
        if self.tools.rag.enable_tool and not self.tools.rag.force:
            teacher_tools.append(RAG_SEARCH_TOOL)
        if self.tools.subagents.enable_tool:
            teacher_tools.append(SUMMON_SUBAGENT_TOOL)

        characters = [
            PresetCharacter(
                id="teacher",
                display_name="老師",
                role="teacher",
                prompt_name=self.teacher_prompt_name,
                tools=teacher_tools,
            )
        ]
        if not self.tools.subagents.enable_tool:
            return characters

        forcing = self.tools.subagents.character_forcing
        subagent_tools = (
            [RAG_SEARCH_TOOL]
            if self.tools.rag.enable_tool and not self.tools.rag.force
            else []
        )
        if forcing and self.tools.subagents.personas:
            characters.extend(
                PresetCharacter(
                    id=persona.id,
                    display_name=persona.display_name,
                    role=persona.id,
                    prompt_name=persona.prompt_name,
                    tools=subagent_tools,
                    max_steps=persona.max_steps,
                )
                for persona in self.tools.subagents.personas
            )
        else:
            characters.append(
                PresetCharacter(
                    id="student" if forcing else "subagent",
                    display_name="學生" if forcing else "子代理人",
                    role="student" if forcing else "assistant",
                    prompt_name=self.tools.subagents.prompt_name if forcing else None,
                    tools=subagent_tools,
                    max_steps=self.tools.subagents.max_steps,
                )
            )
        return characters


#: The presets this server ships with. Each one is seeded into the preset
#: dataset at startup if it is not already there (``ensure_default_presets``),
#: and each stays in service from code even when Langfuse is unreachable.
DEFAULT_PRESETS: dict[str, Preset] = {
    "default-agents": Preset(
        name="default-agents",
        description=(
            "老師可搜尋課本，也可召喚學生先作答，再加以訂正與補充。"
            "這是 /agents 的預設行為。"
        ),
        max_steps=8,
        teacher_prompt_name="agents/teacher-system",
        tools=PresetTools(
            rag=RagToolConfig(enable_tool=True),
            subagents=SubagentToolConfig(
                enable_tool=True,
                character_forcing=True,
                prompt_name="agents/student",
                max_steps=3,
            ),
        ),
    ),
    "default-chat": Preset(
        name="default-chat",
        description=(
            "老師透過 rag_search 工具自行判斷何時搜尋課本。"
            "/chat 在 enable_rag 為 true 時使用此預設值。"
        ),
        # A model-chosen search costs a step to call and a step to answer from,
        # so a single-step budget would make the tool unusable.
        max_steps=8,
        # No teacher prompt: /chat's contract is that the server injects no
        # persona of its own, and the caller's own system message is the only one
        # the model sees.
        tools=PresetTools(rag=RagToolConfig(enable_tool=True)),
    ),
    "default-chat-plain": Preset(
        name="default-chat-plain",
        description=(
            "單輪老師，不使用工具或檢索。/chat 在 enable_rag 為 false 時使用此預設值。"
        ),
        max_steps=1,
        tools=PresetTools(),
    ),
}


class PresetNotFoundError(KeyError):
    """No preset by that name is being served."""


@dataclass(frozen=True)
class PresetLoadReport:
    """What one refresh produced.

    ``loaded`` is every preset name now being served (code defaults included), not
    just the dataset's — it answers "what can I run right now". ``fetched_at`` is
    a wall-clock timestamp, and ``None`` when the dataset fetch itself failed and
    the previous map is still in service.
    """

    loaded: list[str]
    errors: dict[str, str]
    fetched_at: float | None


class PresetRegistry:
    """Serves presets, refreshing the dataset half of them on a TTL.

    Construction never touches the network and never raises: the code defaults
    are in service from the first instant, and the first ``get()`` pulls the
    dataset in. A failed refresh keeps whatever was already loaded.
    """

    def __init__(self, *, langfuse: Any, settings: Any) -> None:
        self._langfuse = langfuse
        self._settings = settings
        self._presets: dict[str, Preset] = dict(DEFAULT_PRESETS)
        self._load_errors: dict[str, str] = {}
        self._fetched_at: float | None = None
        # A monotonic deadline. 0.0 means "never loaded", so the first `get()`
        # fetches.
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    # --- public surface ----------------------------------------------------

    async def get(self, name: str) -> Preset:
        await self._refresh_if_stale()
        preset = self._presets.get(name)
        if preset is None:
            raise PresetNotFoundError(name)
        return preset

    def names(self) -> list[str]:
        """Preset names as of the last refresh (no fetch — this is sync)."""
        return sorted(self._presets)

    def snapshot(self) -> dict[str, Preset]:
        """A copy of the serving map, safe to iterate while a refresh happens."""
        return dict(self._presets)

    async def refresh(self) -> PresetLoadReport:
        """Reload from the dataset now, ignoring the TTL."""
        async with self._lock:
            return await self._load()

    @property
    def load_errors(self) -> dict[str, str]:
        return dict(self._load_errors)

    # --- refresh mechanics -------------------------------------------------

    @property
    def _ttl(self) -> float:
        return float(
            getattr(
                self._settings,
                "presets_cache_ttl_seconds",
                DEFAULT_PRESETS_CACHE_TTL_SECONDS,
            )
        )

    @property
    def _dataset_name(self) -> str:
        return getattr(
            self._settings, "presets_dataset_name", DEFAULT_PRESETS_DATASET_NAME
        )

    async def _refresh_if_stale(self) -> None:
        if time.monotonic() < self._expires_at:
            return
        async with self._lock:
            # Single-flight: whoever held the lock has already refreshed (or
            # armed the failure backoff), so concurrent callers pile up behind
            # one fetch instead of each launching their own.
            if time.monotonic() < self._expires_at:
                return
            await self._load()

    async def _load(self) -> PresetLoadReport:
        """Fetch and validate the dataset, then swap the serving map atomically.

        Caller must hold ``self._lock``.
        """
        name = self._dataset_name
        try:
            # `get_dataset` is a blocking HTTP call. AttributeError lands here
            # too, which is deliberate: a Langfuse stand-in without the method
            # degrades to the code defaults rather than breaking every request.
            dataset = await asyncio.to_thread(self._langfuse.get_dataset, name)
            items = [
                item
                for item in (getattr(dataset, "items", None) or [])
                if is_active_preset_item(item)
            ]
        except Exception as e:
            logger.exception(
                "could not load the preset dataset '%s'; continuing to serve the "
                "%d preset(s) already loaded",
                name,
                len(self._presets),
            )
            self._load_errors = {f"dataset:{name}": str(e) or type(e).__name__}
            self._expires_at = time.monotonic() + _FAILED_RETRY_SECONDS
            return PresetLoadReport(
                loaded=sorted(self._presets),
                errors=self.load_errors,
                fetched_at=None,
            )

        loaded, errors = _parse_items(items)
        # Dataset entries shadow the code defaults of the same name.
        self._presets = dict(DEFAULT_PRESETS) | loaded
        self._load_errors = errors
        self._fetched_at = time.time()
        self._expires_at = time.monotonic() + self._ttl
        if errors:
            logger.warning(
                "skipped %d malformed item(s) in preset dataset '%s': %s",
                len(errors),
                name,
                "; ".join(f"{key}: {value}" for key, value in errors.items()),
            )
        return PresetLoadReport(
            loaded=sorted(self._presets),
            errors=dict(errors),
            fetched_at=self._fetched_at,
        )


async def ensure_default_presets(langfuse: Any, settings: Any) -> list[str]:
    """Write the missing ``DEFAULT_PRESETS`` into the preset dataset.

    "Create if not found, otherwise use what is already there": an existing item
    is *never* rewritten, because it may well have been tuned in production
    through ``PUT /admin/presets/{name}`` since it was seeded, and the registry
    already prefers it over the code default anyway. Seeding only exists so an
    operator can *see* the shipped defaults in Langfuse — and edit them — rather
    than having to guess what a name they cannot find in the dataset does.

    Returns the names created (empty when everything was already present).
    Raises whatever Langfuse raises: the caller (the app lifespan) decides that
    a Langfuse outage must not stop the app from booting, since the code
    defaults stay in service either way.
    """
    dataset_name = getattr(
        settings, "presets_dataset_name", DEFAULT_PRESETS_DATASET_NAME
    )
    # `get_dataset` and `create_dataset_item` are blocking HTTP calls.
    try:
        dataset = await asyncio.to_thread(langfuse.get_dataset, dataset_name)
        items = list(getattr(dataset, "items", None) or [])
    except NotFoundError:
        logger.info("creating the preset dataset '%s'", dataset_name)
        await asyncio.to_thread(
            lambda: langfuse.create_dataset(
                name=dataset_name, description=PRESETS_DATASET_DESCRIPTION
            )
        )
        items = []

    present = _existing_preset_names(items)
    created: list[str] = []
    for name, preset in DEFAULT_PRESETS.items():
        if name in present:
            continue
        document = preset.model_dump()
        await asyncio.to_thread(
            lambda name=name, document=document: langfuse.create_dataset_item(
                dataset_name=dataset_name, id=name, input=document
            )
        )
        created.append(name)
    return created


def _existing_preset_names(items: list[Any]) -> set[str]:
    """Every preset name the dataset already holds, however it is keyed.

    Items written by this service (and by ``/admin/presets``) use the preset
    name as the item id, but an item hand-created in the Langfuse UI can carry
    any id — so a document's own ``name`` counts as "already there" too, or
    seeding would quietly duplicate it under a second id.
    """
    names: set[str] = set()
    for item in items:
        item_id = getattr(item, "id", None)
        if item_id:
            names.add(str(item_id))
        document = getattr(item, "input", None)
        if isinstance(document, str):
            try:
                document = json.loads(document)
            except json.JSONDecodeError:
                continue
        if isinstance(document, dict) and isinstance(document.get("name"), str):
            names.add(document["name"])
    return names


def _parse_items(items: list[Any]) -> tuple[dict[str, Preset], dict[str, str]]:
    """Validate dataset items into ``(presets_by_name, errors_by_item_id)``.

    Failures are per item: one unparseable document must not cost the others
    their place in service.
    """
    presets: dict[str, Preset] = {}
    errors: dict[str, str] = {}
    for item in items:
        item_id = str(getattr(item, "id", None) or "?")
        document = getattr(item, "input", None)
        if isinstance(document, str):
            # Some dataset UIs store the document as a JSON string rather than
            # an object; both are accepted.
            try:
                document = json.loads(document)
            except json.JSONDecodeError as e:
                errors[item_id] = f"input is not valid JSON: {e}"
                continue
        if not isinstance(document, dict):
            errors[item_id] = "input is not a preset document"
            continue
        try:
            preset = Preset.model_validate(document)
        except ValidationError as e:
            errors[item_id] = _format_validation_error(e)
            continue
        if preset.name in presets:
            errors[item_id] = (
                f"duplicate preset name '{preset.name}'; the earlier item wins"
            )
            continue
        presets[preset.name] = preset
    return presets, errors


def _format_validation_error(error: ValidationError) -> str:
    """Flatten a pydantic error into one short line for the load report."""
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "(root)"
        problems.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(problems)
