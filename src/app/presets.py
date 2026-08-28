"""Named run configurations — the only thing a client picks when it calls the
agentic endpoint.

A preset says *everything* about a run that the server is willing to decide:
which model, which characters exist, what each of them may call, how many steps
they get, and whether retrieval is forced. Clients no longer ship tool lists or
step budgets, so adding a new behaviour is a config change (a Langfuse dataset
item) rather than a deploy.

Two sources, one serving map:

* ``DEFAULT_PRESETS`` — the code-defined defaults (``default-agents`` /
  ``default-chat`` / ``default-chat-plain``), always available, so a dead or
  empty Langfuse still serves both endpoints.
* A Langfuse dataset (``settings.presets_dataset_name``) whose items each hold
  one preset document. Dataset entries *shadow* code defaults of the same name,
  which is how a default gets tuned in production without a release.

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

# Fallbacks used when `Settings` does not carry the preset knobs (older config,
# or a fake in tests). The real defaults live in `app.dependencies.Settings`.
DEFAULT_PRESETS_DATASET_NAME = "config/presets"
DEFAULT_PRESETS_CACHE_TTL_SECONDS = 300.0

# Written on the dataset when this deployment has never had one.
PRESETS_DATASET_DESCRIPTION = (
    "Run configurations served by /agents. One item per preset; the item id is "
    "the preset name and the item input is the preset document."
)

# After a failed fetch, wait this long before trying again. Without it, a
# Langfuse outage turns every single request into its own failing round trip.
_FAILED_RETRY_SECONDS = 30.0


class PresetCharacter(BaseModel):
    """One speaker in a preset.

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
    # "forced" prepends the RAG-compiled system prompt and swaps the latest user
    # turn for the context-augmented one, before the model gets a say — what
    # /chat's `enable_rag` used to do. "off" leaves retrieval to the
    # `rag_search` tool, if the preset grants it, which is how every preset this
    # server ships retrieves; "forced" stays for a deployment that wants
    # retrieval on every single turn.
    rag_mode: Literal["off", "forced"] = "off"
    orchestrator: str = "assistant"
    characters: list[PresetCharacter] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _validate_shape(self) -> "Preset":
        ids = [character.id for character in self.characters]
        duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if duplicates:
            raise ValueError(f"duplicate character ids: {', '.join(duplicates)}")
        if self.orchestrator not in ids:
            raise ValueError(
                f"orchestrator '{self.orchestrator}' is not one of the "
                f"characters: {', '.join(ids)}"
            )

        known_tools = set(registered_tool_names())
        for character in self.characters:
            unknown = [name for name in character.tools if name not in known_tools]
            if unknown:
                raise ValueError(
                    f"character '{character.id}' requests unknown tool(s): "
                    f"{', '.join(unknown)}. Available tools: "
                    f"{', '.join(sorted(known_tools))}."
                )

        orchestrator = next(c for c in self.characters if c.id == self.orchestrator)
        # Only the orchestrator can summon, and only if there is somebody to
        # summon: a preset granting the tool with nobody to call would fail at
        # run time as a tool error the model cannot fix.
        for character in self.characters:
            if character.id == self.orchestrator:
                continue
            if SUMMON_SUBAGENT_TOOL in character.tools:
                raise ValueError(
                    f"'{SUMMON_SUBAGENT_TOOL}' is only allowed on the "
                    f"orchestrator, not on character '{character.id}'"
                )
            if not character.prompt_name:
                raise ValueError(
                    f"character '{character.id}' is summoned and therefore needs "
                    "a prompt_name"
                )
        if SUMMON_SUBAGENT_TOOL in orchestrator.tools and len(self.characters) < 2:
            raise ValueError(
                f"'{SUMMON_SUBAGENT_TOOL}' needs a second character to summon"
            )

        if self.rag_mode == "forced":
            # Forced RAG *is* the prompt: the pipeline supplies the system
            # message and rewrites the user turn, so a second system prompt or a
            # retrieval tool would only fight it.
            if len(self.characters) != 1:
                raise ValueError(
                    "rag_mode='forced' supports exactly one character, got "
                    f"{len(self.characters)}"
                )
            only = self.characters[0]
            if only.tools:
                raise ValueError("rag_mode='forced' does not allow any tools")
            if only.prompt_name is not None:
                raise ValueError(
                    "rag_mode='forced' supplies the system prompt itself, so "
                    "the character must not set prompt_name"
                )

        return self


#: The presets this server ships with. Each one is seeded into the preset
#: dataset at startup if it is not already there (``ensure_default_presets``),
#: and each stays in service from code even when Langfuse is unreachable.
DEFAULT_PRESETS: dict[str, Preset] = {
    "default-agents": Preset(
        name="default-agents",
        description=(
            "A teacher who can search the textbook and summon a student to "
            "answer first, then correct and extend the student's answer. The "
            "default /agents behaviour."
        ),
        max_steps=8,
        orchestrator="teacher",
        characters=[
            PresetCharacter(
                id="teacher",
                display_name="老師",
                role="teacher",
                prompt_name="agents/teacher-system",
                # Every registered tool, named one by one rather than pulled
                # from the registry: what the default cast may call is a product
                # decision, not "whatever happens to be installed".
                tools=[RAG_SEARCH_TOOL, SUMMON_SUBAGENT_TOOL],
            ),
            PresetCharacter(
                id="student",
                display_name="學生",
                role="student",
                prompt_name="agents/student",
                tools=[RAG_SEARCH_TOOL],
                max_steps=3,
            ),
        ],
    ),
    "default-chat": Preset(
        name="default-chat",
        description=(
            "Assistant that decides for itself when to search the textbook, via "
            "the rag_search tool. What /chat runs for `enable_rag: true`."
        ),
        # A model-chosen search costs a step to call and a step to answer from,
        # so a single-step budget would make the tool unusable.
        max_steps=8,
        orchestrator="assistant",
        characters=[
            # No `prompt_name`: /chat's contract is that the server injects no
            # persona of its own, and the caller's own system message is the
            # only one the model sees.
            PresetCharacter(
                id="assistant", display_name="助教", tools=[RAG_SEARCH_TOOL]
            )
        ],
    ),
    "default-chat-plain": Preset(
        name="default-chat-plain",
        description=(
            "Plain single-turn assistant. No tools, no retrieval. What /chat "
            "runs for `enable_rag: false`."
        ),
        max_steps=1,
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="助教")],
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
            items = list(getattr(dataset, "items", None) or [])
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
