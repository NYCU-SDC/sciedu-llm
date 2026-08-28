import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "mock_key")

import pytest
from langfuse.api.commons.errors.not_found_error import NotFoundError
from pydantic import ValidationError

from app.agents import errors
from app.agents.cast import build_cast
from app.agents.events import DoneEvent, ErrorEvent
from app.agents.tools import registered_tool_names
from app.agents.run import (
    PresetRunError,
    latest_user_message,
    prepare_preset_run,
    run_preset,
)
from app.presets import (
    DEFAULT_PRESETS,
    DEFAULT_PRESETS_DATASET_NAME,
    Preset,
    PresetNotFoundError,
    PresetRegistry,
    ensure_default_presets,
)

# --- fakes ------------------------------------------------------------------
# SimpleNamespace stand-ins in the style of tests/app/agents_test.py, kept local
# rather than shared: these fakes only need to answer the two or three calls the
# preset machinery makes.


def _item(id_: str, document):
    return SimpleNamespace(id=id_, input=document)


class _FakeDataset:
    def __init__(self, items):
        self.items = list(items)


class _FakeLangfuse:
    """A Langfuse whose dataset content (or failure) is scripted per test."""

    def __init__(self, items=(), error: Exception | None = None):
        self.items = list(items)
        self.error = error
        self.dataset_calls: list[str] = []
        self.prompt_calls: list[str] = []
        self.prompt_error: Exception | None = None

    def get_dataset(self, name):
        self.dataset_calls.append(name)
        if self.error is not None:
            raise self.error
        return _FakeDataset(self.items)

    def get_prompt(self, name, type=None):
        self.prompt_calls.append(name)
        if self.prompt_error is not None:
            raise self.prompt_error
        return SimpleNamespace(compile=lambda **variables: f"PROMPT<{name}>")


class _SlowLangfuse(_FakeLangfuse):
    """Blocks inside ``get_dataset`` so overlapping refreshes are observable."""

    def get_dataset(self, name):
        self.dataset_calls.append(name)
        import time

        time.sleep(0.05)
        return _FakeDataset(self.items)


def _settings(**overrides):
    base = dict(
        openai_default_model="gpt-oss-120b",
        allowed_model_names=[],
        presets_dataset_name="config/presets",
        presets_cache_ttl_seconds=300.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeRAGPipeline:
    def __init__(self):
        self.retrieve_calls: list[str] = []

    async def retrieve(self, *, query: str, **_kwargs):
        self.retrieve_calls.append(query)
        return {"context": f"CTX for {query}", "reference_chunks": [1, 2]}

    def compile_generator_prompt(self, *, context: str, query: str):
        return (
            {"role": "system", "content": f"RAG SYSTEM<{context}>"},
            {"role": "user", "content": f"RAG USER<{query}>"},
            SimpleNamespace(name="rag/generator-user"),
        )


class _FailingRAGPipeline:
    async def retrieve(self, *, query: str, **_kwargs):
        raise RuntimeError("connection to https://secret.internal:8080 refused")


def _preset(**overrides) -> dict:
    """A minimal valid preset document, ready to be broken one field at a time."""
    document = {
        "name": "demo",
        "characters": [{"id": "assistant", "display_name": "助教"}],
    }
    document.update(overrides)
    return document


def _forced_rag_preset() -> Preset:
    """A preset that retrieves before the model speaks.

    No default ships with ``rag_mode: "forced"`` any more — /chat's `enable_rag`
    offers the model a search tool instead — but the mode is still part of the
    schema and still supported by ``run_preset``, so a deployment can author one
    in the dataset. These tests are where that behaviour is pinned.
    """
    return Preset.model_validate(_preset(name="forced-rag", rag_mode="forced"))


# --- schema validation ------------------------------------------------------


def test_minimal_preset_takes_the_documented_defaults():
    preset = Preset.model_validate(_preset())

    assert preset.model is None
    assert preset.max_steps == 8
    assert preset.tool_choice == "auto"
    assert preset.rag_mode == "off"
    assert preset.orchestrator == "assistant"
    assert preset.characters[0].role == "assistant"
    assert preset.characters[0].tools == []
    assert preset.characters[0].max_steps == 3


@pytest.mark.parametrize("name", ["Demo", "-demo", "de mo", "", "d" * 65])
def test_preset_name_must_be_a_slug(name):
    with pytest.raises(ValidationError):
        Preset.model_validate(_preset(name=name))


def test_preset_rejects_unknown_fields():
    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(_preset(temperature=0.7))

    assert "temperature" in str(excinfo.value)


def test_preset_rejects_a_step_budget_over_the_engine_cap():
    with pytest.raises(ValidationError):
        Preset.model_validate(_preset(max_steps=99))
    with pytest.raises(ValidationError):
        Preset.model_validate(_preset(max_steps=0))


def test_preset_rejects_duplicate_character_ids():
    document = _preset(
        characters=[
            {"id": "assistant", "display_name": "A"},
            {"id": "assistant", "display_name": "B", "prompt_name": "p"},
        ]
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "duplicate character ids" in str(excinfo.value)


def test_preset_rejects_an_orchestrator_that_is_not_a_character():
    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(_preset(orchestrator="teacher"))

    assert "is not one of the characters" in str(excinfo.value)


def test_preset_rejects_an_unregistered_tool():
    document = _preset(
        characters=[
            {"id": "assistant", "display_name": "助教", "tools": ["search_textbook"]}
        ]
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "unknown tool(s): search_textbook" in str(excinfo.value)


def test_preset_accepts_a_registered_tool():
    document = _preset(
        characters=[
            {"id": "assistant", "display_name": "助教", "tools": ["rag_search"]}
        ]
    )

    assert Preset.model_validate(document).characters[0].tools == ["rag_search"]


def test_preset_rejects_summon_without_anybody_to_summon():
    document = _preset(
        characters=[
            {"id": "assistant", "display_name": "助教", "tools": ["summon_subagent"]}
        ]
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "needs a second character" in str(excinfo.value)


def test_preset_rejects_summon_on_a_non_orchestrator():
    document = _preset(
        orchestrator="teacher",
        characters=[
            {"id": "teacher", "display_name": "老師"},
            {
                "id": "student",
                "display_name": "學生",
                "prompt_name": "agents/student",
                "tools": ["summon_subagent"],
            },
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "only allowed on the orchestrator" in str(excinfo.value)


def test_preset_accepts_summon_on_the_orchestrator_of_a_pair():
    document = _preset(
        orchestrator="teacher",
        characters=[
            {"id": "teacher", "display_name": "老師", "tools": ["summon_subagent"]},
            {
                "id": "student",
                "display_name": "學生",
                "prompt_name": "agents/student",
            },
        ],
    )

    assert len(Preset.model_validate(document).characters) == 2


def test_preset_requires_a_prompt_name_on_a_summoned_character():
    document = _preset(
        orchestrator="teacher",
        characters=[
            {"id": "teacher", "display_name": "老師", "tools": ["summon_subagent"]},
            {"id": "student", "display_name": "學生"},
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "needs a prompt_name" in str(excinfo.value)


def test_forced_rag_preset_is_accepted_when_it_is_solo_toolless_and_promptless():
    preset = Preset.model_validate(_preset(rag_mode="forced"))

    assert preset.rag_mode == "forced"


def test_forced_rag_rejects_a_second_character():
    document = _preset(
        rag_mode="forced",
        orchestrator="teacher",
        characters=[
            {"id": "teacher", "display_name": "老師"},
            {"id": "student", "display_name": "學生", "prompt_name": "p"},
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "exactly one character" in str(excinfo.value)


def test_forced_rag_rejects_tools():
    document = _preset(
        rag_mode="forced",
        characters=[
            {"id": "assistant", "display_name": "助教", "tools": ["rag_search"]}
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "does not allow any tools" in str(excinfo.value)


def test_forced_rag_rejects_a_prompt_name():
    document = _preset(
        rag_mode="forced",
        characters=[
            {"id": "assistant", "display_name": "助教", "prompt_name": "agents/x"}
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        Preset.model_validate(document)

    assert "must not set prompt_name" in str(excinfo.value)


# --- the code defaults ------------------------------------------------------


def test_default_presets_are_exactly_the_three_documented_ones():
    assert sorted(DEFAULT_PRESETS) == [
        "default-agents",
        "default-chat",
        "default-chat-plain",
    ]


def test_default_chat_presets_are_a_single_assistant():
    for name in ("default-chat", "default-chat-plain"):
        preset = DEFAULT_PRESETS[name]
        assert preset.orchestrator == "assistant"
        assert [c.id for c in preset.characters] == ["assistant"]
        assert preset.characters[0].display_name == "助教"
        # /chat's contract is that the server injects no persona of its own.
        assert preset.characters[0].prompt_name is None
        # Neither default forces retrieval: /chat either offers the tool or not.
        assert preset.rag_mode == "off"


def test_default_chat_can_search_the_textbook_and_has_the_steps_to_do_it():
    preset = DEFAULT_PRESETS["default-chat"]

    assert preset.characters[0].tools == ["rag_search"]
    # One step would leave the model able to call the tool but never answer
    # from it.
    assert preset.max_steps == 8


def test_default_chat_plain_has_no_tools_and_one_step():
    preset = DEFAULT_PRESETS["default-chat-plain"]

    assert preset.characters[0].tools == []
    assert preset.max_steps == 1


def test_default_agents_pairs_a_summoner_with_a_student():
    preset = DEFAULT_PRESETS["default-agents"]

    assert preset.max_steps == 8
    assert preset.orchestrator == "teacher"
    teacher, student = preset.characters
    assert (teacher.display_name, teacher.role) == ("老師", "teacher")
    # The orchestrator gets every registered tool.
    assert teacher.tools == sorted(registered_tool_names())
    assert teacher.prompt_name == "agents/teacher-system"
    assert (student.display_name, student.role) == ("學生", "student")
    assert student.tools == ["rag_search"]
    assert student.prompt_name == "agents/student"
    assert student.max_steps == 3


def test_every_default_preset_is_named_after_its_key():
    assert all(name == preset.name for name, preset in DEFAULT_PRESETS.items())


# --- cast -------------------------------------------------------------------


def test_build_cast_puts_the_orchestrator_first_and_names_the_summon_target():
    cast = build_cast(DEFAULT_PRESETS["default-agents"])

    assert list(cast.characters) == ["teacher", "student"]
    assert cast.orchestrator.id == "teacher"
    assert cast.summon_target_id == "student"
    assert cast.characters["student"].max_steps == 3
    assert cast.characters["teacher"].tool_names == ("rag_search", "summon_subagent")


def test_build_cast_of_a_solo_preset_has_no_summon_target():
    cast = build_cast(DEFAULT_PRESETS["default-chat-plain"])

    assert cast.summon_target_id is None
    assert list(cast.characters) == ["assistant"]


# --- registry ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_serves_builtins_before_any_fetch():
    registry = PresetRegistry(langfuse=_FakeLangfuse(), settings=_settings())

    assert registry.names() == sorted(DEFAULT_PRESETS)
    assert registry.load_errors == {}


@pytest.mark.asyncio
async def test_registry_loads_dataset_presets_alongside_the_builtins():
    langfuse = _FakeLangfuse([_item("i1", _preset(name="socratic"))])
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    preset = await registry.get("socratic")

    assert preset.name == "socratic"
    assert registry.names() == sorted([*DEFAULT_PRESETS, "socratic"])
    assert langfuse.dataset_calls == ["config/presets"]


@pytest.mark.asyncio
async def test_registry_reads_the_dataset_name_from_settings():
    langfuse = _FakeLangfuse()
    registry = PresetRegistry(
        langfuse=langfuse, settings=_settings(presets_dataset_name="config/other")
    )

    await registry.refresh()

    assert langfuse.dataset_calls == ["config/other"]


@pytest.mark.asyncio
async def test_registry_falls_back_to_the_default_dataset_name():
    # Settings predating the preset knobs must not break the registry.
    langfuse = _FakeLangfuse()
    registry = PresetRegistry(
        langfuse=langfuse, settings=SimpleNamespace(openai_default_model="m")
    )

    await registry.refresh()

    assert langfuse.dataset_calls == [DEFAULT_PRESETS_DATASET_NAME]


@pytest.mark.asyncio
async def test_a_dataset_item_shadows_a_builtin_of_the_same_name():
    langfuse = _FakeLangfuse(
        [_item("i1", _preset(name="default-chat", description="tuned in production"))]
    )
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    preset = await registry.get("default-chat")

    assert preset.description == "tuned in production"
    # Shadowing replaces, it does not add.
    assert registry.names() == sorted(DEFAULT_PRESETS)


@pytest.mark.asyncio
async def test_registry_parses_an_item_stored_as_a_json_string():
    import json

    langfuse = _FakeLangfuse([_item("i1", json.dumps(_preset(name="socratic")))])
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    assert (await registry.get("socratic")).name == "socratic"


@pytest.mark.asyncio
async def test_one_bad_item_does_not_cost_the_others_their_place():
    langfuse = _FakeLangfuse(
        [
            _item("bad-json", "{not json"),
            _item("bad-schema", _preset(name="broken", orchestrator="nobody")),
            _item("not-a-document", 42),
            _item("good", _preset(name="socratic")),
        ]
    )
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    report = await registry.refresh()

    assert (await registry.get("socratic")).name == "socratic"
    assert sorted(report.errors) == ["bad-json", "bad-schema", "not-a-document"]
    assert "not valid JSON" in report.errors["bad-json"]
    assert "is not one of the characters" in report.errors["bad-schema"]
    assert registry.load_errors == report.errors


@pytest.mark.asyncio
async def test_a_duplicate_name_keeps_the_first_item_and_reports_the_second():
    langfuse = _FakeLangfuse(
        [
            _item("first", _preset(name="socratic", description="winner")),
            _item("second", _preset(name="socratic", description="loser")),
        ]
    )
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    report = await registry.refresh()

    assert (await registry.get("socratic")).description == "winner"
    assert "duplicate preset name 'socratic'" in report.errors["second"]


@pytest.mark.asyncio
async def test_load_report_lists_everything_now_served_and_when():
    langfuse = _FakeLangfuse([_item("i1", _preset(name="socratic"))])
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    report = await registry.refresh()

    assert report.loaded == sorted([*DEFAULT_PRESETS, "socratic"])
    assert report.errors == {}
    assert report.fetched_at is not None


@pytest.mark.asyncio
async def test_get_raises_for_an_unknown_preset():
    registry = PresetRegistry(langfuse=_FakeLangfuse(), settings=_settings())

    with pytest.raises(PresetNotFoundError):
        await registry.get("nope")


@pytest.mark.asyncio
async def test_snapshot_is_a_copy():
    registry = PresetRegistry(langfuse=_FakeLangfuse(), settings=_settings())

    snapshot = registry.snapshot()
    snapshot.clear()

    assert registry.names() == sorted(DEFAULT_PRESETS)


@pytest.mark.asyncio
async def test_a_fresh_cache_is_not_refetched():
    langfuse = _FakeLangfuse()
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    await registry.get("default-chat-plain")
    await registry.get("default-chat-plain")

    assert len(langfuse.dataset_calls) == 1


@pytest.mark.asyncio
async def test_a_stale_cache_is_refetched():
    langfuse = _FakeLangfuse()
    registry = PresetRegistry(
        langfuse=langfuse, settings=_settings(presets_cache_ttl_seconds=0.0)
    )

    await registry.get("default-chat-plain")
    await registry.get("default-chat-plain")

    assert len(langfuse.dataset_calls) == 2


@pytest.mark.asyncio
async def test_concurrent_gets_share_a_single_fetch():
    langfuse = _SlowLangfuse([_item("i1", _preset(name="socratic"))])
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    presets = await asyncio.gather(registry.get("socratic"), registry.get("socratic"))

    assert [p.name for p in presets] == ["socratic", "socratic"]
    assert len(langfuse.dataset_calls) == 1


@pytest.mark.asyncio
async def test_a_failed_fetch_still_serves_the_builtins():
    langfuse = _FakeLangfuse(error=RuntimeError("langfuse is down"))
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())

    assert (await registry.get("default-chat-plain")).name == "default-chat-plain"
    assert "langfuse is down" in "".join(registry.load_errors.values())


@pytest.mark.asyncio
async def test_a_langfuse_without_get_dataset_degrades_to_builtins():
    registry = PresetRegistry(langfuse=SimpleNamespace(), settings=_settings())

    assert (await registry.get("default-chat-plain")).name == "default-chat-plain"
    assert registry.load_errors


@pytest.mark.asyncio
async def test_a_failed_fetch_keeps_the_presets_already_loaded():
    langfuse = _FakeLangfuse([_item("i1", _preset(name="socratic"))])
    registry = PresetRegistry(langfuse=langfuse, settings=_settings())
    await registry.refresh()

    langfuse.error = RuntimeError("langfuse is down")
    report = await registry.refresh()

    assert report.fetched_at is None
    assert report.loaded == sorted([*DEFAULT_PRESETS, "socratic"])
    assert (await registry.get("socratic")).name == "socratic"


@pytest.mark.asyncio
async def test_a_failed_fetch_backs_off_instead_of_retrying_per_request():
    langfuse = _FakeLangfuse(error=RuntimeError("langfuse is down"))
    registry = PresetRegistry(
        langfuse=langfuse, settings=_settings(presets_cache_ttl_seconds=0.0)
    )

    await registry.get("default-chat-plain")
    await registry.get("default-chat-plain")
    await registry.get("default-chat-plain")

    # Even with a zero TTL, the 30s failure backoff holds the retries off.
    assert len(langfuse.dataset_calls) == 1


# --- ensure_default_presets -------------------------------------------------


class _SeedableLangfuse(_FakeLangfuse):
    """A Langfuse whose preset dataset can be written to as well as read.

    Records every write, so "created the missing ones and touched nothing else"
    is directly assertable.
    """

    def __init__(self, items=(), *, dataset_exists: bool = True):
        super().__init__(items)
        self.dataset_exists = dataset_exists
        self.created_datasets: list[str] = []
        self.written: list[tuple[str, str, dict]] = []

    def get_dataset(self, name):
        self.dataset_calls.append(name)
        if self.error is not None:
            raise self.error
        if not self.dataset_exists:
            raise NotFoundError(body=f"no dataset '{name}'")
        return _FakeDataset(self.items)

    def create_dataset(self, *, name, description=None, **_kwargs):
        self.created_datasets.append(name)
        self.dataset_exists = True

    def create_dataset_item(self, *, dataset_name, input, id=None, **_kwargs):
        self.written.append((dataset_name, id, input))
        self.items.append(_item(id, input))


@pytest.mark.asyncio
async def test_ensure_default_presets_writes_every_default_into_an_empty_dataset():
    langfuse = _SeedableLangfuse()

    created = await ensure_default_presets(langfuse, _settings())

    assert created == list(DEFAULT_PRESETS)
    # Keyed by preset name, which is the id `/admin/presets` upserts to as well,
    # so the two write paths address the same item.
    assert [item_id for _dataset, item_id, _doc in langfuse.written] == list(
        DEFAULT_PRESETS
    )
    assert all(dataset == "config/presets" for dataset, _id, _doc in langfuse.written)
    # What was written is a servable document, not an approximation of one.
    for _dataset, item_id, document in langfuse.written:
        assert Preset.model_validate(document) == DEFAULT_PRESETS[item_id]


@pytest.mark.asyncio
async def test_ensure_default_presets_never_overwrites_an_existing_item():
    tuned = _preset(name="default-chat", description="tuned in production")
    langfuse = _SeedableLangfuse([_item("default-chat", tuned)])

    created = await ensure_default_presets(langfuse, _settings())

    assert "default-chat" not in created
    assert sorted(created) == ["default-agents", "default-chat-plain"]
    assert all(item_id != "default-chat" for _d, item_id, _doc in langfuse.written)


@pytest.mark.asyncio
async def test_ensure_default_presets_recognises_an_item_under_a_foreign_id():
    # Hand-created in the Langfuse UI: arbitrary item id, correct document name.
    langfuse = _SeedableLangfuse(
        [_item("cm-generated-id", _preset(name="default-agents"))]
    )

    created = await ensure_default_presets(langfuse, _settings())

    # Seeding it again would leave two items claiming the same preset name.
    assert "default-agents" not in created


@pytest.mark.asyncio
async def test_ensure_default_presets_is_a_no_op_once_everything_is_present():
    langfuse = _SeedableLangfuse(
        [_item(name, _preset(name=name)) for name in DEFAULT_PRESETS]
    )

    assert await ensure_default_presets(langfuse, _settings()) == []
    assert langfuse.written == []


@pytest.mark.asyncio
async def test_ensure_default_presets_creates_the_dataset_when_it_is_missing():
    langfuse = _SeedableLangfuse(dataset_exists=False)

    created = await ensure_default_presets(langfuse, _settings())

    assert langfuse.created_datasets == ["config/presets"]
    assert created == list(DEFAULT_PRESETS)


@pytest.mark.asyncio
async def test_ensure_default_presets_uses_the_dataset_name_from_settings():
    langfuse = _SeedableLangfuse()

    await ensure_default_presets(
        langfuse, _settings(presets_dataset_name="config/other")
    )

    assert langfuse.dataset_calls == ["config/other"]
    assert all(dataset == "config/other" for dataset, _id, _doc in langfuse.written)


@pytest.mark.asyncio
async def test_ensure_default_presets_reports_a_failure_to_its_caller():
    # Seeding does not swallow Langfuse failures itself — the lifespan does (see
    # the startup test below), because only the caller knows that the code
    # defaults are in service either way.
    langfuse = _SeedableLangfuse()
    langfuse.error = RuntimeError("langfuse is down")

    with pytest.raises(RuntimeError, match="langfuse is down"):
        await ensure_default_presets(langfuse, _settings())


@pytest.mark.asyncio
async def test_startup_survives_a_failed_seed(monkeypatch, caplog):
    """A Langfuse outage at boot must not stop the app from serving."""
    from app import main

    async def _explode(*_args, **_kwargs):
        raise RuntimeError("langfuse is down")

    async def _no_rag():
        return None

    async def _allowed():
        return ["gpt-oss-120b"]

    monkeypatch.setattr(main, "ensure_default_presets", _explode)
    monkeypatch.setattr(main, "validate_allowed_models", _allowed)
    monkeypatch.setattr(main, "build_rag_pipeline", _no_rag)
    monkeypatch.setattr(main, "get_langfuse_client", _FakeLangfuse)

    async def _openai():
        return SimpleNamespace()

    monkeypatch.setattr(main, "get_openai_client", _openai)
    monkeypatch.setattr(
        main, "EvalRunner", lambda *_args: SimpleNamespace(shutdown=lambda: None)
    )

    with caplog.at_level("ERROR"):
        async with main.lifespan(main.app):
            registry = main.app.state.preset_registry

    assert "Could not seed the default presets" in caplog.text
    # The defaults are in service regardless, which is what makes the failure
    # survivable.
    assert set(DEFAULT_PRESETS) <= set(registry.names())


# --- prepare_preset_run -----------------------------------------------------


_UNSET = object()


def _prepare(preset, *, settings=None, langfuse=None, rag_pipeline=_UNSET, **kwargs):
    return prepare_preset_run(
        preset=preset,
        settings=settings if settings is not None else _settings(),
        langfuse=langfuse if langfuse is not None else _FakeLangfuse(),
        # A sentinel, so a test can pass `rag_pipeline=None` to mean "RAG is not
        # configured on this server".
        rag_pipeline=_FakeRAGPipeline() if rag_pipeline is _UNSET else rag_pipeline,
        **kwargs,
    )


def test_prepare_falls_back_to_the_server_default_model():
    prepared = _prepare(DEFAULT_PRESETS["default-chat-plain"])

    assert prepared.model == "gpt-oss-120b"


def test_prepare_prefers_the_presets_own_model():
    preset = Preset.model_validate(_preset(model="custom-model"))

    assert _prepare(preset).model == "custom-model"


def test_prepare_lets_an_override_win_over_the_preset():
    preset = Preset.model_validate(_preset(model="custom-model"))

    assert _prepare(preset, model_override="other-model").model == "other-model"


def test_prepare_rejects_a_model_outside_the_allow_list_as_a_misconfiguration():
    preset = Preset.model_validate(_preset(model="banned-model"))

    with pytest.raises(PresetRunError) as excinfo:
        _prepare(preset, settings=_settings(allowed_model_names=["gpt-oss-120b"]))

    assert excinfo.value.status_code == 503
    assert "is not in the allowed models list" in excinfo.value.detail


def test_prepare_allows_any_model_when_no_allow_list_is_configured():
    preset = Preset.model_validate(_preset(model="anything"))

    assert _prepare(preset).model == "anything"


def test_prepare_resolves_the_orchestrators_tools_only():
    prepared = _prepare(DEFAULT_PRESETS["default-agents"])

    assert [tool.name for tool in prepared.tools] == ["rag_search", "summon_subagent"]
    assert prepared.cast.summon_target_id == "student"


def test_prepare_compiles_the_orchestrator_prompt_into_a_system_message():
    langfuse = _FakeLangfuse()

    prepared = _prepare(DEFAULT_PRESETS["default-agents"], langfuse=langfuse)

    assert langfuse.prompt_calls == ["agents/teacher-system"]
    assert prepared.system_message == {
        "role": "system",
        "content": "PROMPT<agents/teacher-system>",
    }


def test_prepare_leaves_the_system_message_unset_without_a_prompt_name():
    prepared = _prepare(DEFAULT_PRESETS["default-chat-plain"])

    assert prepared.system_message is None


def test_prepare_reports_a_failed_prompt_load_as_a_bad_gateway():
    langfuse = _FakeLangfuse()
    langfuse.prompt_error = RuntimeError("no such prompt")

    with pytest.raises(PresetRunError) as excinfo:
        _prepare(DEFAULT_PRESETS["default-agents"], langfuse=langfuse)

    assert excinfo.value.status_code == 502
    assert "Failed to load prompt 'agents/teacher-system'" in excinfo.value.detail


def test_prepare_rejects_a_rag_tool_when_rag_is_not_configured():
    with pytest.raises(PresetRunError) as excinfo:
        _prepare(DEFAULT_PRESETS["default-agents"], rag_pipeline=None)

    assert excinfo.value.status_code == 503
    assert "Tool 'rag_search' requires RAG" in excinfo.value.detail


def test_prepare_rejects_forced_rag_when_rag_is_not_configured():
    with pytest.raises(PresetRunError) as excinfo:
        _prepare(_forced_rag_preset(), rag_pipeline=None)

    assert excinfo.value.status_code == 503
    assert "Preset 'forced-rag' requires RAG" in excinfo.value.detail


def test_prepare_allows_a_toolless_preset_without_rag():
    assert (
        _prepare(DEFAULT_PRESETS["default-chat-plain"], rag_pipeline=None).tools == []
    )


# --- run_preset -------------------------------------------------------------


class _RecordingRunAgents:
    """Stands in for ``run_agents`` so message shape can be asserted directly."""

    def __init__(self):
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs

        async def _events():
            yield DoneEvent(finish_reason="stop")

        return _events()


@pytest.fixture
def recorded_run_agents(monkeypatch):
    recorder = _RecordingRunAgents()
    monkeypatch.setattr("app.agents.run.run_agents", recorder)
    return recorder


async def _collect(prepared, messages, *, rag_pipeline=None, langfuse=None):
    return [
        event
        async for event in run_preset(
            prepared=prepared,
            messages=messages,
            openai=SimpleNamespace(),
            langfuse=langfuse if langfuse is not None else _FakeLangfuse(),
            settings=_settings(),
            rag_pipeline=rag_pipeline,
        )
    ]


@pytest.mark.asyncio
async def test_run_preset_hands_the_engine_the_casts_own_arguments(
    recorded_run_agents,
):
    prepared = _prepare(DEFAULT_PRESETS["default-agents"])

    events = await _collect(
        prepared, [{"role": "user", "content": "Hi"}], rag_pipeline=_FakeRAGPipeline()
    )

    assert events == [DoneEvent(finish_reason="stop")]
    kwargs = recorded_run_agents.kwargs
    assert kwargs["orchestrator"].id == "teacher"
    assert list(kwargs["characters"]) == ["teacher", "student"]
    assert kwargs["summon_target_id"] == "student"
    assert kwargs["max_steps"] == 8
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["model"] == "gpt-oss-120b"
    assert [tool.name for tool in kwargs["tools"]] == ["rag_search", "summon_subagent"]


@pytest.mark.asyncio
async def test_run_preset_prepends_the_orchestrator_system_prompt(
    recorded_run_agents,
):
    prepared = _prepare(DEFAULT_PRESETS["default-agents"])

    await _collect(
        prepared, [{"role": "user", "content": "Hi"}], rag_pipeline=_FakeRAGPipeline()
    )

    assert recorded_run_agents.kwargs["messages"] == [
        {"role": "system", "content": "PROMPT<agents/teacher-system>"},
        {"role": "user", "content": "Hi"},
    ]


@pytest.mark.asyncio
async def test_run_preset_sends_the_history_untouched_without_a_prompt(
    recorded_run_agents,
):
    prepared = _prepare(DEFAULT_PRESETS["default-chat-plain"])
    messages = [{"role": "user", "content": "Hi"}]

    await _collect(prepared, messages)

    assert recorded_run_agents.kwargs["messages"] == messages


@pytest.mark.asyncio
async def test_forced_rag_swaps_the_latest_user_turn_and_keeps_the_history(
    recorded_run_agents,
):
    pipeline = _FakeRAGPipeline()
    prepared = _prepare(_forced_rag_preset(), rag_pipeline=pipeline)
    messages = [
        {"role": "user", "content": "第一個問題"},
        {"role": "assistant", "content": "先前的回答"},
        {"role": "user", "content": "光合作用是什麼"},
    ]

    await _collect(prepared, messages, rag_pipeline=pipeline)

    assert pipeline.retrieve_calls == ["光合作用是什麼"]
    assert recorded_run_agents.kwargs["messages"] == [
        {"role": "system", "content": "RAG SYSTEM<CTX for 光合作用是什麼>"},
        {"role": "user", "content": "第一個問題"},
        {"role": "assistant", "content": "先前的回答"},
        {"role": "user", "content": "RAG USER<光合作用是什麼>"},
    ]
    # The caller's list is not mutated.
    assert messages[2] == {"role": "user", "content": "光合作用是什麼"}


@pytest.mark.asyncio
async def test_forced_rag_ends_the_stream_when_retrieval_fails(recorded_run_agents):
    prepared = _prepare(_forced_rag_preset())

    events = await _collect(
        prepared,
        [{"role": "user", "content": "Hi"}],
        rag_pipeline=_FailingRAGPipeline(),
    )

    assert events == [
        ErrorEvent(error=errors.RAG_FAILED_MESSAGE, code=errors.RAG_FAILED)
    ]
    # The upstream detail never reaches the client.
    assert "secret.internal" not in events[0].error
    assert recorded_run_agents.kwargs is None


@pytest.mark.asyncio
async def test_forced_rag_ends_the_stream_without_a_user_message(recorded_run_agents):
    prepared = _prepare(_forced_rag_preset())

    events = await _collect(
        prepared,
        [{"role": "assistant", "content": "說了什麼"}],
        rag_pipeline=_FakeRAGPipeline(),
    )

    assert events == [
        ErrorEvent(error=errors.RAG_FAILED_MESSAGE, code=errors.RAG_FAILED)
    ]
    assert recorded_run_agents.kwargs is None


# --- latest_user_message ----------------------------------------------------


def test_latest_user_message_finds_the_last_non_empty_user_turn():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]

    assert latest_user_message(messages) == (2, "second")


def test_latest_user_message_skips_blank_and_non_string_content():
    messages = [
        {"role": "user", "content": "real"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]},
    ]

    assert latest_user_message(messages) == (0, "real")


def test_latest_user_message_returns_none_without_a_user_turn():
    assert latest_user_message([{"role": "system", "content": "x"}]) is None
