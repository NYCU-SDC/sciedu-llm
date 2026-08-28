"""Assumptions this service makes about Langfuse, checked against a real one.

The rest of the suite fakes Langfuse, so it can only prove that our code is
consistent with our idea of the SDK. These tests check that idea: that a preset
stored as a dataset item comes back as a servable preset, that a text prompt
compiles to a `str` and a chat prompt to a list of messages (which is exactly
what `prepare_preset_run` and `_compile_subagent_messages` assume), and that the
dataset-runs listing behind `/admin/evals/history` still exists.

They are opt-in, because they talk to — and briefly write to — a real project:

    uv run pytest -m live tests/live

`poe test` deselects them (see `addopts` in pyproject.toml), and every test here
skips as a body if the credentials are missing. Everything written here lives under a
`citest-…` / `citest/…` name unique to the run and is deleted again in the same
test, so two runs can overlap and a crashed run leaves at most one obviously
disposable item behind. Nothing that a real preset or prompt is named is ever
read-modify-written.
"""

import os
import urllib.parse
import uuid
import warnings
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from langfuse.api.commons.errors.not_found_error import NotFoundError
from langfuse.api.core.api_error import ApiError

from app.listings import list_experiment_runs
from app.presets import (
    DEFAULT_PRESETS,
    DEFAULT_PRESETS_DATASET_NAME,
    PRESETS_DATASET_DESCRIPTION,
    Preset,
    PresetCharacter,
    PresetNotFoundError,
    PresetRegistry,
)
from observability import init_langfuse_client

# The mark is what makes these opt-in; it is applied to the module rather than
# to each test so a new test here cannot forget it.
pytestmark = pytest.mark.live

DATASET_NAME = DEFAULT_PRESETS_DATASET_NAME
DATASET_DESCRIPTION = PRESETS_DATASET_DESCRIPTION

#: Every object these tests create is named with this prefix, so anything left
#: over after a crash is recognisable at a glance and safe to delete.
NAMESPACE = "citest"


def throwaway_name(separator: str = "-") -> str:
    return f"{NAMESPACE}{separator}{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def langfuse():
    """The real client, or a skip.

    The credential check is here rather than at import time on purpose: at
    import time it would report a skip on every ordinary `pytest` run, where
    these tests are deselected and nobody asked for them. `load_dotenv` because
    the SDK reads its credentials from the environment and a developer's are
    usually in `.env` — nothing in this module imports `app.main`, which is what
    normally loads it.
    """
    load_dotenv()
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")

    client = init_langfuse_client()
    yield client
    client.flush()


def registry_settings():
    """Just the two attributes `PresetRegistry` reads off `Settings`."""
    return SimpleNamespace(
        presets_dataset_name=DATASET_NAME,
        # Long enough that only the explicit `refresh()` calls below fetch.
        presets_cache_ttl_seconds=300.0,
    )


def ensure_dataset(langfuse) -> None:
    try:
        langfuse.get_dataset(DATASET_NAME)
    except NotFoundError:
        langfuse.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)


def delete_prompt(langfuse, name: str) -> None:
    """Remove every version of a throwaway prompt.

    There *is* a delete surface in 4.5.1 — `api.prompts.delete`, which drops all
    versions when given neither `version` nor `label`. What it does not do is
    percent-encode the name into the path: the high-level `Langfuse.get_prompt`
    runs the name through `_url_encode` first, but the generated `api.*` clients
    interpolate it raw, so a foldered name like `agents/student` addresses
    `.../v2/prompts/agents/student` and collects the web app's HTML 404. Hence
    the quoting here.
    """
    try:
        langfuse.api.prompts.delete(prompt_name=urllib.parse.quote(name, safe=""))
    except Exception as e:  # noqa: BLE001 - cleanup is best-effort by design
        warnings.warn(
            f"could not delete the throwaway prompt {name!r}: {e!r}", stacklevel=2
        )


@pytest.mark.asyncio
async def test_preset_dataset_round_trip(langfuse):
    """A dataset item written today is a servable preset a refresh later."""
    name = throwaway_name()
    document = Preset(
        name=name,
        description="Throwaway preset written by the live smoke test.",
        max_steps=1,
        orchestrator="assistant",
        characters=[PresetCharacter(id="assistant", display_name="CI")],
    ).model_dump()

    ensure_dataset(langfuse)
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        id=name,
        input=document,
        metadata={"source": "tests/live/langfuse_smoke_test.py"},
    )

    registry = PresetRegistry(langfuse=langfuse, settings=registry_settings())
    try:
        report = await registry.refresh()
        assert name in report.loaded
        assert name not in report.errors
        # The code defaults are unaffected by whatever the dataset holds.
        assert set(DEFAULT_PRESETS) <= set(report.loaded)

        served = await registry.get(name)
        assert served.name == name
        assert served.model is None
        assert [character.id for character in served.characters] == ["assistant"]
    finally:
        langfuse.api.dataset_items.delete(id=name)

    after = await registry.refresh()
    assert name not in after.loaded
    with pytest.raises(PresetNotFoundError):
        await registry.get(name)


def test_text_prompt_compiles_to_a_string(langfuse):
    """What an orchestrator's `prompt_name` resolves to (see `prepare_preset_run`)."""
    name = throwaway_name("/")
    body = "You are a throwaway prompt created by the live smoke test."
    langfuse.create_prompt(
        name=name, type="text", prompt=body, labels=["production"], tags=[NAMESPACE]
    )
    try:
        # `cache_ttl_seconds=0` because the SDK caches prompt fetches for a
        # minute by default, and this one was created a moment ago.
        prompt = langfuse.get_prompt(name, cache_ttl_seconds=0)
        compiled = prompt.compile()
        assert isinstance(compiled, str)
        assert compiled == body
    finally:
        delete_prompt(langfuse, name)


def test_chat_prompt_compiles_to_messages(langfuse):
    """What a summoned character's `prompt_name` resolves to.

    `_compile_subagent_messages` hands the result straight to the upstream
    chat-completions call, so it has to be a list of role/content dicts with the
    `{{task}}` substitution already applied.
    """
    name = throwaway_name("/")
    langfuse.create_prompt(
        name=name,
        type="chat",
        prompt=[
            {"role": "system", "content": "You are a throwaway student."},
            {"role": "user", "content": "{{task}}"},
        ],
        labels=["production"],
        tags=[NAMESPACE],
    )
    try:
        prompt = langfuse.get_prompt(name, type="chat", cache_ttl_seconds=0)
        messages = prompt.compile(task="explain the light reactions")

        assert isinstance(messages, list)
        assert all(
            isinstance(message, dict) and {"role", "content"} <= set(message)
            for message in messages
        )
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[-1]["content"] == "explain the light reactions"
    finally:
        delete_prompt(langfuse, name)


def _is_events_only(body: str) -> bool:
    """Whether a runs-API failure is "this deployment does not serve it"."""
    lowered = body.lower()
    return "events_only" in lowered or "events only" in lowered


@pytest.mark.asyncio
async def test_dataset_runs_listing_still_exists(langfuse):
    """Read-only: pins the SDK surface `/admin/evals/history` is built on.

    It calls the production helper rather than the SDK directly, because the two
    things that can break this are both in the gap between them: the raw
    `api.datasets.get_runs` does not percent-encode the dataset name (every
    dataset here is foldered, e.g. `questions/biology7-9trim`) — which
    `list_experiment_runs` now does for it — and the runs endpoint itself is not
    served by a Langfuse deployment in v4 `events_only` mode. The first is our
    bug and fails this test; the second is the deployment's shape and skips it.
    """
    datasets = langfuse.api.datasets.list(page=1, limit=1)
    if not datasets.data:
        pytest.skip("this Langfuse project has no datasets to list runs for")

    name = datasets.data[0].name
    failure: ApiError | None = None
    try:
        runs = await list_experiment_runs(langfuse, name)
    except ApiError as e:
        failure = e
    # Reported from outside the handler, and without a traceback: the SDK's
    # `ApiError` repr carries the entire 404 page.
    if failure is not None:
        body = str(failure.body)
        if _is_events_only(body):
            # Not a bug on our side and not fixable from here: this deployment
            # simply does not serve the runs API. `/admin/evals/history` answers
            # 502 against it, and the endpoint stays because other deployments
            # do serve it.
            pytest.skip(
                "this Langfuse deployment runs in v4 events_only mode, where the "
                f"dataset-runs API is unavailable (status {failure.status_code})"
            )
        pytest.fail(
            f"GET /admin/evals/history cannot list runs for dataset {name!r}: "
            f"status {failure.status_code}, body {body[:120]!r}",
            pytrace=False,
        )

    assert isinstance(runs, list)
    for run in runs:
        assert run.name
        assert run.created_at
