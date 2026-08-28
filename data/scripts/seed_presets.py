"""Seed Langfuse with the preset dataset and the prompts the presets reference.

The app seeds its *own* defaults now: `app.presets.ensure_default_presets` runs
at startup and writes any of `default-agents` / `default-chat` /
`default-chat-plain` that the dataset does not already hold. What this script is
still for is everything startup cannot do — creating the **prompts** those
presets reference (a preset naming a prompt Langfuse does not have fails at run
time with a 502), and seeding the **example presets** kept here as starting
points for authoring (`teacher-student`, `rag-agent`), which the server does not
ship as defaults. The default documents under `data/presets/` are checked in as
the readable copy of what startup writes, and re-seeding them is harmless.

Two kinds of file live under `data/presets/`:

* `data/presets/*.json` — one **preset document** each, exactly the schema
  `app.presets.Preset` validates. Every file is validated *before* anything is
  uploaded, so a typo in one preset aborts the run instead of half-seeding the
  dataset. Each becomes one item of the Langfuse dataset `config/presets`, with
  the item id set to the preset name (that is the id the `/admin/presets` API
  writes to as well, so the two paths upsert the same item).

* `data/presets/prompts/**/*.json` — **prompt descriptors**, the Langfuse
  prompts the presets name in `prompt_name`. The descriptor schema is:

      {
        "name":   "agents/teacher-system",   # canonical Langfuse prompt name
        "type":   "text" | "chat",           # text -> str, chat -> messages
        "labels": ["production"],            # "production" = the served version
        "prompt": "..."                      # str for text; for chat, a list of
                                             # {"role": ..., "content": ...}
      }

  Variables use Langfuse's `{{name}}` syntax. `agents/student` is a *chat*
  prompt taking `{{task}}` because that is how a summoned character is
  prompted; orchestrator prompts are *text* prompts compiled with no variables.

Re-running is safe and conservative: presets and prompts that already exist are
skipped with a warning, because a preset may well have been tuned in production
through `PUT /admin/presets/{name}` since it was seeded. `--overwrite` upserts
the dataset items and pushes a new version of each prompt. `--dry-run`
validates every file and reports what it would do without contacting Langfuse
(useful in CI, and it needs no credentials).

Usage:
    uv run python data/scripts/seed_presets.py [--overwrite] [--dry-run]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from _common import (  # noqa: E402
    DATA_ROOT,
    configure_logging,
    dataset_exists,
    get_langfuse_client,
    retry_on_transport_error,
)
from langfuse.api.commons.errors.not_found_error import NotFoundError
from pydantic import ValidationError

from app.presets import DEFAULT_PRESETS_DATASET_NAME, Preset

PRESETS_ROOT = DATA_ROOT / "presets"
PROMPTS_ROOT = PRESETS_ROOT / "prompts"

DATASET_NAME = DEFAULT_PRESETS_DATASET_NAME
DATASET_DESCRIPTION = (
    "Run configurations served by /agents. One item per preset; the item id is "
    "the preset name and the item input is the preset document."
)

PRODUCTION_LABEL = "production"


class SeedError(RuntimeError):
    """A file is malformed. Reported with its path and aborts the run."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Langfuse preset dataset and prompts from data/presets/"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace presets that already exist in the dataset and push a new "
            "version of prompts that already exist. Without it, existing "
            "presets and prompts are left alone."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every file and report the plan without contacting Langfuse.",
    )
    return parser.parse_args()


# --- loading ----------------------------------------------------------------


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SeedError(f"{path}: not valid JSON: {e}") from e


def load_presets() -> list[tuple[Path, Preset]]:
    """Validate every preset file, or raise on the first bad one."""
    presets: list[tuple[Path, Preset]] = []
    seen: dict[str, Path] = {}
    for path in sorted(PRESETS_ROOT.glob("*.json")):
        document = _load_json(path)
        if not isinstance(document, dict):
            raise SeedError(f"{path}: expected a preset object, got {type(document)}")
        try:
            preset = Preset.model_validate(document)
        except ValidationError as e:
            raise SeedError(f"{path}: invalid preset document:\n{e}") from e
        if preset.name in seen:
            raise SeedError(
                f"{path}: preset name '{preset.name}' is already defined by "
                f"{seen[preset.name]}"
            )
        seen[preset.name] = path
        presets.append((path, preset))
    return presets


def load_prompts() -> list[tuple[Path, dict]]:
    """Validate every prompt descriptor, or raise on the first bad one."""
    prompts: list[tuple[Path, dict]] = []
    seen: dict[str, Path] = {}
    for path in sorted(PROMPTS_ROOT.rglob("*.json")):
        descriptor = _load_json(path)
        if not isinstance(descriptor, dict):
            raise SeedError(f"{path}: expected a prompt descriptor object")

        name = descriptor.get("name")
        if not isinstance(name, str) or not name:
            raise SeedError(f"{path}: descriptor needs a non-empty 'name'")
        prompt_type = descriptor.get("type", "text")
        if prompt_type not in ("text", "chat"):
            raise SeedError(
                f"{path}: 'type' must be 'text' or 'chat', got {prompt_type!r}"
            )
        body = descriptor.get("prompt")
        if prompt_type == "text" and not isinstance(body, str):
            raise SeedError(f"{path}: a text prompt's 'prompt' must be a string")
        if prompt_type == "chat":
            if not isinstance(body, list) or not body:
                raise SeedError(
                    f"{path}: a chat prompt's 'prompt' must be a non-empty list of messages"
                )
            for message in body:
                if not isinstance(message, dict) or "role" not in message:
                    raise SeedError(f"{path}: each chat message needs a 'role'")
        labels = descriptor.get("labels", [PRODUCTION_LABEL])
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise SeedError(f"{path}: 'labels' must be a list of strings")
        descriptor["labels"] = labels
        descriptor["type"] = prompt_type

        if name in seen:
            raise SeedError(
                f"{path}: prompt '{name}' is already defined by {seen[name]}"
            )
        seen[name] = path
        prompts.append((path, descriptor))
    return prompts


def check_references(
    presets: list[tuple[Path, Preset]],
    prompts: list[tuple[Path, dict]],
    logger: logging.Logger,
) -> None:
    """Warn about prompts a preset names but no descriptor here provides.

    Not fatal: the prompt may have been authored directly in Langfuse. It is
    still worth saying out loud, because a missing prompt only fails at run
    time, as a 502 from the agentic endpoint.
    """
    described = {descriptor["name"] for _path, descriptor in prompts}
    for path, preset in presets:
        for character in preset.characters:
            if character.prompt_name and character.prompt_name not in described:
                logger.warning(
                    "%s: preset '%s' references prompt '%s', which has no "
                    "descriptor under %s — it must already exist in Langfuse.",
                    path.name,
                    preset.name,
                    character.prompt_name,
                    PROMPTS_ROOT,
                )


# --- uploading --------------------------------------------------------------


def prompt_exists(client, name: str, prompt_type: str) -> bool:
    try:
        # Bypass the SDK's prompt cache: a stale hit would make an existing
        # prompt look present after it was deleted, and vice versa.
        client.get_prompt(name, type=prompt_type, cache_ttl_seconds=0)
    except NotFoundError:
        return False
    return True


def existing_preset_ids(client) -> set[str]:
    if not dataset_exists(client, DATASET_NAME):
        return set()
    return {str(item.id) for item in client.get_dataset(DATASET_NAME).items}


def seed_prompts(client, prompts, overwrite: bool, logger: logging.Logger) -> None:
    for path, descriptor in prompts:
        name = descriptor["name"]
        prompt_type = descriptor["type"]
        if prompt_exists(client, name, prompt_type):
            if not overwrite:
                logger.warning(
                    "Prompt '%s' already exists on Langfuse, skipping. Pass "
                    "--overwrite to push a new version.",
                    name,
                )
                continue
            logger.info("Pushing a new version of prompt '%s'", name)
        retry_on_transport_error(
            lambda descriptor=descriptor: client.create_prompt(
                name=descriptor["name"],
                type=descriptor["type"],
                prompt=descriptor["prompt"],
                labels=descriptor["labels"],
            ),
            logger=logger,
        )
        logger.info("Seeded %s prompt '%s' from %s", prompt_type, name, path.name)


def seed_presets(client, presets, overwrite: bool, logger: logging.Logger) -> None:
    if not dataset_exists(client, DATASET_NAME):
        client.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)
        logger.info("Created dataset '%s'", DATASET_NAME)
        existing: set[str] = set()
    else:
        existing = existing_preset_ids(client)

    for path, preset in presets:
        if preset.name in existing and not overwrite:
            logger.warning(
                "Preset '%s' already exists in '%s', skipping. Pass --overwrite "
                "to replace it (this discards any edit made via /admin/presets).",
                preset.name,
                DATASET_NAME,
            )
            continue
        document = preset.model_dump()
        retry_on_transport_error(
            lambda preset=preset,
            document=document,
            path=path: client.create_dataset_item(
                id=preset.name,
                dataset_name=DATASET_NAME,
                input=document,
                metadata={"source": path.name},
            ),
            logger=logger,
        )
        logger.info("Seeded preset '%s' from %s", preset.name, path.name)


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    if not PRESETS_ROOT.is_dir():
        logger.error("No preset directory at %s", PRESETS_ROOT)
        return

    presets = load_presets()
    prompts = load_prompts()
    if not presets:
        logger.warning("No preset files found under %s", PRESETS_ROOT)
        return
    check_references(presets, prompts, logger)
    logger.info(
        "Validated %d preset(s) and %d prompt descriptor(s)", len(presets), len(prompts)
    )

    if args.dry_run:
        for _path, preset in presets:
            logger.info("Would upsert preset '%s' into '%s'", preset.name, DATASET_NAME)
        for _path, descriptor in prompts:
            logger.info(
                "Would create %s prompt '%s'", descriptor["type"], descriptor["name"]
            )
        return

    client = get_langfuse_client()
    seed_prompts(client, prompts, args.overwrite, logger)
    seed_presets(client, presets, args.overwrite, logger)
    client.flush()


if __name__ == "__main__":
    main()
