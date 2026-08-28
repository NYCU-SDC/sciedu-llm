"""Turning a validated preset into the cast the engine runs.

This is the whole seam between config and the loop: ``Preset`` is the thing
operators edit, ``Cast`` is the thing ``run_agents`` takes. Keeping the
translation here means the engine never imports ``app.presets`` and stays
testable with hand-built ``Character`` objects.
"""

from dataclasses import dataclass

from app.agents.events import Character
from app.presets import Preset


@dataclass(frozen=True)
class Cast:
    """Who can speak in one run, and who the summon tool reaches.

    ``characters`` includes the orchestrator and is ordered orchestrator-first,
    which is the order the ``cast`` event presents to the frontend.
    """

    orchestrator: Character
    characters: dict[str, Character]
    # The single non-orchestrator character, if the preset has one. `None` for a
    # solo preset, in which case `summon_subagent` is not in the cast's tools
    # either (preset validation guarantees that pairing).
    summon_target_id: str | None


def build_cast(preset: Preset) -> Cast:
    built = {
        character.id: Character(
            id=character.id,
            display_name=character.display_name,
            role=character.role,
            tool_names=tuple(character.tools),
            prompt_name=character.prompt_name,
            max_steps=character.max_steps,
        )
        for character in preset.characters
    }

    # Validation guarantees the orchestrator is one of the characters.
    orchestrator = built[preset.orchestrator]
    others = [id_ for id_ in built if id_ != orchestrator.id]

    characters = {orchestrator.id: orchestrator}
    for id_ in others:
        characters[id_] = built[id_]

    return Cast(
        orchestrator=orchestrator,
        characters=characters,
        # At most one, since a preset carries at most two characters.
        summon_target_id=others[0] if others else None,
    )
