"""Typed parts and SSE events for the agentic protocol.

The wire format is defined in ``docs/agents-spec.md``: an assistant message is an
ordered list of typed ``parts`` (text / reasoning / tool_call / tool_result), and
the stream carries typed events that reference those parts by ``index``.

These are plain dataclasses rather than pydantic models because they are internal
to the engine — the router serializes them with ``payload()`` and never validates
them. Every ``payload()`` drops ``None`` fields so the frontend only ever sees
keys that apply to the part type at hand.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

PartType = Literal["text", "reasoning", "tool_call", "tool_result"]
ToolResultStatus = Literal["ok", "error"]

# The conventional `agent` id for a single-character run, and the default
# `orchestrator` of a preset. Nothing in the engine keys off it any more — a
# preset names its own orchestrator — but it stays the id the built-in
# single-character presets use, which is what the spec prescribes for a run with
# no cast.
ORCHESTRATOR_AGENT_ID = "assistant"

# Part types whose content arrives incrementally via `delta` events. `part_start`
# omits the streaming field for these; `part_end` carries the complete value.
_STREAMING_PART_TYPES: frozenset[str] = frozenset({"text", "reasoning", "tool_call"})


@dataclass
class Part:
    """One typed step within a single assistant message.

    Mutable because the streaming fields (``text`` / ``raw_arguments``) are
    accumulated as deltas arrive. ``raw_arguments`` is engine bookkeeping — it
    holds the JSON fragments emitted by the model so the exact original string can
    be replayed back in the assistant message on the next step — and is never
    serialized to the client.
    """

    type: PartType
    id: str
    agent: str
    internal: bool = False

    # text / reasoning
    text: str | None = None

    # tool_call / tool_result
    tool_call_id: str | None = None

    # tool_call
    name: str | None = None
    arguments: dict[str, Any] | None = None
    raw_arguments: str = ""

    # tool_result
    status: ToolResultStatus | None = None
    code: str | None = None
    content: str | None = None

    @property
    def is_streaming(self) -> bool:
        return self.type in _STREAMING_PART_TYPES

    def start_payload(self) -> dict[str, Any]:
        """Serialize for ``part_start`` — content fields are not filled in yet."""
        payload = self.end_payload()
        for key in ("text", "arguments", "status", "code", "content"):
            payload.pop(key, None)
        return payload

    def end_payload(self) -> dict[str, Any]:
        """Serialize the complete part for ``part_end`` / the non-streaming fold."""
        payload: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "agent": self.agent,
        }
        if self.internal:
            payload["internal"] = True
        for key in ("text", "tool_call_id", "name", "arguments", "status", "code"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.content is not None:
            payload["content"] = self.content
        return payload


@dataclass(frozen=True)
class CharacterInfo:
    """A speaker the frontend needs an avatar / name tag for."""

    id: str
    display_name: str
    role: str

    def payload(self) -> dict[str, Any]:
        return {"id": self.id, "displayName": self.display_name, "role": self.role}


@dataclass(frozen=True)
class CastEvent:
    characters: tuple[CharacterInfo, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "type": "cast",
            "characters": [c.payload() for c in self.characters],
        }


@dataclass(frozen=True)
class AgentStartEvent:
    agent: str
    parent: str | None = None
    summoned_by: str | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "agent_start", "agent": self.agent}
        if self.parent is not None:
            payload["parent"] = self.parent
        if self.summoned_by is not None:
            payload["summonedBy"] = self.summoned_by
        return payload


@dataclass(frozen=True)
class AgentEndEvent:
    agent: str

    def payload(self) -> dict[str, Any]:
        return {"type": "agent_end", "agent": self.agent}


@dataclass(frozen=True)
class PartStartEvent:
    index: int
    part: Part

    def payload(self) -> dict[str, Any]:
        return {
            "type": "part_start",
            "index": self.index,
            "part": self.part.start_payload(),
        }


@dataclass(frozen=True)
class DeltaEvent:
    index: int
    delta: str

    def payload(self) -> dict[str, Any]:
        return {"type": "delta", "index": self.index, "delta": self.delta}


@dataclass(frozen=True)
class PartEndEvent:
    index: int
    part: Part

    def payload(self) -> dict[str, Any]:
        return {
            "type": "part_end",
            "index": self.index,
            "part": self.part.end_payload(),
        }


@dataclass(frozen=True)
class DoneEvent:
    finish_reason: str | None = None
    status: str = "completed"

    def payload(self) -> dict[str, Any]:
        return {
            "type": "done",
            "finishReason": self.finish_reason,
            "status": self.status,
        }


@dataclass(frozen=True)
class ErrorEvent:
    error: str
    code: str

    def payload(self) -> dict[str, Any]:
        return {"type": "error", "error": self.error, "code": self.code}


Event = (
    CastEvent
    | AgentStartEvent
    | AgentEndEvent
    | PartStartEvent
    | DeltaEvent
    | PartEndEvent
    | DoneEvent
    | ErrorEvent
)


@dataclass(frozen=True)
class Character:
    """A role that can speak in a run.

    Built from a preset's ``characters`` by ``app.agents.cast.build_cast``; the
    engine only ever sees this, so which characters exist is a config decision
    rather than a code one.

    ``max_steps`` is the budget this character gets *when it is summoned*. The
    orchestrator's budget comes from the preset instead, so its value here is
    ignored.
    """

    id: str
    display_name: str
    role: str
    tool_names: tuple[str, ...] = field(default_factory=tuple)
    prompt_name: str | None = None
    max_steps: int = 3

    def info(self) -> CharacterInfo:
        return CharacterInfo(id=self.id, display_name=self.display_name, role=self.role)
