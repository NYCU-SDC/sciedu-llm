"""Speaking the old ``/chat`` wire format from a typed event stream.

``/chat`` predates the part protocol and has clients in the field, so its frames
are a hard contract: ``{"delta": ..., "isFinished": ...}``, one JSON object per
``data:`` line, ``json.dumps`` defaults (so non-ASCII is ``\\uXXXX``-escaped —
``/agents`` deliberately does *not* escape, but changing ``/chat`` would change
every byte a shipped client already parses).

The translation is lossy on purpose. The old format has room for exactly one
thing — the assistant's visible text — so reasoning, tool calls, tool results,
speaker changes and the cast are all dropped. What survives is the text an old
client would have received from a plain completion, which is the whole point:
``/chat`` is now a preset run wearing its old clothes.
"""

import json
from collections.abc import AsyncIterator

from app.agents import errors
from app.agents.events import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    Part,
    PartEndEvent,
    PartStartEvent,
)

# Terminal error codes the legacy format has a fixed message for. Anything else
# falls back to the event's own message.
_LEGACY_ERROR_MESSAGES: dict[str, str] = {
    errors.UPSTREAM_ERROR: errors.UPSTREAM_ERROR_MESSAGE,
    errors.RAG_FAILED: errors.RAG_FAILED_MESSAGE,
}


def _frame(payload: dict[str, object]) -> str:
    # Plain `json.dumps`, i.e. `ensure_ascii=True`: byte parity with the old
    # endpoint matters more than frame size here.
    return f"data: {json.dumps(payload)}\n\n"


def _is_visible_text(part: Part | None) -> bool:
    return part is not None and part.type == "text" and not part.internal


async def legacy_frames(
    events: AsyncIterator[Event], *, sink: list[str] | None = None
) -> AsyncIterator[str]:
    """Render a typed event stream as legacy ``/chat`` SSE frames.

    ``sink``, when given, collects every delta that made it into a frame, so the
    caller can record the real (possibly partial) output on its Langfuse span
    without re-parsing its own frames.
    """
    parts: dict[int, Part] = {}
    async for event in events:
        if isinstance(event, PartStartEvent):
            parts[event.index] = event.part
        elif isinstance(event, DeltaEvent):
            # Reasoning and tool-call argument deltas have nowhere to go in the
            # old format, and an internal part is one the frontend hides anyway.
            if not _is_visible_text(parts.get(event.index)):
                continue
            if sink is not None:
                sink.append(event.delta)
            yield _frame({"delta": event.delta, "isFinished": False})
        elif isinstance(event, DoneEvent):
            yield _frame({"delta": "", "isFinished": True})
        elif isinstance(event, ErrorEvent):
            message = _LEGACY_ERROR_MESSAGES.get(event.code, event.error)
            yield _frame({"delta": "", "isFinished": True, "error": message})
        # PartEndEvent / AgentStartEvent / AgentEndEvent / CastEvent carry no
        # information the legacy format can express.


def fold_legacy(events: list[Event]) -> tuple[str, str | None, ErrorEvent | None]:
    """Collapse a finished event stream into ``(content, finishReason, error)``.

    The ``Part`` objects reached through ``part_start`` / ``part_end`` are the
    same mutable objects the engine accumulated into, so reading either event is
    enough to get a part's final text — the same trick ``fold_parts`` relies on,
    and what makes a run that failed mid-answer still report what it managed to
    say.
    """
    parts: dict[int, Part] = {}
    finish_reason: str | None = None
    error: ErrorEvent | None = None
    for event in events:
        if isinstance(event, (PartStartEvent, PartEndEvent)):
            parts[event.index] = event.part
        elif isinstance(event, DoneEvent):
            finish_reason = event.finish_reason
        elif isinstance(event, ErrorEvent):
            error = event
    content = "".join(
        parts[index].text or ""
        for index in sorted(parts)
        if _is_visible_text(parts[index])
    )
    return content, finish_reason, error
