"""Part identity for one assistant message."""

from app.agents.events import Part, PartType

# Engine invariant: a single answer producing more parts than this means something
# is looping. Hitting it aborts the run with an `internal_error` rather than
# streaming unboundedly.
MAX_PARTS = 200


class PartLedgerExhausted(RuntimeError):
    """Raised when a single message exceeds ``MAX_PARTS`` parts."""


class PartLedger:
    """Allocates part ids and indices for one assistant message.

    A single ledger is shared by the orchestrator and every subagent so that
    ``index`` is globally ordered across characters — the frontend aligns ``delta``
    events (and dedupes after a reconnect) on that number, so it has to be a single
    monotonic sequence regardless of who is speaking.
    """

    def __init__(self) -> None:
        self._parts: list[Part] = []

    def open(
        self,
        type: PartType,
        *,
        agent: str,
        internal: bool = False,
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> tuple[int, Part]:
        """Append a new part and return its ``(index, part)``."""
        if len(self._parts) >= MAX_PARTS:
            raise PartLedgerExhausted(f"message exceeded the {MAX_PARTS}-part limit")
        index = len(self._parts)
        part = Part(
            type=type,
            id=f"p{index}",
            agent=agent,
            internal=internal,
            tool_call_id=tool_call_id,
            name=name,
        )
        self._parts.append(part)
        return index, part

    @property
    def parts(self) -> list[Part]:
        return self._parts

    def visible_text(self, *, agent: str | None = None, since: int = 0) -> str:
        """Join the user-facing text parts.

        ``agent`` filters to one speaker and ``since`` to parts opened at or after
        that index — the summon tool needs both so a second summon in the same
        message does not pick up the first subagent's answer.
        """
        return "".join(
            part.text or ""
            for part in self._parts[since:]
            if part.type == "text"
            and not part.internal
            and (agent is None or part.agent == agent)
        )
