"""The agentic step loop.

``AgentRunner`` drives one character: call the model, stream its parts out, run any
tools it asked for, feed the results back, repeat. It is deliberately
HTTP-agnostic — ``run()`` is an async generator of events and the router does the
SSE framing — so the loop can be tested without parsing SSE.

Two structural notes:

* Everything happens inside the generator, so the Langfuse observations stay
  current while the response streams and nested spans (``rag-retrieve``, tool
  spans, the subagent's own tree) group under the right parent. This mirrors
  ``src/app/routers/chat.py:190-194``.
* ``finally`` blocks in here only ever touch Langfuse — they never ``yield``.
  When a client disconnects, ``GeneratorExit`` is thrown in at the current
  ``yield``; yielding again from the unwind would raise
  ``RuntimeError: async generator ignored GeneratorExit`` and mask the real story.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.agents import errors
from app.agents.events import (
    AgentEndEvent,
    AgentStartEvent,
    CastEvent,
    Character,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    Event,
    Part,
    PartEndEvent,
    PartStartEvent,
)
from app.agents.ledger import PartLedger, PartLedgerExhausted
from app.agents.tools import (
    ToolContext,
    ToolOutcome,
    ToolSpec,
    invalid_arguments_outcome,
    truncate_tool_content,
    validation_problems,
)
from rag.retry import with_openai_retry

logger = logging.getLogger(__name__)

# Hard ceiling on the client-supplied `max_steps`.
MAX_STEPS_CAP = 16

# After this many tool calls in a row come back as errors, the tools are dropped
# for the rest of the run and the model is told to answer with what it has. Without
# it, a model that misunderstands an argument schema burns the whole step budget
# retrying the same broken call and the user gets nothing.
MAX_CONSECUTIVE_TOOL_ERRORS = 3

DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0


class UpstreamError(RuntimeError):
    """The upstream OpenAI-compatible API failed. Terminal for the stream."""


@with_openai_retry(max_attempts=3)
async def _create_stream(openai: Any, **kwargs: Any) -> Any:
    """Open a streaming completion, retrying transient upstream failures.

    ``/chat`` deliberately has no retries, but an agent loop makes several calls
    per answer, so a single transient 429 on step 5 of 8 would throw away all the
    work done so far. Only the connection is retried — once chunks are flowing a
    failure is not safely repeatable.
    """
    return await openai.chat.completions.create(**kwargs)


class AgentRunner:
    """Runs one character's turn-taking loop over a shared part ledger."""

    def __init__(
        self,
        *,
        openai: Any,
        langfuse: Any,
        settings: Any,
        ledger: PartLedger,
        character: Character,
        characters: dict[str, Character],
        tools: list[ToolSpec],
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_steps: int,
        tool_choice: Any = "auto",
        depth: int = 0,
        rag_pipeline: Any = None,
        summon_target_id: str | None = None,
        parent: str | None = None,
        summoned_by: str | None = None,
        observation_name: str = "agent",
    ) -> None:
        self._openai = openai
        self._langfuse = langfuse
        self._settings = settings
        self._ledger = ledger
        self._character = character
        self._characters = characters
        self._tools = list(tools)
        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._model = model
        self._messages: list[Any] = list(messages)
        self._max_steps = max(1, max_steps)
        self._tool_choice = tool_choice
        self._depth = depth
        self._rag_pipeline = rag_pipeline
        self._summon_target_id = summon_target_id
        self._parent = parent
        self._summoned_by = summoned_by
        self._observation_name = observation_name
        self._tool_timeout = float(
            getattr(
                settings, "agents_tool_timeout_seconds", DEFAULT_TOOL_TIMEOUT_SECONDS
            )
        )

        self._start_index = len(ledger.parts)
        self._finish_reason: str | None = None
        self._steps_used = 0
        self._tool_calls_made = 0
        self._consecutive_tool_errors = 0
        self._tools_disabled = False
        # Set while streaming one turn.
        self._turn_tool_parts: list[tuple[Part, bool]] = []
        self._turn_text = ""

    # --- public surface ----------------------------------------------------

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    @property
    def visible_text(self) -> str:
        return self._ledger.visible_text(
            agent=self._character.id, since=self._start_index
        )

    async def run(self) -> AsyncIterator[Event]:
        """Stream this character's parts, framed by ``agent_start`` / ``agent_end``.

        On failure the exception propagates *without* an ``agent_end`` — the caller
        emits it, because only the caller knows whether the failure is terminal for
        the whole stream (orchestrator) or recoverable (a summoned subagent).
        """
        failed = False
        with self._langfuse.start_as_current_observation(
            name=self._observation_name,
            as_type="agent",
            input={"messages": list(self._messages)},
            metadata={
                "agent": self._character.id,
                "depth": self._depth,
                "tools": [tool.name for tool in self._tools],
                "max_steps": self._max_steps,
            },
        ) as span:
            try:
                yield AgentStartEvent(
                    agent=self._character.id,
                    parent=self._parent,
                    summoned_by=self._summoned_by,
                )
                async for event in self._loop():
                    yield event
            except Exception:
                failed = True
                raise
            finally:
                update: dict[str, Any] = {
                    "output": self.visible_text,
                    "metadata": {
                        "finish_reason": self._finish_reason,
                        "steps_used": self._steps_used,
                        "tool_calls": self._tool_calls_made,
                        "circuit_breaker": self._tools_disabled,
                    },
                }
                if failed:
                    update["level"] = "ERROR"
                span.update(**update)

        yield AgentEndEvent(agent=self._character.id)

    # --- the loop ----------------------------------------------------------

    async def _loop(self) -> AsyncIterator[Event]:
        step = 0
        while True:
            step += 1
            self._steps_used = step
            tools_active = bool(self._tools) and not self._tools_disabled

            async for event in self._stream_turn(step=step, tools_active=tools_active):
                yield event

            if not self._turn_tool_parts:
                return

            spoke_elsewhere = False
            for part, parse_failed in self._turn_tool_parts:
                self._tool_calls_made += 1
                async for event in self._execute_tool_call(part, parse_failed):
                    if isinstance(event, AgentEndEvent) and event.agent != (
                        self._character.id
                    ):
                        spoke_elsewhere = True
                    yield event

            if spoke_elsewhere:
                # Another character held the floor; hand it back explicitly so the
                # frontend switches the speaker back before the next part arrives.
                yield AgentStartEvent(agent=self._character.id)

            if step >= self._max_steps:
                async for event in self._forced_final_turn(
                    step=step + 1, note=errors.max_steps_note()
                ):
                    yield event
                self._finish_reason = "max_steps"
                return

    async def _forced_final_turn(self, *, step: int, note: str) -> AsyncIterator[Event]:
        """One last call with no tools, so the user always gets a real answer."""
        self._messages.append({"role": "system", "content": note})
        self._steps_used = step
        async for event in self._stream_turn(step=step, tools_active=False):
            yield event

    # --- one model turn ----------------------------------------------------

    async def _stream_turn(
        self, *, step: int, tools_active: bool
    ) -> AsyncIterator[Event]:
        """Stream a single model turn into parts.

        Sets ``self._turn_tool_parts`` to the tool calls the model asked for (with
        a flag for whether their arguments failed to parse) and ``self._turn_text``
        to the plain text it produced.
        """
        self._turn_tool_parts = []
        self._turn_text = ""

        agent = self._character.id
        open_index: int | None = None
        open_part: Part | None = None
        # tc.index -> accumulator. Fragments are buffered until the tool name is
        # known: `internal` (and therefore whether the frontend shows the part at
        # all) depends on which tool it is, so emitting `part_start` before then
        # could flash a part that should have stayed hidden.
        pending: dict[int, dict[str, Any]] = {}
        text_chunks: list[str] = []
        usage: Any = None
        failed = False

        # A snapshot, because `self._messages` keeps growing as the loop appends
        # this turn's assistant and tool messages — handing the live list to
        # Langfuse would trace each step with the *final* history instead of the
        # one it actually ran on.
        messages = list(self._messages)

        with self._langfuse.start_as_current_observation(
            name="generation",
            as_type="generation",
            model=self._model,
            input={"messages": messages},
            metadata={"step": step, "depth": self._depth, "tools": tools_active},
        ) as generation:
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if tools_active:
                    kwargs["tools"] = [tool.definition() for tool in self._tools]
                    # A `required` / named choice only applies to the first step:
                    # re-forcing it every step would make the model unable to ever
                    # stop calling tools and finish its answer.
                    kwargs["tool_choice"] = self._tool_choice if step == 1 else "auto"

                try:
                    stream = await _create_stream(self._openai, **kwargs)
                except Exception as e:
                    raise UpstreamError(str(e)) from e

                # Iterated by hand so only failures coming *out of the stream*
                # become UpstreamError; a bug in the part handling below stays its
                # own exception and is reported as an internal error instead.
                stream_iterator = stream.__aiter__()
                while True:
                    try:
                        chunk = await stream_iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        raise UpstreamError(str(e)) from e

                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage = chunk_usage

                    choices = getattr(chunk, "choices", None) or []
                    if len(choices) == 0:
                        logger.debug("skipping chunk with no choices")
                        continue

                    choice = choices[0]
                    chunk_finish = getattr(choice, "finish_reason", None)
                    if chunk_finish is not None:
                        self._finish_reason = chunk_finish

                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    content = getattr(delta, "content", None)
                    reasoning = _reasoning_delta(delta)
                    tool_calls = getattr(delta, "tool_calls", None) or []

                    for part_type, value in (
                        ("reasoning", reasoning),
                        ("text", content),
                    ):
                        if not value:
                            continue
                        if open_part is None or open_part.type != part_type:
                            if open_part is not None and open_index is not None:
                                yield PartEndEvent(open_index, open_part)
                            open_index, open_part = self._ledger.open(
                                part_type,  # type: ignore[arg-type]
                                agent=agent,
                            )
                            yield PartStartEvent(open_index, open_part)
                        open_part.text = (open_part.text or "") + value
                        yield DeltaEvent(open_index, value)
                        if part_type == "text":
                            text_chunks.append(value)

                    if tool_calls and open_part is not None and open_index is not None:
                        yield PartEndEvent(open_index, open_part)
                        open_part = None
                        open_index = None

                    for fragment in tool_calls:
                        async for event in self._accumulate_tool_call(
                            fragment, pending=pending, step=step, agent=agent
                        ):
                            yield event

                if open_part is not None and open_index is not None:
                    yield PartEndEvent(open_index, open_part)
                    open_part = None
                    open_index = None

                for event in self._close_tool_calls(pending):
                    yield event

            except Exception:
                failed = True
                raise
            finally:
                self._turn_text = "".join(text_chunks)
                update: dict[str, Any] = {
                    "output": self._turn_text,
                    "metadata": {
                        "finish_reason": self._finish_reason,
                        "step": step,
                        "tool_calls": [part.name for part, _ in self._turn_tool_parts],
                    },
                }
                if usage is not None:
                    update["usage_details"] = {
                        "input": getattr(usage, "prompt_tokens", None),
                        "output": getattr(usage, "completion_tokens", None),
                    }
                if failed:
                    update["level"] = "ERROR"
                    update["status_message"] = "agent generation failed"
                generation.update(**update)

        self._append_assistant_message()

    async def _accumulate_tool_call(
        self,
        fragment: Any,
        *,
        pending: dict[int, dict[str, Any]],
        step: int,
        agent: str,
    ) -> AsyncIterator[Event]:
        """Fold one streamed ``delta.tool_calls[i]`` fragment into its part."""
        tc_index = getattr(fragment, "index", None)
        if tc_index is None:
            tc_index = 0
        entry = pending.setdefault(
            tc_index,
            {
                "id": None,
                "name": "",
                "args": "",
                "emitted": 0,
                "index": None,
                "part": None,
            },
        )

        call_id = getattr(fragment, "id", None)
        if call_id and entry["id"] is None:
            entry["id"] = call_id

        function = getattr(fragment, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        if name:
            entry["name"] = f"{entry['name']}{name}"
        arguments = (
            getattr(function, "arguments", None) if function is not None else None
        )
        if arguments:
            entry["args"] = f"{entry['args']}{arguments}"

        if entry["part"] is None:
            if not entry["name"]:
                # Nothing to show yet — keep buffering until the name arrives.
                return
            if not entry["id"]:
                entry["id"] = f"call_{step}_{tc_index}"
                logger.warning(
                    "tool call fragment arrived without an id; synthesized %s",
                    entry["id"],
                )
            spec = self._tools_by_name.get(entry["name"])
            part_index, part = self._ledger.open(
                "tool_call",
                agent=agent,
                internal=spec.internal if spec is not None else False,
                tool_call_id=entry["id"],
                name=entry["name"],
            )
            entry["index"] = part_index
            entry["part"] = part
            yield PartStartEvent(part_index, part)

        part = entry["part"]
        part_index = entry["index"]
        if len(entry["args"]) > entry["emitted"]:
            yield DeltaEvent(part_index, entry["args"][entry["emitted"] :])
            entry["emitted"] = len(entry["args"])
        part.raw_arguments = entry["args"]

    def _close_tool_calls(self, pending: dict[int, dict[str, Any]]) -> list[Event]:
        """Finish every tool_call part of this turn and record it for execution."""
        events: list[Event] = []
        seen_ids: set[str] = set()
        for tc_index in sorted(pending):
            entry = pending[tc_index]
            part: Part | None = entry["part"]
            if part is None:
                logger.warning(
                    "dropping tool call %d: the model never sent a tool name", tc_index
                )
                continue
            if part.tool_call_id in seen_ids:
                # A duplicate id would make the assistant message invalid upstream.
                logger.warning("dropping duplicate tool_call_id %s", part.tool_call_id)
                continue
            seen_ids.add(part.tool_call_id or "")

            parse_failed = False
            raw = entry["args"].strip()
            if not raw:
                part.arguments = {}
            else:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "tool %s arguments are not valid JSON: %r", part.name, raw
                    )
                    parse_failed = True
                    part.arguments = {}
                else:
                    if isinstance(parsed, dict):
                        part.arguments = parsed
                    else:
                        parse_failed = True
                        part.arguments = {}

            events.append(PartEndEvent(entry["index"], part))
            self._turn_tool_parts.append((part, parse_failed))
        return events

    def _append_assistant_message(self) -> None:
        """Replay the turn back into the history so the next step is well-formed."""
        if not self._turn_tool_parts and not self._turn_text:
            return
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self._turn_text or None,
        }
        if self._turn_tool_parts:
            message["tool_calls"] = [
                {
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": part.name,
                        "arguments": part.raw_arguments or "{}",
                    },
                }
                for part, _ in self._turn_tool_parts
            ]
        self._messages.append(message)

    # --- tool execution ----------------------------------------------------

    async def _execute_tool_call(
        self, part: Part, parse_failed: bool
    ) -> AsyncIterator[Event]:
        """Run one tool call and emit its ``tool_result`` part.

        Every failure here is recoverable: it becomes a ``tool_result`` with
        ``status: "error"`` and a code, the model is told what to do about it, and
        the run continues. Nothing a tool does ends the stream.
        """
        name = part.name or ""
        spec = self._tools_by_name.get(name)
        outcome: ToolOutcome

        with self._langfuse.start_as_current_observation(
            name=f"tool-{name or 'unknown'}",
            as_type="tool",
            input=part.arguments,
        ) as tool_span:
            if spec is None:
                logger.warning("model requested unregistered tool %r", name)
                outcome = ToolOutcome(
                    status="error",
                    code=errors.UNKNOWN_TOOL,
                    content=errors.unknown_tool_message(
                        name, [tool.name for tool in self._tools]
                    ),
                )
            elif parse_failed:
                outcome = ToolOutcome(
                    status="error",
                    code=errors.INVALID_ARGUMENTS,
                    content=errors.invalid_json_message(spec.name, spec.parameters),
                )
            else:
                try:
                    args = spec.parse_arguments(part.arguments or {})
                except ValidationError as e:
                    outcome = invalid_arguments_outcome(spec, validation_problems(e))
                else:
                    holder: list[ToolOutcome] = []
                    async for event in self._invoke(spec, args, part, holder):
                        yield event
                    outcome = holder[0] if holder else _missing_outcome(spec)

            content = truncate_tool_content(outcome.content)
            tool_span.update(
                output=outcome.trace_output
                if outcome.trace_output is not None
                else content,
                metadata={"status": outcome.status, "code": outcome.code},
                **(
                    {"level": "ERROR", "status_message": outcome.code or "tool failed"}
                    if outcome.status == "error"
                    else {}
                ),
            )

        if outcome.status == "error":
            self._consecutive_tool_errors += 1
        else:
            self._consecutive_tool_errors = 0

        result_index, result_part = self._ledger.open(
            "tool_result",
            agent=self._character.id,
            internal=spec.internal if spec is not None else part.internal,
            tool_call_id=part.tool_call_id,
        )
        result_part.status = outcome.status
        result_part.code = outcome.code
        result_part.content = content
        yield PartStartEvent(result_index, result_part)
        yield PartEndEvent(result_index, result_part)

        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": part.tool_call_id,
                "content": content,
            }
        )

        if (
            not self._tools_disabled
            and self._consecutive_tool_errors >= MAX_CONSECUTIVE_TOOL_ERRORS
        ):
            logger.warning(
                "%d consecutive tool failures — disabling tools for this run",
                self._consecutive_tool_errors,
            )
            self._tools_disabled = True
            self._messages.append(
                {"role": "system", "content": errors.circuit_breaker_note()}
            )

    async def _invoke(
        self,
        spec: ToolSpec,
        args: BaseModel,
        part: Part,
        holder: list[ToolOutcome],
    ) -> AsyncIterator[Event]:
        """Drive a tool executor, re-yielding its events and capturing its outcome.

        The timeout is applied per yielded item rather than to the whole call: a
        summoned subagent may legitimately run for a while, but it should never go
        quiet for longer than one tool's worth of time.
        """
        context = ToolContext(
            ledger=self._ledger,
            langfuse=self._langfuse,
            openai=self._openai,
            settings=self._settings,
            model=self._model,
            caller=self._character,
            characters=self._characters,
            tool_call_id=part.tool_call_id or "",
            depth=self._depth,
            rag_pipeline=self._rag_pipeline,
            summon_target_id=self._summon_target_id,
        )
        generator = spec.execute(context, args)
        # Characters this tool started but has not finished. If it dies mid-run we
        # still owe the frontend an `agent_end`, or it stays showing a speaker that
        # will never say anything again.
        open_agents: list[str] = []
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        generator.__anext__(), timeout=self._tool_timeout
                    )
                except StopAsyncIteration:
                    break
                if isinstance(item, ToolOutcome):
                    holder.append(item)
                    continue
                if isinstance(item, AgentStartEvent):
                    open_agents.append(item.agent)
                elif isinstance(item, AgentEndEvent) and item.agent in open_agents:
                    open_agents.remove(item.agent)
                yield item
        except TimeoutError:
            logger.warning(
                "tool %s produced nothing for %.0fs — giving up",
                spec.name,
                self._tool_timeout,
            )
            for agent_id in reversed(open_agents):
                yield AgentEndEvent(agent=agent_id)
            holder.clear()
            holder.append(
                ToolOutcome(
                    status="error",
                    code=errors.TOOL_TIMEOUT,
                    content=errors.tool_timeout_message(spec.name, self._tool_timeout),
                )
            )
        except Exception:
            logger.exception("tool %s failed", spec.name)
            for agent_id in reversed(open_agents):
                yield AgentEndEvent(agent=agent_id)
            holder.clear()
            holder.append(
                ToolOutcome(
                    status="error",
                    code=errors.TOOL_FAILED,
                    content=errors.tool_failed_message(spec.name),
                )
            )
        finally:
            await generator.aclose()


def _reasoning_delta(delta: Any) -> str | None:
    """Pull a reasoning token out of a delta, whatever the provider calls it."""
    for attribute in ("reasoning_content", "reasoning"):
        value = getattr(delta, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _missing_outcome(spec: ToolSpec) -> ToolOutcome:
    logger.error("tool %s finished without producing an outcome", spec.name)
    return ToolOutcome(
        status="error",
        code=errors.TOOL_FAILED,
        content=errors.tool_failed_message(spec.name),
    )


async def run_agents(
    *,
    openai: Any,
    langfuse: Any,
    settings: Any,
    characters: dict[str, Character],
    orchestrator: Character,
    tools: list[ToolSpec],
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_steps: int,
    tool_choice: Any = "auto",
    rag_pipeline: Any = None,
    summon_target_id: str | None = None,
) -> AsyncIterator[Event]:
    """Run one whole answer: ``cast`` → the orchestrator's parts → ``done``.

    Terminal failures become an ``error`` event rather than an exception, so the
    router only has to frame whatever comes out. The orchestrator's ``agent_end``
    is emitted on the failure path too, so the frontend never keeps a speaker
    highlighted forever.
    """
    ledger = PartLedger()

    # A cast is only meaningful when more than one character can speak; the spec
    # says to omit it for single-character runs. The orchestrator comes first.
    if len(characters) > 1:
        ordered = [orchestrator] + [
            characters[key] for key in sorted(characters) if key != orchestrator.id
        ]
        yield CastEvent(characters=tuple(c.info() for c in ordered))

    runner = AgentRunner(
        openai=openai,
        langfuse=langfuse,
        settings=settings,
        ledger=ledger,
        character=orchestrator,
        characters=characters,
        tools=tools,
        model=model,
        messages=messages,
        max_steps=max_steps,
        tool_choice=tool_choice,
        rag_pipeline=rag_pipeline,
        summon_target_id=summon_target_id,
        observation_name="agents",
    )

    try:
        async for event in runner.run():
            yield event
    except PartLedgerExhausted:
        logger.exception("agent run exceeded the part limit")
        yield AgentEndEvent(agent=orchestrator.id)
        yield ErrorEvent(
            error=errors.INTERNAL_ERROR_MESSAGE, code=errors.INTERNAL_ERROR
        )
        return
    except UpstreamError:
        logger.exception("agent run failed against the upstream API")
        yield AgentEndEvent(agent=orchestrator.id)
        yield ErrorEvent(
            error=errors.UPSTREAM_ERROR_MESSAGE, code=errors.UPSTREAM_ERROR
        )
        return
    except Exception:
        logger.exception("agent run failed")
        yield AgentEndEvent(agent=orchestrator.id)
        yield ErrorEvent(
            error=errors.INTERNAL_ERROR_MESSAGE, code=errors.INTERNAL_ERROR
        )
        return

    yield DoneEvent(finish_reason=runner.finish_reason, status="completed")


def fold_parts(events: list[Event]) -> tuple[list[dict[str, Any]], str | None, str]:
    """Collapse a finished event stream into ``(parts, finishReason, status)``.

    Used for ``stream: false``. Reading the same events the streaming path emits
    keeps the two representations from drifting apart.

    Both ``part_start`` and ``part_end`` are serialized with ``end_payload()``:
    the events hold a reference to the same mutable ``Part``, so by the time the
    run is over even a part that was cut off mid-stream carries everything it
    managed to accumulate. Without that, a failed run would report empty parts.
    """
    parts: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    status = "completed"
    for event in events:
        if isinstance(event, (PartStartEvent, PartEndEvent)):
            parts[event.index] = event.part.end_payload()
        elif isinstance(event, DoneEvent):
            finish_reason = event.finish_reason
            status = event.status
        elif isinstance(event, ErrorEvent):
            status = "failed"
            finish_reason = event.code
    return [parts[index] for index in sorted(parts)], finish_reason, status
