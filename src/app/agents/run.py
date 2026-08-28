"""Running a preset: everything between "here is a preset" and "here are events".

Split in two on purpose:

* ``prepare_preset_run`` does the parts that can fail *before* a byte is
  written — model allow-list, RAG availability, prompt loading — and reports
  them as ``PresetRunError`` so a router can turn each into an HTTP status.
* ``run_preset`` is an async generator, so everything it does (notably the
  forced retrieval a ``rag_mode: "forced"`` preset asks for) happens while the
  caller's Langfuse observation is still current and the ``rag-retrieve`` span
  lands inside the live trace rather than beside it. This is the same reason
  both routers open their stream generator inside the trace context.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from app.agents import errors
from app.agents.cast import Cast, build_cast
from app.agents.engine import run_agents
from app.agents.events import ErrorEvent, Event
from app.agents.tools import ToolSpec, resolve_tools
from app.presets import Preset

logger = logging.getLogger(__name__)


class PresetRunError(Exception):
    """A run cannot start. Carries the HTTP status a router should return."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class PreparedRun:
    """A preset resolved against this server's configuration."""

    preset: Preset
    cast: Cast
    tools: list[ToolSpec]
    model: str
    # The orchestrator's compiled system prompt, or `None` when the preset has
    # no prompt_name (or when forced RAG supplies the system message instead).
    system_message: dict[str, Any] | None


def latest_user_message(
    messages: list[ChatCompletionMessageParam],
) -> tuple[int, str] | None:
    """Return (index, text) of the most recent user message with string content.

    Lives here rather than in ``/chat`` because both endpoints need the same
    answer to "what is the user actually asking?" before retrieving.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return index, content
    return None


def prepare_preset_run(
    *,
    preset: Preset,
    settings: Any,
    langfuse: Any,
    rag_pipeline: Any,
    model_override: str | None = None,
) -> PreparedRun:
    """Resolve a preset into something runnable, or explain why it is not.

    Every failure here is a 5xx: the preset is server-owned, so a preset asking
    for a model this deployment does not allow is a misconfiguration, not a bad
    request. A caller that lets a *client* pick the model must validate that
    itself (and 400) before passing it in as ``model_override``.
    """
    model = model_override or preset.model or settings.openai_default_model
    allowed = settings.allowed_model_names
    if allowed and model not in allowed:
        raise PresetRunError(
            503,
            f"Preset '{preset.name}' uses model '{model}', which is not in the "
            "allowed models list.",
        )

    cast = build_cast(preset)
    # Preset validation already checked every name against the registry, so this
    # cannot raise UnknownToolError for a preset that made it into service.
    tools = resolve_tools(cast.orchestrator.tool_names)

    if rag_pipeline is None:
        if preset.rag_mode == "forced":
            raise PresetRunError(
                503,
                f"Preset '{preset.name}' requires RAG, which is not enabled on "
                "this server. Configure RAG_CORPUS_DATASETS to enable it.",
            )
        for tool in tools:
            if tool.requires_rag:
                raise PresetRunError(
                    503,
                    f"Tool '{tool.name}' requires RAG, which is not enabled on "
                    "this server. Configure RAG_CORPUS_DATASETS to enable it.",
                )

    system_message: dict[str, Any] | None = None
    prompt_name = cast.orchestrator.prompt_name
    if prompt_name:
        try:
            prompt = langfuse.get_prompt(prompt_name)
            content = prompt.compile()
        except Exception as e:
            logger.exception(
                "could not load prompt '%s' for preset '%s'", prompt_name, preset.name
            )
            raise PresetRunError(
                502,
                f"Failed to load prompt '{prompt_name}' for preset '{preset.name}'.",
            ) from e
        system_message = {"role": "system", "content": content}

    return PreparedRun(
        preset=preset,
        cast=cast,
        tools=tools,
        model=model,
        system_message=system_message,
    )


async def run_preset(
    *,
    prepared: PreparedRun,
    messages: list[ChatCompletionMessageParam],
    openai: Any,
    langfuse: Any,
    settings: Any,
    rag_pipeline: Any,
) -> AsyncIterator[Event]:
    """Stream one preset run, doing forced retrieval inside the generator."""
    preset = prepared.preset
    outgoing: list[ChatCompletionMessageParam] = list(messages)

    if preset.rag_mode == "forced":
        latest = latest_user_message(outgoing)
        if latest is None:
            logger.warning(
                "preset '%s' forces RAG but the request has no user text",
                preset.name,
            )
            yield ErrorEvent(error=errors.RAG_FAILED_MESSAGE, code=errors.RAG_FAILED)
            return
        user_index, query = latest
        try:
            # The shape `/chat`'s `enable_rag` produced before retrieval became
            # the model's own decision: the whole history is kept, only the
            # latest user turn is swapped for the context-augmented one, and the
            # RAG system instructions go in front.
            retrieval = await rag_pipeline.retrieve(query=query)
            system_message, augmented_user_message, _rag_prompt = (
                rag_pipeline.compile_generator_prompt(
                    context=retrieval["context"], query=query
                )
            )
        except Exception:
            logger.exception("RAG retrieval failed")
            yield ErrorEvent(error=errors.RAG_FAILED_MESSAGE, code=errors.RAG_FAILED)
            return
        outgoing[user_index] = augmented_user_message
        outgoing = [system_message, *outgoing]
    elif prepared.system_message is not None:
        outgoing = [prepared.system_message, *outgoing]  # type: ignore[list-item]

    async for event in run_agents(
        openai=openai,
        langfuse=langfuse,
        settings=settings,
        characters=prepared.cast.characters,
        orchestrator=prepared.cast.orchestrator,
        tools=prepared.tools,
        model=prepared.model,
        messages=outgoing,
        max_steps=preset.max_steps,
        tool_choice=preset.tool_choice,
        rag_pipeline=rag_pipeline,
        summon_target_id=prepared.cast.summon_target_id,
    ):
        yield event
