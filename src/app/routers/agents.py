import contextlib
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import propagate_attributes

from app.agents.engine import fold_parts, run_agents
from app.agents.events import ORCHESTRATOR_AGENT_ID, CastEvent, Event
from app.agents.tools import (
    ToolSpec,
    UnknownToolError,
    build_characters,
    resolve_tools,
)
from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    rag_pipeline_dependency,
    settings_dependency,
)
from app.schema.agents import AGENTS_RESPONSES, AgentsRequest, AgentsResponse

router = APIRouter(tags=["Agents"])

logger = logging.getLogger(__name__)


def _sse(event: Event) -> str:
    # `ensure_ascii=False` unlike /chat: the corpus and every prompt here are
    # Traditional Chinese, and escaping each character to \uXXXX inflates every
    # frame roughly sixfold. /agents is new, so there is no client to break.
    return f"data: {json.dumps(event.payload(), ensure_ascii=False)}\n\n"


@router.post(
    "/agents",
    summary="Agentic chat completion endpoint",
    description=(
        "Runs a multi-step agentic loop with server-executed tools (`rag_search`, "
        "`summon_subagent`). When stream=true, returns Server-Sent Events carrying "
        "the typed part protocol (`cast`, `agent_start`, `part_start`, `delta`, "
        "`part_end`, `agent_end`, `done`, `error`); when stream=false, returns the "
        "same steps folded into a single `parts` array."
    ),
    responses=AGENTS_RESPONSES,
)
async def agents(
    request: AgentsRequest,
    openai: openai_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
    rag_pipeline: rag_pipeline_dependency,
):
    model = request.model or settings.openai_default_model

    # Same allow-list semantics as /chat: an empty list means the endpoint is
    # running unconfigured (e.g. in tests) and no restriction is applied.
    allowed_models = settings.allowed_model_names
    if allowed_models and model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' is not allowed. "
                f"Allowed models: {', '.join(allowed_models)}."
            ),
        )

    try:
        tools = resolve_tools(request.tool_names, enable_rag=request.enable_rag)
    except UnknownToolError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    _validate_tool_choice(request, tools)

    # Unlike /chat — where RAG failures can only surface mid-stream — everything
    # cheap is checked before the first byte, so these stay real HTTP statuses.
    if rag_pipeline is None:
        for tool in tools:
            if tool.requires_rag:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Tool '{tool.name}' requires RAG, which is not enabled on "
                        "this server. Configure RAG_CORPUS_DATASETS to enable it."
                    ),
                )

    tool_names = [tool.name for tool in tools]
    characters = build_characters(settings, tool_names)
    orchestrator = characters[ORCHESTRATOR_AGENT_ID]

    # Optional Langfuse trace attributes for grouping/filtering, matching /chat.
    def trace_context():
        if request.session is not None or request.user is not None:
            return propagate_attributes(
                session_id=request.session,
                user_id=request.user,
            )
        return contextlib.nullcontext()

    def events():
        return run_agents(
            openai=openai,
            langfuse=langfuse,
            settings=settings,
            characters=characters,
            orchestrator=orchestrator,
            tools=tools,
            model=model,
            messages=list(request.messages),
            max_steps=request.max_steps,
            tool_choice=_openai_tool_choice(request),
            rag_pipeline=rag_pipeline,
        )

    if not request.stream:
        collected: list[Event] = []
        with trace_context():
            async for event in events():
                collected.append(event)

        parts, finish_reason, status = fold_parts(collected)
        cast = next(
            (event for event in collected if isinstance(event, CastEvent)), None
        )
        response = AgentsResponse(
            status=status,  # type: ignore[arg-type]
            finishReason=finish_reason,
            cast=(
                [character.payload() for character in cast.characters]  # type: ignore[arg-type]
                if cast is not None
                else None
            ),
            parts=parts,
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))

    async def stream_response():
        # The trace context is entered inside the generator so the Langfuse
        # observations opened by the engine stay current while the response
        # streams — the same reason /chat does it here (chat.py:190-194).
        with trace_context():
            async for event in events():
                yield _sse(event)

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def _validate_tool_choice(request: AgentsRequest, tools: list[ToolSpec]) -> None:
    named = request.named_tool_choice
    if named is not None and named not in {tool.name for tool in tools}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"tool_choice names '{named}', which is not one of this run's tools: "
                f"{', '.join(tool.name for tool in tools) or '(none)'}."
            ),
        )
    if request.tool_choice == "required" and not tools:
        raise HTTPException(
            status_code=400,
            detail="tool_choice='required' needs at least one tool in `tools`.",
        )


def _openai_tool_choice(request: AgentsRequest) -> Any:
    if isinstance(request.tool_choice, str):
        return request.tool_choice
    return request.tool_choice.model_dump()
