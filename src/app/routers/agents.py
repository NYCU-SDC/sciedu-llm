import contextlib
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import propagate_attributes

from app.agents.learning import TeachingStateEvent, allowed_action, run_learning
from app.agents.engine import fold_parts
from app.agents.events import CastEvent, Event
from app.agents.run import PresetRunError, prepare_preset_run, run_preset
from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    preset_registry_dependency,
    rag_cache_dependency,
    settings_dependency,
)
from app.presets import PresetNotFoundError
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
        "Runs a named *preset* — a server-owned run configuration naming the "
        "model, teacher prompt, enabled tool sections, subagent behavior and "
        "step budget — as a "
        "multi-step agentic loop with server-executed tools (`rag_search`, "
        "`summon_subagent`). When stream=true, returns Server-Sent Events "
        "carrying the typed part protocol (`cast`, `agent_start`, `part_start`, "
        "`delta`, `part_end`, `agent_end`, `done`, `error`); when stream=false, "
        "returns the same steps folded into a single `parts` array."
    ),
    responses=AGENTS_RESPONSES,
)
async def agents(
    request: AgentsRequest,
    openai: openai_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
    rag_cache: rag_cache_dependency,
    registry: preset_registry_dependency,
):
    name = request.preset or settings.agents_default_preset
    try:
        preset = await registry.get(name)
    except PresetNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown preset '{name}'. Available presets: "
                f"{', '.join(registry.names())}."
            ),
        ) from e

    # Unlike /chat — where RAG failures can only surface mid-stream — everything
    # cheap is checked before the first byte, so these stay real HTTP statuses.
    rag_pipeline = rag_cache.pipeline(preset)

    try:
        prepared = prepare_preset_run(
            preset=preset,
            settings=settings,
            langfuse=langfuse,
            rag_pipeline=rag_pipeline,
        )
    except PresetRunError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    if request.learning_context is not None and request.teaching_action is not None:
        if not allowed_action(request.teaching_state, request.teaching_action):
            raise HTTPException(
                status_code=400, detail="invalid teaching action or state"
            )
        if (
            len(prepared.cast.summon_target_ids) != 1
            or not prepared.cast.characters[
                prepared.cast.summon_target_ids[0]
            ].prompt_name
        ):
            raise HTTPException(
                status_code=400, detail="argumentation requires exactly one peer"
            )
    elif request.teaching_action is not None or request.teaching_state is not None:
        raise HTTPException(
            status_code=400, detail="teaching controls require learning_context"
        )

    # Optional Langfuse trace attributes for grouping/filtering, matching /chat.
    def trace_context():
        if request.session is not None or request.user is not None:
            return propagate_attributes(
                session_id=request.session,
                user_id=request.user,
            )
        return contextlib.nullcontext()

    def events():
        if request.learning_context is not None and request.teaching_action is not None:
            return run_learning(
                prepared=prepared,
                context=request.learning_context,
                state=request.teaching_state,
                action=request.teaching_action,
                messages=list(request.messages),
                openai=openai,
                langfuse=langfuse,
                settings=settings,
                rag_pipeline=rag_pipeline,
            )
        return run_preset(
            learning_context=request.learning_context.model_dump()
            if request.learning_context is not None
            else None,
            prepared=prepared,
            messages=list(request.messages),
            openai=openai,
            langfuse=langfuse,
            settings=settings,
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
            teachingState=next(
                (
                    event.state
                    for event in collected
                    if isinstance(event, TeachingStateEvent)
                ),
                None,
            ),
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
        # streams — the same reason /chat does it here (chat.py).
        with trace_context():
            async for event in events():
                yield _sse(event)

    return StreamingResponse(stream_response(), media_type="text/event-stream")
