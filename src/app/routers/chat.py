"""The legacy chat endpoint, now a preset run wearing its old clothes.

Nothing about the wire contract changed: the same request body, the same status
codes and messages, and byte-identical SSE frames (see ``app.agents.legacy``).
What changed is underneath — instead of calling the upstream API directly, this
runs a preset through the agent engine, so both endpoints share one code path
for retrieval, tracing and upstream failure handling.

``enable_rag`` picks which preset: ``CHAT_RAG_PRESET_NAME`` (``default-chat``,
which grants the model the ``rag_search`` tool) or ``CHAT_PRESET_NAME``
(``default-chat-plain``, no tools). Retrieval is therefore the model's decision
rather than an unconditional pre-step — a question that needs no textbook no
longer pays for a retrieval, and one that needs two gets two. Nothing about that
is visible on the wire: tool calls and tool results have no place in the legacy
frame format and are dropped by the translation, so a client still sees only the
answer's text.
"""

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import propagate_attributes

from app.agents import errors
from app.agents.events import ErrorEvent, Event
from app.agents.legacy import fold_legacy, legacy_frames
from app.agents.run import (
    PresetRunError,
    latest_user_message,
    prepare_preset_run,
    run_preset,
)
from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    preset_registry_dependency,
    rag_pipeline_dependency,
    settings_dependency,
)
from app.presets import PresetNotFoundError
from app.schema.chat import CHAT_RESPONSE, ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"])

logger = logging.getLogger(__name__)


class _ErrorWatch:
    """Passes events through, remembering the terminal one if there is any.

    The legacy frames are opaque strings by the time they leave
    ``legacy_frames``, but the Langfuse span still needs to know whether the run
    ended badly, so the *events* are tapped on the way in.
    """

    def __init__(self) -> None:
        self.error: ErrorEvent | None = None

    async def tap(self, events: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async for event in events:
            if isinstance(event, ErrorEvent):
                self.error = event
            yield event


@router.post(
    "/chat",
    summary="Chat completion endpoint",
    description="Chat completions using OpenAI-compatible models. When stream=true, returns Server-Sent Events (SSE) with delta updates; when stream=false, returns a single JSON response with the full message.",
    responses=CHAT_RESPONSE,
)
async def chat(
    request: ChatRequest,
    openai: openai_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
    rag_pipeline: rag_pipeline_dependency,
    registry: preset_registry_dependency,
):
    model = request.model or settings.openai_default_model

    # Enforce the configured allow-list. Startup validation guarantees it is
    # non-empty in a real deployment, so an empty list here means the endpoint is
    # running unconfigured (e.g. in tests) and no restriction is applied.
    #
    # This stays a 400 rather than the preset machinery's 503: here the model is
    # *client* input, so an unknown one is a bad request.
    allowed_models = settings.allowed_model_names
    if allowed_models and model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' is not allowed. "
                f"Allowed models: {', '.join(allowed_models)}."
            ),
        )

    # Everything cheap is still checked before the first byte, with the exact
    # statuses and messages the old implementation returned. The 422 no longer
    # protects anything internally — retrieval is the model's call now, and it
    # can read the whole history — but it is part of the published contract, so
    # a request that used to be rejected still is.
    if request.enable_rag:
        if rag_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it.",
            )
        if latest_user_message(request.messages) is None:
            raise HTTPException(
                status_code=422,
                detail="enable_rag=true requires a user message with text content.",
            )

    preset_name = (
        settings.chat_rag_preset_name
        if request.enable_rag
        else settings.chat_preset_name
    )
    try:
        preset = await registry.get(preset_name)
    except PresetNotFoundError as e:
        logger.error(
            "/chat is configured to run preset '%s', which is not served", preset_name
        )
        raise HTTPException(
            status_code=503,
            detail=f"Chat preset '{preset_name}' is not available on this server.",
        ) from e

    try:
        prepared = prepare_preset_run(
            preset=preset,
            settings=settings,
            langfuse=langfuse,
            rag_pipeline=rag_pipeline,
            # The allow-list check above already validated it as client input.
            model_override=model,
        )
    except PresetRunError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    # Optional Langfuse trace attributes for grouping/filtering. When either is
    # provided they are propagated onto the generation span (and any child spans)
    # via `propagate_attributes`, which must wrap the observation.
    def trace_context():
        if request.session is not None or request.user is not None:
            return propagate_attributes(
                session_id=request.session,
                user_id=request.user,
            )
        return contextlib.nullcontext()

    def events():
        return run_preset(
            prepared=prepared,
            messages=list(request.messages),
            openai=openai,
            langfuse=langfuse,
            settings=settings,
            rag_pipeline=rag_pipeline,
        )

    if not request.stream:
        # Outer span groups retrieval + generation into a single trace. When RAG
        # is disabled it wraps the generation alone (same name, so both shapes
        # look consistent in Langfuse).
        with (
            trace_context(),
            langfuse.start_as_current_observation(
                name="chat",
                as_type="span",
                input={"messages": request.messages},
                metadata={"stream": False, "rag": request.enable_rag},
            ) as span,
        ):
            collected: list[Event] = []
            async for event in events():
                collected.append(event)

            content, finish_reason, error = fold_legacy(collected)
            if error is not None:
                span.update(output=content, level="ERROR", status_message=error.error)
                if error.code == errors.RAG_FAILED:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error during RAG retrieval: {error.error}",
                    )
                raise HTTPException(
                    status_code=502,
                    detail=errors.UPSTREAM_ERROR_MESSAGE,
                )

            response = ChatResponse(content=content, finishReason=finish_reason)
            span.update(output=response.content)
            return JSONResponse(content=response.model_dump())

    async def stream_response():
        # Everything runs inside the generator so the chat span stays the current
        # observation while the response streams: this is what lets `retrieve`
        # and the engine's own observations nest under it. The trade-off is that
        # retrieval / upstream connection errors here surface mid-stream rather
        # than as an HTTP status.
        with (
            trace_context(),
            langfuse.start_as_current_observation(
                name="chat",
                as_type="span",
                input={"messages": request.messages},
                metadata={"stream": True, "rag": request.enable_rag},
            ) as span,
        ):
            # Accumulate the streamed content as it arrives so the span can be
            # updated with the real (possibly partial) output on every exit path,
            # including a mid-stream failure.
            accumulated: list[str] = []
            watch = _ErrorWatch()
            async for frame in legacy_frames(watch.tap(events()), sink=accumulated):
                yield frame

            output = "".join(accumulated)
            if watch.error is not None:
                span.update(
                    output=output, level="ERROR", status_message=watch.error.error
                )
            else:
                span.update(output=output)

    return StreamingResponse(stream_response(), media_type="text/event-stream")
