import contextlib
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import propagate_attributes
from langfuse.model import PromptClient
from openai.types.chat import ChatCompletionMessageParam

from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    rag_pipeline_dependency,
    settings_dependency,
)
from app.schema.chat import CHAT_RESPONSE, ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"])

logger = logging.getLogger(__name__)


def _latest_user_message(
    messages: list[ChatCompletionMessageParam],
) -> tuple[int, str] | None:
    """Return (index, text) of the most recent user message with string content."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return index, content
    return None


async def _augment_messages_with_rag(
    rag_pipeline,
    messages: list[ChatCompletionMessageParam],
    user_index: int,
    query: str,
) -> tuple[list[ChatCompletionMessageParam], PromptClient | None]:
    """Run RAG retrieval and return ``(messages_to_send, rag_prompt)``.

    Retrieval emits a ``rag-retrieve`` span under whatever observation is
    currently active, so this must be called inside the chat span for the two to
    be grouped in the same trace. The full conversation history is retained; only
    the latest user turn is swapped for the context-augmented one and the RAG
    system instructions are prepended.
    """
    retrieval = await rag_pipeline.retrieve(query=query)
    system_message, augmented_user_message, rag_prompt = (
        rag_pipeline.compile_generator_prompt(context=retrieval["context"], query=query)
    )
    augmented = list(messages)
    augmented[user_index] = augmented_user_message
    return [system_message, *augmented], rag_prompt


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
):
    model = request.model or settings.openai_default_model

    # Enforce the configured allow-list. Startup validation guarantees it is
    # non-empty in a real deployment, so an empty list here means the endpoint is
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

    rag_target: tuple[int, str] | None = None
    if request.enable_rag:
        if rag_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it.",
            )
        latest = _latest_user_message(request.messages)
        if latest is None:
            raise HTTPException(
                status_code=422,
                detail="enable_rag=true requires a user message with text content.",
            )
        rag_target = latest

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
            messages: list[ChatCompletionMessageParam] = list(request.messages)
            rag_prompt: PromptClient | None = None
            if rag_target is not None:
                try:
                    messages, rag_prompt = await _augment_messages_with_rag(
                        rag_pipeline, request.messages, *rag_target
                    )
                except Exception as e:
                    logger.exception("RAG retrieval failed")
                    span.update(level="ERROR", status_message="RAG retrieval failed")
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error during RAG retrieval: {str(e)}",
                    ) from e

            with langfuse.start_as_current_observation(
                name="generation",
                as_type="generation",
                model=model,
                input={"messages": messages},
            ) as generation:
                if rag_prompt is not None:
                    langfuse.update_current_generation(prompt=rag_prompt)
                try:
                    completion = await openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Error while communicating with the OpenAI API: {str(e)}",
                    ) from e

                if len(completion.choices) == 0:
                    raise HTTPException(
                        status_code=502,
                        detail="OpenAI API returned no choices",
                    )

                choice = completion.choices[0]
                response = ChatResponse(
                    content=choice.message.content or "",
                    finishReason=choice.finish_reason,
                )
                usage = getattr(completion, "usage", None)
                generation.update(
                    output=response.content,
                    usage_details=(
                        {
                            "input": usage.prompt_tokens,
                            "output": usage.completion_tokens,
                        }
                        if usage is not None
                        else None
                    ),
                )

            span.update(output=response.content)
            return JSONResponse(content=response.model_dump())

    async def stream_response():
        # Everything runs inside the generator so the chat span stays the current
        # observation while the response streams: this is what lets `retrieve`
        # nest under it. The trade-off is that retrieval / upstream connection
        # errors here surface mid-stream rather than as an HTTP status.
        with (
            trace_context(),
            langfuse.start_as_current_observation(
                name="chat",
                as_type="span",
                input={"messages": request.messages},
                metadata={"stream": True, "rag": request.enable_rag},
            ) as span,
        ):
            messages: list[ChatCompletionMessageParam] = list(request.messages)
            rag_prompt: PromptClient | None = None
            if rag_target is not None:
                try:
                    messages, rag_prompt = await _augment_messages_with_rag(
                        rag_pipeline, request.messages, *rag_target
                    )
                except Exception:
                    logger.exception("RAG retrieval failed")
                    span.update(level="ERROR", status_message="RAG retrieval failed")
                    yield f"data: {json.dumps({'delta': '', 'isFinished': True, 'error': 'RAG retrieval failed'})}\n\n"
                    return

            # Accumulate the streamed content as it arrives so both observations
            # can be updated with the real (possibly partial) output on every exit
            # path, including a mid-stream failure.
            accumulated: list[str] = []
            finish_reason: str | None = None
            usage = None
            with langfuse.start_as_current_observation(
                name="generation",
                as_type="generation",
                model=model,
                input={"messages": messages},
            ) as generation:
                if rag_prompt is not None:
                    langfuse.update_current_generation(prompt=rag_prompt)
                try:
                    streaming_response = await openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    async for chunk in streaming_response:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage = chunk_usage

                        if len(chunk.choices) == 0:
                            logger.warning(
                                "received empty chunk from OpenAI API, skipping SSE response..."
                            )
                            continue

                        delta_content = chunk.choices[0].delta.content
                        chunk_finish_reason = chunk.choices[0].finish_reason
                        is_finished = chunk_finish_reason is not None
                        if is_finished:
                            finish_reason = chunk_finish_reason

                        if delta_content:
                            accumulated.append(delta_content)

                        # Prevent sending empty chunks
                        if not delta_content and not is_finished:
                            continue

                        response_chunk = {
                            "delta": delta_content or "",
                            "isFinished": is_finished,
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                except Exception:
                    logger.exception("OpenAI streaming request failed")
                    partial = "".join(accumulated)
                    generation.update(
                        output=partial,
                        level="ERROR",
                        status_message="OpenAI streaming request failed",
                    )
                    span.update(output=partial, level="ERROR")
                    yield f"data: {json.dumps({'delta': '', 'isFinished': True, 'error': 'Error while communicating with the OpenAI API'})}\n\n"
                    return

                output = "".join(accumulated)
                generation.update(
                    output=output,
                    usage_details=(
                        {
                            "input": usage.prompt_tokens,
                            "output": usage.completion_tokens,
                        }
                        if usage is not None
                        else None
                    ),
                    metadata={"finish_reason": finish_reason},
                )

            span.update(output=output)

    return StreamingResponse(stream_response(), media_type="text/event-stream")
