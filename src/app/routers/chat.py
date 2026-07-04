import contextlib
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import propagate_attributes
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

    # `messages` is what we actually send to the model. When RAG is enabled we
    # retain the full conversation history, prepend the RAG system instructions,
    # and swap only the latest user turn for a context-augmented one. Retrieval
    # is keyed off the latest user query alone. The generator prompt
    # (`rag_prompt`) is linked to the Langfuse generation below.
    messages: list[ChatCompletionMessageParam] = list(request.messages)
    rag_prompt = None
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
        user_index, query = latest
        try:
            retrieval = await rag_pipeline.retrieve(query=query)
            system_message, augmented_user_message, rag_prompt = (
                rag_pipeline.compile_generator_prompt(
                    context=retrieval["context"], query=query
                )
            )
        except Exception as e:
            logger.exception("RAG retrieval failed")
            raise HTTPException(
                status_code=502,
                detail=f"Error during RAG retrieval: {str(e)}",
            ) from e
        messages[user_index] = augmented_user_message
        messages = [system_message, *messages]

    if not request.stream:
        with trace_context(), langfuse.start_as_current_observation(
            name="chat",
            as_type="generation",
            model=model,
            input={"messages": messages},
            metadata={"stream": False, "rag": request.enable_rag},
        ) as span:
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
            span.update(
                output=response.content,
                usage_details=(
                    {"input": usage.prompt_tokens, "output": usage.completion_tokens}
                    if usage is not None
                    else None
                ),
            )
            return JSONResponse(content=response.model_dump())

    try:
        streaming_response = await openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error while communicating with the OpenAI API: {str(e)}",
        ) from e

    async def stream_response():
        with trace_context(), langfuse.start_as_current_observation(
            name="chat",
            as_type="generation",
            model=model,
            input={"messages": messages},
            metadata={"stream": True, "rag": request.enable_rag},
        ) as span:
            if rag_prompt is not None:
                langfuse.update_current_generation(prompt=rag_prompt)
            accumulated: list[str] = []
            finish_reason: str | None = None
            usage = None
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

            span.update(
                output="".join(accumulated),
                usage_details=(
                    {"input": usage.prompt_tokens, "output": usage.completion_tokens}
                    if usage is not None
                    else None
                ),
                metadata={
                    "stream": True,
                    "rag": request.enable_rag,
                    "finish_reason": finish_reason,
                },
            )

    return StreamingResponse(stream_response(), media_type="text/event-stream")
