import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    settings_dependency,
)
from app.schema.chat import CHAT_RESPONSE, ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"])

logger = logging.getLogger(__name__)


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
):
    model = request.model or settings.openai_default_model

    if not request.stream:
        with langfuse.start_as_current_observation(
            name="chat",
            as_type="generation",
            model=model,
            input={"messages": list(request.messages)},
            metadata={"stream": False},
        ) as span:
            try:
                completion = await openai.chat.completions.create(
                    model=model,
                    messages=request.messages,
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
            messages=request.messages,
            stream=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error while communicating with the OpenAI API: {str(e)}",
        ) from e

    async def stream_response():
        with langfuse.start_as_current_observation(
            name="chat",
            as_type="generation",
            model=model,
            input={"messages": list(request.messages)},
            metadata={"stream": True},
        ) as span:
            accumulated: list[str] = []
            finish_reason: str | None = None
            usage = None
            async for chunk in streaming_response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

                if len(chunk.choices) == 0:
                    logger.warning(
                        "recieved empty chunk from OpenAI API, skipping SSE response..."
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
                metadata={"stream": True, "finish_reason": finish_reason},
            )

    return StreamingResponse(stream_response(), media_type="text/event-stream")
