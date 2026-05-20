import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.dependencies import openai_dependency, settings_dependency
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
    request: ChatRequest, openai: openai_dependency, settings: settings_dependency
):
    model = request.model or settings.openai_default_model

    if not request.stream:
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
        async for chunk in streaming_response:
            if len(chunk.choices) == 0:
                logger.warning(
                    "recieved empty chunk from OpenAI API, skipping SSE response..."
                )
                continue

            delta_content = chunk.choices[0].delta.content
            is_finished = chunk.choices[0].finish_reason is not None

            # Prevent sending empty chunks
            if not delta_content and not is_finished:
                continue

            response_chunk = {"delta": delta_content or "", "isFinished": is_finished}
            yield f"data: {json.dumps(response_chunk)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")
