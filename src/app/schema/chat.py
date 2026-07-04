from typing import Any, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    stream: bool
    model: Optional[str] = None
    enable_rag: bool = False
    session: Optional[str] = Field(
        default=None,
        description=(
            "Optional session identifier. When provided, it is forwarded to "
            "Langfuse as the trace `session_id` to group related turns of a "
            "conversation together."
        ),
    )
    user: Optional[str] = Field(
        default=None,
        description=(
            "Optional user identifier. When provided, it is forwarded to "
            "Langfuse as the trace `user_id` for per-user tracking and analytics."
        ),
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
                "enable_rag": False,
                "session": "05aec25d-a8eb-4b50-bb3f-57bbf03c05a3",
                "user": "fd965427-14c9-47cb-9d95-8ffc488d90d4",
            }
        }
    }


class ChatResponseChunk(BaseModel):
    delta: str
    isFinished: bool


class ChatResponse(BaseModel):
    content: str
    finishReason: Optional[str] = None


CHAT_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "text/event-stream": {"example": {"delta": "string", "isFinished": False}},
            "application/json": {
                "example": {"content": "Hello!", "finishReason": "stop"}
            },
        }
    },
    502: {
        "description": "Bad Gateway - Error communicating with the OpenAI API",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Error while communicating with the OpenAI API: Connection timeout"
                }
            }
        },
    },
}
