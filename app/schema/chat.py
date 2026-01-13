from typing import Any, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    stream: bool
    model: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            }
        }
    }


class ChatResponseChunk(BaseModel):
    delta: str
    isFinished: bool


CHAT_SSE_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "text/event-stream": {"example": {"delta": "string", "isFinished": False}}
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
