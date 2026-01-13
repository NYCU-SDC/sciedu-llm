from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.dependencies import get_settings


class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    stream: bool
    model: str = Field(default_factory=lambda: get_settings().openai_default_model)
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
    }
}
