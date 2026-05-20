from typing import Any, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


class ChatTitleRequest(BaseModel):
    messages: list[ChatCompletionMessageParam] = Field(..., min_length=1)
    model: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "什麼是量子糾纏？"},
                    {"role": "assistant", "content": "量子糾纏是…"},
                ]
            }
        }
    }


class ChatTitleResponse(BaseModel):
    title: str


CHAT_TITLE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/json": {"example": {"title": "量子糾纏的基本概念"}}}
    },
    422: {"description": "No usable user/assistant turns or empty model output"},
    502: {"description": "Upstream OpenAI or LangFuse error"},
}
