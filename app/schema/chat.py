from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.dependencies import get_settings


class ChatRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    stream: bool
    model: str = Field(default_factory=lambda: get_settings().openai_default_model)
