import logging
import re
from typing import cast

from fastapi import APIRouter, HTTPException
from openai.types.chat import ChatCompletionMessageParam

from app.dependencies import (
    langfuse_dependency,
    openai_dependency,
    settings_dependency,
)
from app.schema.title import CHAT_TITLE_RESPONSES, ChatTitleRequest, ChatTitleResponse
from rag.retry import with_openai_retry

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)

MAX_CONVERSATION_CHARS = 8000
MAX_TITLE_CHARS = 200
_QUOTE_CHARS = "\"'“”‘’「」『』《》"


def _format_conversation(messages: list[ChatCompletionMessageParam]) -> str:
    rendered: list[str] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if text:
            rendered.append(f"{role.capitalize()}: {text}")

    total = sum(len(r) + 2 for r in rendered)
    while rendered and total > MAX_CONVERSATION_CHARS:
        total -= len(rendered.pop(0)) + 2
    return "\n\n".join(rendered)


def _sanitize_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw.strip())
    for _ in range(2):
        if len(title) >= 2 and title[0] in _QUOTE_CHARS and title[-1] in _QUOTE_CHARS:
            title = title[1:-1].strip()
    title = title.rstrip(".。")
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rstrip()
    return title


@with_openai_retry()
async def _call_llm(openai, *, model: str, messages: list[ChatCompletionMessageParam]):
    return await openai.chat.completions.create(model=model, messages=messages)


@router.post(
    "/chat/title",
    summary="Generate a short chat title",
    description="Generates a 10–20 word title in the chat's source language. Prompt template is managed in LangFuse.",
    response_model=ChatTitleResponse,
    responses=CHAT_TITLE_RESPONSES,
)
async def chat_title(
    request: ChatTitleRequest,
    openai: openai_dependency,
    langfuse: langfuse_dependency,
    settings: settings_dependency,
) -> ChatTitleResponse:
    conversation = _format_conversation(request.messages)
    if not conversation:
        raise HTTPException(
            status_code=422,
            detail="No usable user or assistant turns in messages.",
        )

    model = request.model or settings.openai_default_model
    prompt_name = settings.chat_title_prompt_name
    max_attempts = max(1, settings.chat_title_max_attempts)

    try:
        prompt = langfuse.get_prompt(prompt_name, type="chat")
        compiled_messages = cast(
            list[ChatCompletionMessageParam],
            prompt.compile(conversation=conversation),
        )
    except Exception as e:
        logger.exception("Failed to fetch LangFuse prompt %r", prompt_name)
        raise HTTPException(
            status_code=502,
            detail=f"LangFuse prompt '{prompt_name}' unavailable: {e}",
        ) from e

    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        with langfuse.start_as_current_observation(
            name=f"chat-title-{prompt_name}-attempt-{attempt}",
            as_type="generation",
            model=model,
            input={"conversation": conversation},
        ) as span:
            langfuse.update_current_generation(prompt=prompt)
            try:
                response = await _call_llm(
                    openai, model=model, messages=compiled_messages
                )
            except Exception as e:
                logger.exception("OpenAI call failed in chat-title")
                raise HTTPException(
                    status_code=502,
                    detail=f"Error while communicating with the OpenAI API: {e}",
                ) from e
            raw = response.choices[0].message.content or ""
            usage = response.usage
            span.update(
                output=raw,
                usage_details=(
                    {"input": usage.prompt_tokens, "output": usage.completion_tokens}
                    if usage is not None
                    else None
                ),
            )

        last_raw = raw
        title = _sanitize_title(raw)
        if title:
            return ChatTitleResponse(title=title)
        logger.warning(
            "chat-title attempt %d/%d produced empty title (raw=%r)",
            attempt,
            max_attempts,
            raw,
        )

    raise HTTPException(
        status_code=422,
        detail=f"Model returned an empty title after {max_attempts} attempts. last_raw={last_raw!r}",
    )
