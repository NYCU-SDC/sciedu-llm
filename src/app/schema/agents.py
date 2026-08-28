from typing import Any, Literal, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


class AgentsRequest(BaseModel):
    """What a client may decide about an agentic run: the conversation, and which
    server-owned *preset* to run it with.

    Everything else — model, tools, tool_choice, step budget, whether retrieval is
    forced, who else may speak — is named by the preset (see ``app.presets``). The
    model's default pydantic ``extra="ignore"`` is deliberately kept so an older
    client still sending ``tools`` / ``tool_choice`` / ``max_steps`` /
    ``enable_rag`` / ``model`` gets a normal 200 with those fields dropped, rather
    than a 422 on a field the server no longer honours.
    """

    messages: list[ChatCompletionMessageParam]
    stream: bool
    preset: Optional[str] = Field(
        default=None,
        description=(
            "Name of the preset to run. A preset decides the model, the cast, "
            "each character's tools, the step budget and whether retrieval is "
            "forced. When omitted, the server default (AGENTS_DEFAULT_PRESET) is "
            "used; an unknown name is a 400 listing what is available. Legacy "
            "fields (`tools`, `tool_choice`, `max_steps`, `enable_rag`, `model`) "
            "are ignored if sent."
        ),
    )
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
                "messages": [
                    {"role": "user", "content": "幫我根據課本第三章解釋光合作用"}
                ],
                "stream": True,
                "preset": "default-agents",
                "session": "05aec25d-a8eb-4b50-bb3f-57bbf03c05a3",
                "user": "fd965427-14c9-47cb-9d95-8ffc488d90d4",
            }
        }
    }


class CharacterInfoResponse(BaseModel):
    id: str
    displayName: str
    role: str


class AgentsResponse(BaseModel):
    """The non-streaming form: the same parts the SSE stream would have emitted."""

    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "failed"]
    finishReason: Optional[str] = None
    cast: Optional[list[CharacterInfoResponse]] = None
    parts: list[dict[str, Any]]


AGENTS_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "text/event-stream": {
                "example": {
                    "type": "part_start",
                    "index": 0,
                    "part": {"type": "text", "id": "p0", "agent": "assistant"},
                }
            },
            "application/json": {
                "example": {
                    "role": "assistant",
                    "status": "completed",
                    "finishReason": "stop",
                    "parts": [
                        {
                            "type": "text",
                            "id": "p0",
                            "agent": "assistant",
                            "text": "光合作用是…",
                        }
                    ],
                }
            },
        }
    },
    400: {
        "description": "Bad Request - no preset is served under the requested name",
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "Unknown preset 'teacher'. Available presets: "
                        "default-agents, default-chat, default-chat-plain."
                    )
                }
            }
        },
    },
    422: {
        "description": "Unprocessable Entity - the request body failed validation",
        "content": {
            "application/json": {
                "example": {"detail": "stream: Field required"},
            }
        },
    },
    502: {
        "description": (
            "Bad Gateway - the preset's system prompt could not be loaded from Langfuse"
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "Failed to load prompt 'agents/teacher-system' for preset "
                        "'default-agents'."
                    )
                }
            }
        },
    },
    503: {
        "description": (
            "Service Unavailable - the preset needs RAG, which is not configured, "
            "or it names a model outside the server's allowed list. Both are "
            "server-side misconfiguration: a preset is not client input."
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": "Tool 'rag_search' requires RAG, which is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it."
                }
            }
        },
    },
}
