from typing import Any, Literal, Optional, Union

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.agents.engine import MAX_STEPS_CAP
from app.agents.tools import registered_tool_names


class ToolFunctionRef(BaseModel):
    name: str


class FunctionToolSelector(BaseModel):
    """The spec's tool shape.

    ``docs/agents-spec.md`` writes tools as full OpenAI function objects. Every
    tool here is executed by this server, so the parameter schema has to be ours —
    only ``function.name`` is read, and any ``description`` / ``parameters`` sent
    along are ignored in favour of the registry's.
    """

    type: Literal["function"] = "function"
    function: ToolFunctionRef


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionRef


ToolSelector = Union[str, FunctionToolSelector]
ToolChoice = Union[Literal["auto", "none", "required"], NamedToolChoice]


class AgentsRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    stream: bool
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional model id to use for this run. Must be one of the server's "
            "configured allowed models (ALLOWED_MODELS); requests for any other "
            "model are rejected with a 400. When omitted, the server default "
            "(OPENAI_DEFAULT_MODEL) is used."
        ),
    )
    tools: list[ToolSelector] = Field(
        default_factory=list,
        description=(
            'Tools to make available for this run, as names (e.g. "rag_search") '
            "or as OpenAI function objects whose `function.name` matches one. All "
            "tools are executed by this server, so only registered names are "
            "accepted: "
            f"{', '.join(registered_tool_names())}. An unknown name is a 400."
        ),
    )
    tool_choice: ToolChoice = Field(
        default="auto",
        description=(
            "How freely the model may call tools on its first step. `required` and "
            "a named choice apply to the first step only — after that the model "
            "must be able to stop calling tools and answer."
        ),
    )
    max_steps: int = Field(
        default=8,
        ge=1,
        le=MAX_STEPS_CAP,
        description=(
            "Maximum number of model turns that may request tools. When the budget "
            "runs out the server makes one final tool-free call so the user still "
            'gets an answer, and reports finishReason="max_steps".'
        ),
    )
    enable_rag: bool = Field(
        default=False,
        description=(
            "Shorthand for registering `rag_search`, leaving the model to decide "
            "when to search. Unlike /chat, retrieval is not forced on every turn."
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
                "model": "gpt-oss-120b",
                "tools": ["rag_search", "summon_subagent"],
                "tool_choice": "auto",
                "max_steps": 8,
                "session": "05aec25d-a8eb-4b50-bb3f-57bbf03c05a3",
                "user": "fd965427-14c9-47cb-9d95-8ffc488d90d4",
            }
        }
    }

    @property
    def tool_names(self) -> list[str]:
        """The requested tool names, whichever shape they arrived in."""
        return [
            tool if isinstance(tool, str) else tool.function.name for tool in self.tools
        ]

    @property
    def named_tool_choice(self) -> str | None:
        if isinstance(self.tool_choice, NamedToolChoice):
            return self.tool_choice.function.name
        return None


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
        "description": (
            "Bad Request - unknown tool name, a tool_choice naming a tool that is "
            "not registered for the run, or a model outside the allowed list"
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": "Unknown tool 'search'. Available tools: rag_search, summon_subagent."
                }
            }
        },
    },
    422: {
        "description": "Unprocessable Entity - the request body failed validation",
        "content": {
            "application/json": {
                "example": {"detail": "max_steps must be between 1 and 16."}
            }
        },
    },
    503: {
        "description": "Service Unavailable - a requested tool needs RAG, which is not configured",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Tool 'rag_search' requires RAG, which is not enabled on this server. Configure RAG_CORPUS_DATASETS to enable it."
                }
            }
        },
    },
}
