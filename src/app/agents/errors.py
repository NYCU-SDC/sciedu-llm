"""Error codes and the model-facing messages that go with them.

Two tiers of failure:

*Recoverable* — anything to do with a tool. It becomes a ``tool_result`` part with
``status: "error"`` and the run continues, so the model gets a chance to fix its
call, fall back to its own knowledge, or tell the user what it could not do. The
messages below are written for that: they say what went wrong and what to do next.

*Terminal* — the orchestrator's own LLM call failing, or an engine invariant being
violated. The stream ends with an ``error`` event.

The ``content`` of a ``tool_result`` reaches the frontend as well as the model, so
it must never embed ``str(e)`` from an upstream exception: those messages routinely
carry base URLs, hostnames, and occasionally credentials from a query string. The
detail goes to the log and to Langfuse instead.

Guidance text is Traditional Chinese because the corpus, the prompts, and the
end users all are; machine-readable parts (tool names, JSON schemas, codes) stay
in English so the model reproduces them verbatim.
"""

import json
from typing import Any, Final

# --- tool_result error codes (recoverable) ---------------------------------
UNKNOWN_TOOL: Final = "unknown_tool"
INVALID_ARGUMENTS: Final = "invalid_arguments"
TOOL_FAILED: Final = "tool_failed"
TOOL_TIMEOUT: Final = "tool_timeout"
TOOL_UNAVAILABLE: Final = "tool_unavailable"
SUBAGENT_FAILED: Final = "subagent_failed"
DEPTH_EXCEEDED: Final = "depth_exceeded"

# --- terminal stream error codes -------------------------------------------
UPSTREAM_ERROR: Final = "upstream_error"
INTERNAL_ERROR: Final = "internal_error"

# Kept byte-identical to the /chat streaming error message so both endpoints
# surface upstream trouble the same way (see src/app/routers/chat.py:276).
UPSTREAM_ERROR_MESSAGE: Final = "Error while communicating with the OpenAI API"
INTERNAL_ERROR_MESSAGE: Final = "The agent run failed unexpectedly"


def _schema_hint(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, ensure_ascii=False, sort_keys=True)


def unknown_tool_message(name: str, available: list[str]) -> str:
    """The model asked for a tool that is not registered for this run."""
    names = "、".join(available) if available else "（無）"
    return (
        f"`{name}` is not an available tool。"
        f"這次可以使用的工具：{names}。"
        "請改用上述工具，或直接依你已知的內容回答使用者。"
    )


def invalid_json_message(name: str, parameters: dict[str, Any]) -> str:
    """The accumulated argument fragments did not parse as JSON."""
    return (
        f"呼叫 `{name}` 的參數不是合法的 JSON，無法執行。"
        "請重新呼叫一次，並確認參數符合以下 schema："
        f"{_schema_hint(parameters)}"
    )


def invalid_arguments_message(
    name: str, parameters: dict[str, Any], problems: list[str]
) -> str:
    """The arguments parsed but failed schema validation."""
    detail = "；".join(problems) if problems else "參數不符合 schema"
    return (
        f"呼叫 `{name}` 的參數有誤（{detail}）。"
        "請修正後再呼叫一次，參數 schema："
        f"{_schema_hint(parameters)}"
    )


def tool_failed_message(name: str) -> str:
    """The tool raised. Steer the model away from retrying in a loop."""
    return (
        f"`{name}` 執行失敗，這次沒有取得結果。"
        "請不要反覆重試同一個工具，改用你已知的內容回答，"
        "並告知使用者這次未能查到課本內容。"
    )


def tool_timeout_message(name: str, timeout_seconds: float) -> str:
    return (
        f"`{name}` 在 {timeout_seconds:g} 秒內沒有回應。"
        "你可以再試一次；若仍然失敗，請直接作答並說明未能查到課本內容。"
    )


def tool_unavailable_message(name: str) -> str:
    """Registered but its backing service is missing at call time."""
    return (
        f"`{name}` 目前無法使用（伺服器未提供這項服務）。"
        "請改用你已知的內容回答，並告知使用者這次沒有查詢課本。"
    )


def depth_exceeded_message() -> str:
    return "已經在子代理層級，不能再召喚下一層子代理。請自己完成這次的任務。"


def subagent_failed_message(partial_text: str) -> str:
    """The subagent errored, possibly after saying something useful."""
    if partial_text.strip():
        return (
            f"{partial_text}\n\n"
            "（子代理的回答在此中斷，未能完成。）"
            "請根據以上片段自行補完，或告知使用者這部分不完整。"
        )
    return "子代理沒有產生任何回答就失敗了。請自己回答這個問題，不要再次召喚子代理。"


def max_steps_note() -> str:
    """System note injected when the step budget runs out."""
    return (
        "已達工具使用上限，不能再呼叫任何工具。請根據目前已取得的資訊直接回答使用者。"
    )


def circuit_breaker_note() -> str:
    """System note injected when consecutive tool failures trip the breaker."""
    return (
        "工具連續失敗多次，已停用本次回答的所有工具。"
        "請根據目前已取得的資訊直接回答使用者，"
        "並在必要時說明有部分內容未能查證。"
    )
