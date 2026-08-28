"""The server-side tool registry.

Every tool here is executed by this module — nothing is handed back to the caller
to run. That is why a preset selects tools *by name* (``tools: ["rag_search"]``)
instead of shipping function schemas: the schema the model sees has to match what
we can actually execute, so the server owns it (see ``app.presets``, which
validates every name against this registry at load time).

Executors are async generators yielding ``Event | ToolOutcome``, ending with
exactly one ``ToolOutcome``. That shape exists for ``summon_subagent``: a summoned
character is a real speaker whose text has to appear *in place* in the parent
stream, which means the tool needs to emit events while it runs. A generator gives
that with no queue or extra task, so ordering is guaranteed and a client
disconnect cancels straight through the generator chain. ``rag_search`` just
yields its single outcome.
"""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.agents import errors
from app.agents.events import AgentEndEvent, Character, Event
from app.agents.ledger import PartLedger

logger = logging.getLogger(__name__)

RAG_SEARCH_TOOL = "rag_search"
SUMMON_SUBAGENT_TOOL = "summon_subagent"

# A subagent may not summon another one. Enforced structurally (the tool is
# removed from its list) and defensively here.
MAX_AGENT_DEPTH = 1

# Tool results are appended to the prompt on every subsequent step, so an
# unbounded RAG context would compound across the run.
TOOL_RESULT_MAX_CHARS = 8000
_TRUNCATION_MARKER = "\n\n…（內容過長，已截斷）"


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one tool execution, as the model will read it."""

    status: Literal["ok", "error"]
    content: str
    code: str | None = None
    # A richer value recorded on the Langfuse tool span only — never sent to the
    # model or the client.
    trace_output: Any = None


@dataclass
class ToolContext:
    """Everything a tool executor may need, assembled per call."""

    ledger: PartLedger
    langfuse: Any
    openai: Any
    settings: Any
    model: str
    caller: Character
    characters: dict[str, Character]
    tool_call_id: str
    depth: int
    rag_pipeline: Any = None
    # Who `summon_subagent` reaches. `None` on a summoned character's own
    # context, which is the structural half of "a subagent may not summon".
    summon_target_id: str | None = None


ToolExecutor = Callable[[ToolContext, Any], AsyncIterator[Event | ToolOutcome]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    args_model: type[BaseModel]
    execute: ToolExecutor
    # `internal: true` parts are hidden by the frontend by default. The summon
    # mechanism is plumbing; a textbook search is a step worth showing.
    internal: bool = False
    requires_rag: bool = False

    def definition(self) -> dict[str, Any]:
        """The OpenAI tool definition handed to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def parse_arguments(self, arguments: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(arguments)


def truncate_tool_content(content: str) -> str:
    if len(content) <= TOOL_RESULT_MAX_CHARS:
        return content
    return content[:TOOL_RESULT_MAX_CHARS] + _TRUNCATION_MARKER


def validation_problems(error: ValidationError) -> list[str]:
    """Flatten a pydantic error into short `loc: msg` strings for the model."""
    problems: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "(root)"
        problems.append(f"{location}: {item.get('msg', 'invalid')}")
    return problems


def invalid_arguments_outcome(spec: "ToolSpec", problems: list[str]) -> ToolOutcome:
    """The arguments parsed as JSON but did not satisfy the tool's schema."""
    logger.warning("invalid arguments for tool %s: %s", spec.name, problems)
    return ToolOutcome(
        status="error",
        code=errors.INVALID_ARGUMENTS,
        content=errors.invalid_arguments_message(spec.name, spec.parameters, problems),
    )


# --- rag_search -------------------------------------------------------------


class RagSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=512,
        description="要在課本中搜尋的問題或關鍵字，使用繁體中文。",
    )


_RAG_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要在課本中搜尋的問題或關鍵字，使用繁體中文。",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _run_rag_search(
    ctx: ToolContext, args: RagSearchArgs
) -> AsyncIterator[Event | ToolOutcome]:
    if ctx.rag_pipeline is None:
        # The router rejects the request up front when RAG is unconfigured, so
        # reaching here means the pipeline went away mid-run.
        logger.warning("rag_search invoked without a RAG pipeline")
        yield ToolOutcome(
            status="error",
            code=errors.TOOL_UNAVAILABLE,
            content=errors.tool_unavailable_message(RAG_SEARCH_TOOL),
        )
        return

    retrieval = await ctx.rag_pipeline.retrieve(query=args.query)
    context = (retrieval.get("context") or "").strip()
    chunk_ids = retrieval.get("reference_chunks") or []

    if not context:
        yield ToolOutcome(
            status="ok",
            content="這次檢索沒有找到相關的課本內容。請改用你已知的內容回答，並告知使用者。",
            trace_output={"reference_chunks": chunk_ids},
        )
        return

    ids = ", ".join(str(cid) for cid in chunk_ids)
    yield ToolOutcome(
        status="ok",
        content=f"以下是課本檢索結果（chunk ids: {ids}）：\n\n{context}",
        trace_output={"reference_chunks": chunk_ids, "context": context},
    )


# --- summon_subagent --------------------------------------------------------


class SummonSubagentArgs(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=4000,
        description="要交給子代理的完整任務描述。子代理看不到對話紀錄，所以請把需要的背景都寫進來。",
    )


_SUMMON_SUBAGENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "要交給子代理的完整任務描述。子代理看不到對話紀錄，"
                "所以請把需要的背景都寫進來。"
            ),
        }
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


def _compile_subagent_messages(
    ctx: ToolContext, target: Character, task: str
) -> list[Any]:
    """Build the summoned character's messages from its Langfuse chat prompt.

    The summoned character deliberately does not see the conversation history: it
    is given one task and answers it, which keeps its context small and makes the
    summon ``prompt`` argument the single contract between the two characters.
    """
    if not target.prompt_name:
        # Preset validation requires a prompt_name on every summonable
        # character, so this only fires on a hand-built cast. The caller turns
        # it into a recoverable tool error.
        raise ValueError(f"character '{target.id}' has no prompt_name to compile")
    prompt = ctx.langfuse.get_prompt(target.prompt_name, type="chat")
    return list(prompt.compile(task=task))


async def _run_summon_subagent(
    ctx: ToolContext, args: SummonSubagentArgs
) -> AsyncIterator[Event | ToolOutcome]:
    # Local import: engine imports this module for the registry, so importing it
    # at module scope would be circular.
    from app.agents.engine import AgentRunner

    if ctx.depth >= MAX_AGENT_DEPTH:
        logger.warning("summon_subagent blocked at depth %d", ctx.depth)
        yield ToolOutcome(
            status="error",
            code=errors.DEPTH_EXCEEDED,
            content=errors.depth_exceeded_message(),
        )
        return

    target = ctx.characters.get(ctx.summon_target_id or "")
    if target is None:
        yield ToolOutcome(
            status="error",
            code=errors.TOOL_UNAVAILABLE,
            content=errors.tool_unavailable_message(SUMMON_SUBAGENT_TOOL),
        )
        return

    try:
        messages = _compile_subagent_messages(ctx, target, args.prompt)
    except Exception:
        logger.exception("failed to compile the subagent prompt")
        yield ToolOutcome(
            status="error",
            code=errors.TOOL_FAILED,
            content=errors.tool_failed_message(SUMMON_SUBAGENT_TOOL),
        )
        return

    runner = AgentRunner(
        openai=ctx.openai,
        langfuse=ctx.langfuse,
        settings=ctx.settings,
        ledger=ctx.ledger,
        character=target,
        tools=resolve_tools(target.tool_names),
        model=ctx.model,
        messages=messages,
        max_steps=target.max_steps,
        depth=ctx.depth + 1,
        rag_pipeline=ctx.rag_pipeline,
        characters=ctx.characters,
        # Nobody left to summon: the tool is not in a summoned character's list
        # anyway, and this closes the loop structurally.
        summon_target_id=None,
        parent=ctx.caller.id,
        summoned_by=ctx.tool_call_id,
        observation_name="subagent",
    )

    # Where the subagent's parts begin, so a second summon in the same message
    # does not collect the first one's answer.
    start_index = len(ctx.ledger.parts)
    failed = False
    try:
        async for event in runner.run():
            yield event
    except Exception:
        # A subagent failure is recoverable for the orchestrator, so it does not
        # end the stream — but the frontend must not be left showing the subagent
        # as still speaking, hence the explicit agent_end below.
        logger.exception("subagent run failed")
        failed = True
        yield AgentEndEvent(agent=target.id)

    partial = ctx.ledger.visible_text(agent=target.id, since=start_index)
    if failed:
        yield ToolOutcome(
            status="error",
            code=errors.SUBAGENT_FAILED,
            content=errors.subagent_failed_message(partial),
            trace_output={"partial": partial},
        )
        return

    yield ToolOutcome(
        status="ok",
        content=partial or "子代理沒有產生任何內容，請自己回答這個問題。",
    )


# --- registry ---------------------------------------------------------------


_REGISTRY: dict[str, ToolSpec] = {
    RAG_SEARCH_TOOL: ToolSpec(
        name=RAG_SEARCH_TOOL,
        description=(
            "搜尋課本內容。當使用者的問題需要課本裡的定義、章節內容或例子時使用，"
            "並以具體的關鍵字或問題作為 query。"
        ),
        parameters=_RAG_SEARCH_PARAMETERS,
        args_model=RagSearchArgs,
        execute=_run_rag_search,
        internal=False,
        requires_rag=True,
    ),
    SUMMON_SUBAGENT_TOOL: ToolSpec(
        name=SUMMON_SUBAGENT_TOOL,
        description=(
            "召喚一位子代理來先試著回答某個問題，之後你可以針對它的回答做訂正與補充。"
            "子代理的回答會直接顯示給使用者，所以請只在真的需要它發言時使用。"
        ),
        parameters=_SUMMON_SUBAGENT_PARAMETERS,
        args_model=SummonSubagentArgs,
        execute=_run_summon_subagent,
        internal=True,
        requires_rag=False,
    ),
}


class UnknownToolError(ValueError):
    """A request named a tool that is not in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"Unknown tool '{name}'. Available tools: {', '.join(registered_tool_names())}."
        )


def registered_tool_names() -> list[str]:
    return list(_REGISTRY)


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def resolve_tools(names: list[str] | tuple[str, ...]) -> list[ToolSpec]:
    """Resolve tool names to specs, preserving order and deduping."""
    resolved: list[ToolSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        spec = _REGISTRY.get(name)
        if spec is None:
            raise UnknownToolError(name)
        seen.add(name)
        resolved.append(spec)
    return resolved
