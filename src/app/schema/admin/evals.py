from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from judge import RunState

CHUNK_OVERLAP_DETAIL = "chunk_overlap must be smaller than chunk_size"


class EvalRunCreate(BaseModel):
    """Request body for `POST /admin/evals/runs`.

    Fields left as ``None`` are filled in from the server's ``RAGConfig``
    defaults by the router — the client only has to name what it wants to
    differ from the deployed configuration.
    """

    eval_model: str = Field(min_length=1, description="Model under evaluation.")
    judge_model: str = Field(min_length=1, description="Model grading the answers.")
    corpus_datasets: list[str] = Field(
        min_length=1, description="Langfuse corpus datasets to index for the run."
    )
    question_datasets: list[str] = Field(
        min_length=1, description="Langfuse question datasets to evaluate against."
    )
    judge_prompts: list[str] = Field(
        min_length=1, description="Langfuse judge prompt names to score with."
    )
    k: int = Field(default=5, ge=1, le=20, description="Final retrieval depth.")

    embedding_model: str | None = None
    rerank_model: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_chunking(self) -> Self:
        # Only checkable here when the client pinned both. When either is left
        # to the server, the router re-checks after resolving against RAGConfig
        # (which is env-dependent and so has no business inside the schema).
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError(CHUNK_OVERLAP_DETAIL)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "eval_model": "gpt-oss-120b",
                "judge_model": "gpt-oss-120b",
                "corpus_datasets": ["corpus/ver3/biology"],
                "question_datasets": ["questions/biology"],
                "judge_prompts": ["judge/faithfulness"],
                "k": 5,
            }
        }
    }


class EvalRunResponse(BaseModel):
    """A single evaluation run, as tracked in the process-local registry."""

    run_id: str
    status: str
    eval_model: str
    judge_model: str
    corpus_datasets: list[str]
    question_datasets: list[str]
    judge_prompts: list[str]
    k: int
    embedding_model: str
    rerank_model: str
    chunk_size: int
    chunk_overlap: int
    max_concurrency: int
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = Field(
        description="Wall-clock seconds elapsed; counts up while the run is live."
    )
    session_id: str | None = Field(
        default=None, description="Langfuse session grouping the run's traces."
    )
    error: str | None = None

    @classmethod
    def from_state(cls, state: RunState) -> "EvalRunResponse":
        return cls(
            run_id=state.run_id,
            status=state.status.value,
            eval_model=state.eval_model,
            judge_model=state.judge_model,
            corpus_datasets=state.corpus_datasets,
            question_datasets=state.question_datasets,
            judge_prompts=state.judge_prompts,
            k=state.k,
            embedding_model=state.embedding_model,
            rerank_model=state.rerank_model,
            chunk_size=state.chunk_size,
            chunk_overlap=state.chunk_overlap,
            max_concurrency=state.max_concurrency,
            started_at=state.started_at,
            finished_at=state.finished_at,
            duration_seconds=state.duration_seconds,
            session_id=state.session_id,
            error=state.error,
        )


class EvalHistoryEntry(BaseModel):
    """A past Langfuse experiment run read back off a question dataset.

    Unlike :class:`EvalRunResponse` (in-memory, lost on restart) this is durable
    — it is what Langfuse itself recorded when a judge run executed.
    """

    dataset_name: str
    run_name: str
    created_at: datetime
    description: str | None = None


EVAL_RUN_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {
        "description": "Not Found - no run with that id in this process",
        "content": {
            "application/json": {"example": {"detail": "Unknown run 'run-x'."}}
        },
    },
}
