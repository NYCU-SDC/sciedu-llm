from typing import Any

from pydantic import BaseModel, Field


class NamedResource(BaseModel):
    """A Langfuse resource identified by its canonical name.

    ``name`` is what downstream API calls must send; ``label`` is the same name
    with its folder prefix stripped, for display.
    """

    name: str = Field(description="Canonical Langfuse name, including folder prefix.")
    label: str = Field(description="Display name, with the folder prefix stripped.")


class ToolInfo(BaseModel):
    """One server-executed tool a preset may grant a character.

    The whole registry, not a filtered view: ``requires_rag`` says which entries
    a deployment without RAG cannot actually run, so a preset editor can warn
    about the combination rather than pretend the tool does not exist.
    """

    name: str = Field(description="The name a preset lists in `tools`.")
    description: str = Field(description="The description the model is given.")
    internal: bool = Field(
        description=(
            "Parts produced by this tool are plumbing the frontend hides by default."
        )
    )
    requires_rag: bool = Field(
        description=(
            "A preset granting this tool only runs where RAG is configured; "
            "elsewhere the run is rejected with 503."
        )
    )


class ModelDefaults(BaseModel):
    """The model ids the server would use when a request omits them."""

    eval_model: str
    judge_model: str
    embedding_model: str
    rerank_model: str


class ModelsResponse(BaseModel):
    """Every model the upstream server advertises, plus this server's policy.

    ``models`` is the unfiltered upstream listing — an admin picking an eval or
    embedding model is not restricted to ``allowed_models``, which only governs
    what `/chat` may serve.
    """

    models: list[str]
    allowed_models: list[str] = Field(
        description="Model ids the /chat endpoint is permitted to serve."
    )
    defaults: ModelDefaults


class DatasetsResponse(BaseModel):
    """Langfuse datasets grouped by the folder they live under."""

    corpus: list[NamedResource]
    questions: list[NamedResource]


UPSTREAM_RESPONSES: dict[int | str, dict[str, Any]] = {
    502: {
        "description": "Bad Gateway - an upstream listing (Langfuse / OpenAI) failed",
        "content": {
            "application/json": {
                "example": {"detail": "Failed to list Langfuse datasets: <reason>"}
            }
        },
    },
}
