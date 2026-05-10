"""Gradio frontend for kicking off RAG evaluation runs.

Runs as a separate process from the FastAPI app — `uv run poe ui` (port 7860).
Lists corpus / question datasets from Langfuse, lets the user pick eval and
judge models plus a `k`, and starts a background evaluation that survives
browser tab closure.
"""

import asyncio
import logging
import os
from datetime import UTC, datetime

import gradio as gr
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import AsyncOpenAI

from eval_ui.config import get_eval_ui_config
from eval_ui.runner import EvalRunner, RunState
from judge.config import get_judge_config
from observability import init_langfuse_client
from rag.config import get_rag_config

logger = logging.getLogger(__name__)

RUNS_TABLE_HEADERS = [
    "run_id",
    "status",
    "eval_model",
    "judge_model",
    "embedding_model",
    "k",
    "judge_prompts",
    "started_at",
    "duration",
    "session_id",
    "error",
]


def list_dataset_names(langfuse: Langfuse) -> tuple[list[str], list[str]]:
    """Return (corpus_names, question_names) sorted alphabetically.

    Falls back to ([], []) on any API failure so the UI can still render and
    the user gets a visible error rather than a crash.
    """
    config = get_eval_ui_config()
    try:
        names: list[str] = []
        page = 1
        while True:
            response = langfuse.api.datasets.list(page=page, limit=100)
            names.extend(d.name for d in response.data)
            if page >= response.meta.total_pages:
                break
            page += 1
    except Exception:
        logger.exception("Failed to list Langfuse datasets")
        return [], []

    corpus = sorted(n for n in names if n.startswith(config.corpus_dataset_prefix))
    questions = sorted(
        n for n in names if n.startswith(config.questions_dataset_prefix)
    )
    return corpus, questions


def list_judge_prompt_names(langfuse: Langfuse) -> list[str]:
    """Return Langfuse prompt names with the configured judge prefix, sorted.

    Falls back to [] on any API failure so the UI still renders.
    """
    prefix = get_judge_config().prompt_prefix
    try:
        names: list[str] = []
        page = 1
        while True:
            response = langfuse.api.prompts.list(page=page, limit=100)
            names.extend(p.name for p in response.data)
            if page >= response.meta.total_pages:
                break
            page += 1
    except Exception:
        logger.exception("Failed to list Langfuse prompts")
        return []

    return sorted(n for n in names if n.startswith(prefix))


def list_model_ids(openai: AsyncOpenAI) -> list[str]:
    """Return model ids served by `OPENAI_BASE_URL/v1/models`, sorted alphabetically.

    Falls back to [] on any API failure. Runs synchronously off the event loop
    via `asyncio.run`, which is fine because this is invoked at UI build time.
    `models.list()` returns an `AsyncPaginator`, not a coroutine — we iterate it.
    """

    async def _fetch() -> list[str]:
        return [model.id async for model in openai.models.list()]

    try:
        return sorted(asyncio.run(_fetch()))
    except Exception:
        logger.exception("Failed to list OpenAI models")
        return []


def _error(message: str) -> str:
    return f'<span style="color: var(--color-red-600, #dc2626)"><b>Error:</b> {message}</span>'


def _format_duration(state: RunState) -> str:
    end = state.finished_at or datetime.now(UTC)
    seconds = int((end - state.started_at).total_seconds())
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s"


def _runs_to_rows(runs: list[RunState]) -> list[list[str]]:
    return [
        [
            r.run_id,
            r.status.value,
            r.eval_model,
            r.judge_model,
            r.embedding_model,
            str(r.k),
            ", ".join(r.judge_prompts),
            r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            _format_duration(r),
            r.session_id or "",
            r.error or "",
        ]
        for r in runs
    ]


def build_demo(
    runner: EvalRunner, langfuse: Langfuse, openai: AsyncOpenAI
) -> gr.Blocks:
    rag_config = get_rag_config()
    default_eval = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-oss-120b")
    default_judge = os.getenv("JUDGE_MODEL", default_eval)
    initial_corpus, initial_questions = list_dataset_names(langfuse)
    initial_models = list_model_ids(openai)
    initial_judges = list_judge_prompt_names(langfuse)

    eval_choices = _ensure_choice(initial_models, default_eval)
    judge_choices = _ensure_choice(initial_models, default_judge)
    embedding_choices = _ensure_choice(initial_models, rag_config.embedding_model)
    rerank_choices = _ensure_choice(initial_models, rag_config.rerank_model)

    with gr.Blocks(title="sciedu-llm — RAG Eval Runner") as demo:
        gr.Markdown("# sciedu-llm — RAG Eval Runner")
        gr.Markdown(
            "Pick datasets and models, click **Start evaluation**. "
            "Runs continue in the background even if you close this tab."
        )

        with gr.Row():
            with gr.Column():
                eval_model = gr.Dropdown(
                    label="Eval model id",
                    choices=eval_choices,
                    value=default_eval,
                    allow_custom_value=True,
                    filterable=True,
                )
                judge_model = gr.Dropdown(
                    label="Judge model id",
                    choices=judge_choices,
                    value=default_judge,
                    allow_custom_value=True,
                    filterable=True,
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                    label="k (final retrieval depth)",
                )
            with gr.Column():
                corpus_select = gr.CheckboxGroup(
                    label="Corpus datasets",
                    choices=initial_corpus,
                )
                question_select = gr.CheckboxGroup(
                    label="Question datasets",
                    choices=initial_questions,
                )
                judge_select = gr.CheckboxGroup(
                    label="Judge evaluators (Langfuse prompts)",
                    choices=initial_judges,
                    value=initial_judges,
                )
                refresh_btn = gr.Button("Refresh datasets, models, and judges")

        with gr.Accordion("RAG settings", open=False):
            with gr.Row():
                embedding_model = gr.Dropdown(
                    label="Embedding model",
                    choices=embedding_choices,
                    value=rag_config.embedding_model,
                    allow_custom_value=True,
                    filterable=True,
                )
                rerank_model = gr.Dropdown(
                    label="Rerank model",
                    choices=rerank_choices,
                    value=rag_config.rerank_model,
                    allow_custom_value=True,
                    filterable=True,
                )
            with gr.Row():
                chunk_size = gr.Number(
                    label="Chunk size",
                    value=rag_config.chunk_size,
                    precision=0,
                    minimum=1,
                )
                chunk_overlap = gr.Number(
                    label="Chunk overlap",
                    value=rag_config.chunk_overlap,
                    precision=0,
                    minimum=0,
                )

        start_btn = gr.Button("Start evaluation", variant="primary")
        feedback = gr.Markdown(visible=False)

        gr.Markdown("## Runs")
        runs_table = gr.Dataframe(
            headers=RUNS_TABLE_HEADERS,
            value=_runs_to_rows(runner.list()),
            interactive=False,
            wrap=True,
        )
        timer = gr.Timer(value=3.0)

        async def on_start(
            eval_m,
            judge_m,
            corpus_sel,
            question_sel,
            judges_sel,
            k,
            embedding_m,
            rerank_m,
            chunk_sz,
            chunk_ov,
        ):
            corpus_sel = corpus_sel or []
            question_sel = question_sel or []
            judges_sel = judges_sel or []
            eval_m = (eval_m or "").strip()
            judge_m = (judge_m or "").strip()
            embedding_m = (embedding_m or "").strip()
            rerank_m = (rerank_m or "").strip()

            def _fail(message: str):
                return (
                    gr.update(value=_error(message), visible=True),
                    _runs_to_rows(runner.list()),
                )

            if not eval_m or not judge_m:
                return _fail("eval and judge model ids are required.")
            if not embedding_m or not rerank_m:
                return _fail("embedding and rerank model ids are required.")
            if not corpus_sel:
                return _fail("select at least one corpus dataset.")
            if not question_sel:
                return _fail("select at least one question dataset.")
            if not judges_sel:
                return _fail("select at least one judge evaluator.")
            chunk_sz_int = int(chunk_sz)
            chunk_ov_int = int(chunk_ov)
            if chunk_sz_int < 1:
                return _fail("chunk size must be at least 1.")
            if chunk_ov_int < 0 or chunk_ov_int >= chunk_sz_int:
                return _fail("chunk overlap must be in [0, chunk_size).")

            state = runner.start(
                eval_model=eval_m,
                judge_model=judge_m,
                corpus=corpus_sel,
                questions=question_sel,
                k=int(k),
                embedding_model=embedding_m,
                rerank_model=rerank_m,
                chunk_size=chunk_sz_int,
                chunk_overlap=chunk_ov_int,
                judge_prompts=judges_sel,
                max_concurrency=rag_config.max_concurrency,
            )
            return (
                gr.update(
                    value=f"Started **{state.run_id}** — session `{state.session_id or 'pending'}`",
                    visible=True,
                ),
                _runs_to_rows(runner.list()),
            )

        def on_refresh():
            corpus, questions = list_dataset_names(langfuse)
            models = list_model_ids(openai)
            judges = list_judge_prompt_names(langfuse)
            datasets_failed = not corpus and not questions
            warnings: list[str] = []
            if datasets_failed:
                warnings.append("could not list datasets — check Langfuse credentials")
            if not models:
                warnings.append("could not list models — check OPENAI_BASE_URL")
            if not judges:
                judge_prefix = get_judge_config().prompt_prefix
                warnings.append(
                    f"no Langfuse prompts starting with '{judge_prefix}' found"
                )
            return (
                gr.update(choices=corpus),
                gr.update(choices=questions),
                gr.update(choices=judges, value=judges),
                gr.update(choices=models),
                gr.update(choices=models),
                gr.update(choices=models),
                gr.update(choices=models),
                gr.update(
                    value=_error("; ".join(warnings)) if warnings else "",
                    visible=bool(warnings),
                ),
            )

        def on_tick():
            return _runs_to_rows(runner.list())

        start_btn.click(
            on_start,
            inputs=[
                eval_model,
                judge_model,
                corpus_select,
                question_select,
                judge_select,
                k_slider,
                embedding_model,
                rerank_model,
                chunk_size,
                chunk_overlap,
            ],
            outputs=[feedback, runs_table],
        )
        refresh_btn.click(
            on_refresh,
            inputs=[],
            outputs=[
                corpus_select,
                question_select,
                judge_select,
                eval_model,
                judge_model,
                embedding_model,
                rerank_model,
                feedback,
            ],
        )
        timer.tick(on_tick, inputs=[], outputs=[runs_table])

    return demo


def _ensure_choice(choices: list[str], value: str) -> list[str]:
    """Make sure `value` appears in the dropdown choices so it renders selected.

    With `allow_custom_value=True` Gradio still picks the first choice when the
    value isn't present; prepending keeps env-var defaults visible even if the
    `/v1/models` listing failed or doesn't advertise the model.
    """
    if value and value not in choices:
        return [value, *choices]
    return choices


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    openai_client = AsyncOpenAI()
    langfuse_client = init_langfuse_client()
    runner = EvalRunner(openai_client, langfuse_client)

    demo = build_demo(runner, langfuse_client, openai_client)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=get_eval_ui_config().port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
