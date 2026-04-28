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
from langfuse import Langfuse, get_client
from openai import AsyncOpenAI

from eval_ui.runner import EvalRunner, RunState

logger = logging.getLogger(__name__)

CORPUS_PREFIX = "corpus-"
QUESTIONS_PREFIX = "questions-"
RUNS_TABLE_HEADERS = [
    "run_id",
    "status",
    "eval_model",
    "judge_model",
    "k",
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

    corpus = sorted(n for n in names if n.startswith(CORPUS_PREFIX))
    questions = sorted(n for n in names if n.startswith(QUESTIONS_PREFIX))
    return corpus, questions


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
            str(r.k),
            r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            _format_duration(r),
            r.session_id or "",
            r.error or "",
        ]
        for r in runs
    ]


def build_demo(runner: EvalRunner, langfuse: Langfuse) -> gr.Blocks:
    default_eval = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-oss-120b")
    default_judge = os.getenv("JUDGE_MODEL", default_eval)
    initial_corpus, initial_questions = list_dataset_names(langfuse)

    with gr.Blocks(title="sciedu-llm — RAG Eval Runner") as demo:
        gr.Markdown("# sciedu-llm — RAG Eval Runner")
        gr.Markdown(
            "Pick datasets and models, click **Start evaluation**. "
            "Runs continue in the background even if you close this tab."
        )

        with gr.Row():
            with gr.Column():
                eval_model = gr.Textbox(label="Eval model id", value=default_eval)
                judge_model = gr.Textbox(label="Judge model id", value=default_judge)
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
                refresh_btn = gr.Button("Refresh dataset list")

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

        async def on_start(eval_m, judge_m, corpus_sel, question_sel, k):
            corpus_sel = corpus_sel or []
            question_sel = question_sel or []
            if not eval_m.strip() or not judge_m.strip():
                return (
                    gr.update(
                        value=_error("eval and judge model ids are required."),
                        visible=True,
                    ),
                    _runs_to_rows(runner.list()),
                )
            if not corpus_sel:
                return (
                    gr.update(
                        value=_error("select at least one corpus dataset."),
                        visible=True,
                    ),
                    _runs_to_rows(runner.list()),
                )
            if not question_sel:
                return (
                    gr.update(
                        value=_error("select at least one question dataset."),
                        visible=True,
                    ),
                    _runs_to_rows(runner.list()),
                )

            state = runner.start(
                eval_model=eval_m.strip(),
                judge_model=judge_m.strip(),
                corpus=corpus_sel,
                questions=question_sel,
                k=int(k),
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
            warning_visible = not corpus and not questions
            return (
                gr.update(choices=corpus),
                gr.update(choices=questions),
                gr.update(
                    value=_error(
                        "could not list datasets — check Langfuse credentials."
                    ),
                    visible=warning_visible,
                ),
            )

        def on_tick():
            return _runs_to_rows(runner.list())

        start_btn.click(
            on_start,
            inputs=[eval_model, judge_model, corpus_select, question_select, k_slider],
            outputs=[feedback, runs_table],
        )
        refresh_btn.click(
            on_refresh,
            inputs=[],
            outputs=[corpus_select, question_select, feedback],
        )
        timer.tick(on_tick, inputs=[], outputs=[runs_table])

    return demo


def _max_concurrency() -> int:
    raw = os.getenv("RAG_MAX_CONCURRENCY") or os.getenv("RAG_MAX_CONCURRECNY")
    return int(raw) if raw else 64


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    openai_client = AsyncOpenAI()
    langfuse_client = get_client()
    semaphore = asyncio.Semaphore(_max_concurrency())
    runner = EvalRunner(openai_client, langfuse_client, semaphore)

    demo = build_demo(runner, langfuse_client)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("EVAL_UI_PORT", "7860")),
        show_error=True,
    )


if __name__ == "__main__":
    main()
