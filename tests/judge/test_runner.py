import asyncio
import contextlib

import pytest

from judge.runner import EvalRunner, RunNotCancellableError, RunStatus

pytestmark = pytest.mark.asyncio


@pytest.fixture
def install_fakes(monkeypatch):
    """Swap `RAGPipeline`/`Judge` in the runner's namespace for controllable fakes.

    `on_build` / `on_run` are optional async hooks — a test uses them to hold the
    run open (an `asyncio.Event`) or to blow it up (`raise`).
    """

    def _install(*, on_build=None, on_run=None):
        record: dict = {"pipelines": [], "judges": []}

        class FakePipeline:
            def __init__(self, openai, langfuse, **kwargs):
                self.kwargs = kwargs
                record["pipelines"].append(self)

            async def build(self, corpus_dataset_names, **kwargs):
                self.build_call = (list(corpus_dataset_names), kwargs)
                if on_build is not None:
                    await on_build()

        class FakeJudge:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.session_id = "session-abc"
                record["judges"].append(self)

            async def run(self, question_datasets, corpus_datasets, max_concurrency):
                self.run_call = (
                    list(question_datasets),
                    list(corpus_datasets),
                    max_concurrency,
                )
                if on_run is not None:
                    await on_run()

        monkeypatch.setattr("judge.runner.RAGPipeline", FakePipeline)
        monkeypatch.setattr("judge.runner.Judge", FakeJudge)
        return record

    return _install


def _start(runner: EvalRunner, **overrides):
    kwargs = {
        "eval_model": "eval-model",
        "judge_model": "judge-model",
        "corpus": ["corpus/biology"],
        "questions": ["questions/biology"],
        "k": 5,
        "embedding_model": "bge-m3",
        "rerank_model": "reranker",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "judge_prompts": ["judge/faithfulness"],
        "max_concurrency": 8,
    }
    kwargs.update(overrides)
    return runner.start(**kwargs)


async def _settle(state) -> None:
    """Await the run's task, tolerating cancellation."""
    assert state._task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await state._task


async def test_run_progresses_pending_building_judging_completed(install_fakes):
    building = asyncio.Event()
    judging = asyncio.Event()
    record = install_fakes(on_build=building.wait, on_run=judging.wait)
    runner = EvalRunner(openai=object(), langfuse=object())

    state = _start(runner)
    # `start` schedules and returns — nothing has run yet.
    assert state.status is RunStatus.PENDING
    assert state.finished_at is None

    await asyncio.sleep(0)
    assert state.status is RunStatus.BUILDING

    building.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert state.status is RunStatus.JUDGING
    assert state.session_id == "session-abc"

    judging.set()
    await _settle(state)

    assert state.status is RunStatus.COMPLETED
    assert state.error is None
    assert state.finished_at is not None
    assert state.duration_seconds >= 0
    assert record["pipelines"][0].build_call == (
        ["corpus/biology"],
        {"chunk_size": 500, "chunk_overlap": 100, "max_concurrency": 8},
    )
    assert record["judges"][0].run_call == (
        ["questions/biology"],
        ["corpus/biology"],
        8,
    )


async def test_run_records_failure(install_fakes):
    async def boom():
        raise RuntimeError("langfuse exploded")

    install_fakes(on_build=boom)
    runner = EvalRunner(openai=object(), langfuse=object())

    state = _start(runner)
    await _settle(state)

    assert state.status is RunStatus.FAILED
    assert "langfuse exploded" in (state.error or "")
    assert state.finished_at is not None


async def test_cancel_marks_run_cancelled(install_fakes):
    never = asyncio.Event()
    install_fakes(on_run=never.wait)
    runner = EvalRunner(openai=object(), langfuse=object())

    state = _start(runner)
    while state.status is not RunStatus.JUDGING:
        await asyncio.sleep(0)

    assert runner.cancel(state.run_id) is state
    await _settle(state)

    assert state.status is RunStatus.CANCELLED
    assert state.finished_at is not None
    assert state.is_terminal
    # A cancelled run is not a failed one — no error is recorded.
    assert state.error is None


async def test_cancel_unknown_run_returns_none():
    runner = EvalRunner(openai=object(), langfuse=object())

    assert runner.cancel("run-nope") is None


async def test_cancel_terminal_run_raises(install_fakes):
    install_fakes()
    runner = EvalRunner(openai=object(), langfuse=object())

    state = _start(runner)
    await _settle(state)
    assert state.status is RunStatus.COMPLETED

    with pytest.raises(RunNotCancellableError) as excinfo:
        runner.cancel(state.run_id)
    assert excinfo.value.status is RunStatus.COMPLETED


async def test_shutdown_cancels_in_flight_runs(install_fakes):
    never = asyncio.Event()
    install_fakes(on_run=never.wait)
    runner = EvalRunner(openai=object(), langfuse=object())

    live = _start(runner)
    while live.status is not RunStatus.JUDGING:
        await asyncio.sleep(0)

    runner.shutdown()
    await _settle(live)

    assert live.status is RunStatus.CANCELLED


async def test_shutdown_leaves_finished_runs_alone(install_fakes):
    install_fakes()
    runner = EvalRunner(openai=object(), langfuse=object())

    state = _start(runner)
    await _settle(state)

    runner.shutdown()

    assert state.status is RunStatus.COMPLETED


async def test_list_returns_newest_first(install_fakes):
    install_fakes()
    runner = EvalRunner(openai=object(), langfuse=object())

    first = _start(runner, eval_model="first")
    await _settle(first)
    # Ordering is by `started_at`; make sure the two timestamps cannot collide.
    await asyncio.sleep(0.01)
    second = _start(runner, eval_model="second")
    await _settle(second)

    assert [state.eval_model for state in runner.list()] == ["second", "first"]
    assert runner.get(first.run_id) is first
    assert runner.get("run-nope") is None
