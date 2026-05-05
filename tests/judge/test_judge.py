import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from langfuse import Evaluation
from langfuse.experiment import ExperimentResult

from judge.judge import Judge
from judge.quality import QualityScore


class _FakePipeline:
    def __init__(self, *, retrieved: list[int], output_text: str = "answer") -> None:
        self._retrieved = retrieved
        self._output_text = output_text
        # chapter -> {(start, end) -> [chunk_ids]}
        self._coord_map: dict[str, dict[tuple[int, int], list[int]]] = {}
        self.generate_calls: list[dict] = []

    def stub_resolve(self, chapter: str, start: int, end: int, chunk_ids: list[int]):
        self._coord_map.setdefault(chapter, {})[(start, end)] = chunk_ids

    def resolve_chunks(self, chapter: str, start: int, end: int) -> list[int]:
        return self._coord_map.get(chapter, {}).get((start, end), [])

    async def generate(self, *, query, model, final_k):
        self.generate_calls.append({"query": query, "model": model, "final_k": final_k})
        return {
            "output_text": self._output_text,
            "reference_chunks": list(self._retrieved),
        }


class _FakeLangfuse:
    def __init__(self) -> None:
        self.scores: list[dict] = []
        self.flushed = False

    def get_dataset(self, name: str):
        raise NotImplementedError

    @contextmanager
    def start_as_current_observation(self, **_kwargs):
        yield SimpleNamespace(update=lambda **_: None)

    def update_current_generation(self, **_kwargs) -> None:
        pass

    def get_prompt(self, name: str):
        return SimpleNamespace(compile=lambda **_: f"{name}-compiled")

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def _make_judge(
    pipeline: _FakePipeline,
    langfuse: _FakeLangfuse,
    *,
    k: int = 5,
    judge_prompts=["judge-factuality", "judge-conciseness"],
) -> Judge:
    openai = SimpleNamespace()  # never used directly when we monkey-patch quality
    return Judge(
        pipeline=pipeline,
        openai=openai,  # type: ignore[arg-type]
        langfuse=langfuse,  # type: ignore[arg-type]
        judge_model="judge-m",
        eval_model="eval-m",
        judge_prompts=judge_prompts,
        k=k,
    )


@pytest.mark.asyncio
async def test_retrieval_evaluator_emits_metric_set_with_resolved_chunks():
    pipeline = _FakePipeline(retrieved=[10, 20, 30, 40, 50])
    pipeline.stub_resolve("ch1", 0, 100, [10, 20])  # 2/5 retrieved are relevant
    judge = _make_judge(pipeline, _FakeLangfuse(), k=5)

    expected_output = {
        "ref_text_coords": json.dumps(
            [{"source": "ch1", "coords": [0, 100]}], ensure_ascii=False
        )
    }

    evals = await judge._retrieval_evaluator(
        input={"question": "q"},
        output={"output_text": "a", "reference_chunks": [10, 20, 30, 40, 50]},
        expected_output=expected_output,
        metadata=None,
    )

    by_name = {e.name: e.value for e in evals}
    assert by_name["recall@5"] == 1.0
    assert by_name["precision@5"] == 2 / 5
    assert by_name["mrr"] == 1.0
    assert "f1@5" in by_name


@pytest.mark.asyncio
async def test_retrieval_evaluator_handles_already_parsed_coords_list():
    pipeline = _FakePipeline(retrieved=[1, 2, 3])
    pipeline.stub_resolve("ch1", 5, 50, [2])
    judge = _make_judge(pipeline, _FakeLangfuse(), k=3)

    evals = await judge._retrieval_evaluator(
        input={"question": "q"},
        output={"output_text": "a", "reference_chunks": [1, 2, 3]},
        expected_output={"ref_text_coords": [{"source": "ch1", "coords": [5, 50]}]},
        metadata=None,
    )
    by_name = {e.name: e.value for e in evals}
    assert by_name["recall@3"] == 1.0
    assert by_name["mrr"] == 1 / 2


@pytest.mark.asyncio
async def test_retrieval_evaluator_zero_when_coords_absent():
    pipeline = _FakePipeline(retrieved=[1, 2, 3])
    judge = _make_judge(pipeline, _FakeLangfuse(), k=3)

    evals = await judge._retrieval_evaluator(
        input={"question": "q"},
        output={"output_text": "a", "reference_chunks": [1, 2, 3]},
        expected_output={},
        metadata=None,
    )
    by_name = {e.name: e.value for e in evals}
    assert by_name["recall@3"] == 0.0
    assert by_name["mrr"] == 0.0


@pytest.mark.asyncio
async def test_quality_evaluator_compiles_prompt_inputs_from_expected_output():
    pipeline = _FakePipeline(retrieved=[])
    judge = _make_judge(pipeline, _FakeLangfuse())

    captured: list[dict] = []

    async def fake_score(**kwargs):
        captured.append(kwargs)
        return QualityScore(value=7.0, raw="...7", extract_attempts=0)

    judge._quality.score = fake_score  # type: ignore[assignment]

    expected = {
        "gold_answer": "the answer is 42",
        "ref_text": json.dumps(["passage one", "passage two"], ensure_ascii=False),
    }

    factuality_evaluator = judge._quality_evaluators["factuality"]
    result = await factuality_evaluator(
        input={"question": "Q?"},
        output={"output_text": "G", "reference_chunks": []},
        expected_output=expected,
        metadata=None,
    )

    assert isinstance(result, Evaluation)
    assert result.name == "factuality"
    assert result.value == 7.0
    assert captured[0]["prompt_name"] == "judge-factuality"
    assert captured[0]["question"] == "Q?"
    assert captured[0]["generation"] == "G"
    assert captured[0]["ideal"] == "the answer is 42"
    assert "passage one" in captured[0]["references"]
    assert "passage two" in captured[0]["references"]


@pytest.mark.asyncio
async def test_judge_generates_one_evaluator_per_prompt_with_correct_metric_name():
    pipeline = _FakePipeline(retrieved=[])
    judge = _make_judge(
        pipeline,
        _FakeLangfuse(),
        judge_prompts=["judge-factuality", "judge-conciseness", "judge-helpfulness"],
    )

    captured: list[dict] = []

    async def fake_score(**kwargs):
        captured.append(kwargs)
        return QualityScore(value=1.0, raw="1", extract_attempts=0)

    judge._quality.score = fake_score  # type: ignore[assignment]

    assert set(judge._quality_evaluators) == {
        "factuality",
        "conciseness",
        "helpfulness",
    }

    results = []
    for metric_name, evaluator in judge._quality_evaluators.items():
        evaluation = await evaluator(
            input={"question": "Q"},
            output={"output_text": "A", "reference_chunks": []},
            expected_output={"gold_answer": "g", "ref_text": []},
            metadata=None,
        )
        results.append((metric_name, evaluation))

    assert [r[1].name for r in results] == ["factuality", "conciseness", "helpfulness"]
    # Each closure must forward its OWN prompt_name, not the last one in the loop.
    assert [c["prompt_name"] for c in captured] == [
        "judge-factuality",
        "judge-conciseness",
        "judge-helpfulness",
    ]


def test_judge_requires_at_least_one_judge_prompt():
    with pytest.raises(ValueError, match="judge_prompts"):
        _make_judge(_FakePipeline(retrieved=[]), _FakeLangfuse(), judge_prompts=[])


@pytest.mark.asyncio
async def test_run_invokes_dataset_run_experiment_per_dataset():
    pipeline = _FakePipeline(retrieved=[1, 2, 3])
    langfuse = _FakeLangfuse()

    fake_experiment = ExperimentResult(
        name="x",
        run_name="r",
        description=None,
        item_results=[],
        run_evaluations=[],
        experiment_id="exp-1",
    )

    captured_calls: list[dict] = []

    def get_dataset(name: str):
        return SimpleNamespace(
            name=name,
            items=[],
            run_experiment=lambda **kwargs: (
                captured_calls.append({"dataset": name, **kwargs}),
                fake_experiment,
            )[1],
        )

    langfuse.get_dataset = get_dataset  # type: ignore[assignment]
    judge = _make_judge(pipeline, langfuse, k=3)

    results = await judge.run(
        ["questions-biology", "questions-physical"],
        ["corpus-biology", "corpus-physical"],
    )

    assert len(results) == 2
    assert [c["dataset"] for c in captured_calls] == [
        "questions-biology",
        "questions-physical",
    ]
    assert all("run_evaluators" not in c for c in captured_calls)
    # 1 retrieval evaluator + 1 per judge prompt (defaults: factuality, conciseness)
    for call in captured_calls:
        assert len(call["evaluators"]) == 3
        assert call["evaluators"][0] == judge._retrieval_evaluator
    assert langfuse.scores == []
    assert langfuse.flushed is True


@pytest.mark.asyncio
async def test_task_calls_pipeline_with_question_model_and_k():
    pipeline = _FakePipeline(retrieved=[7])
    judge = _make_judge(pipeline, _FakeLangfuse(), k=4)

    item = SimpleNamespace(input={"question": "what?"})
    result = await judge._task(item=item)
    assert result == {"output_text": "answer", "reference_chunks": [7]}
    assert pipeline.generate_calls == [
        {"query": "what?", "model": "eval-m", "final_k": 4}
    ]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
