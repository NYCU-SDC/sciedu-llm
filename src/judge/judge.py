import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from langfuse import Evaluation, Langfuse, propagate_attributes
from langfuse.experiment import ExperimentResult
from openai import AsyncOpenAI

from judge.metrics import f1_at_k, mrr, precision_at_k, recall_at_k
from judge.quality import LLMQualityJudge, QualityScore
from rag.pipeline import RAGPipeline

JUDGE_PROMPT_PREFIX = "judge-"

QualityEvaluator = Callable[..., Awaitable[Evaluation]]

logger = logging.getLogger(__name__)


class Judge:
    """Evaluate a RAGPipeline against question datasets via Langfuse experiments.

    For each question dataset the judge runs `dataset.run_experiment(...)` so each
    item gets a trace and retrieval/quality scores hang off that trace; Langfuse
    aggregates those item-level scores into run averages on its own. All traces
    in a single `run()` call share a Langfuse session_id so the cross-dataset run
    is also browsable in one place.
    """

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        openai: AsyncOpenAI,
        langfuse: Langfuse,
        judge_model: str,
        eval_model: str,
        judge_prompts: list[str],
        k: int = 5,
        session_id: str | None = None,
        max_extract_retries: int = 10,
    ) -> None:
        if not judge_prompts:
            raise ValueError("judge_prompts must contain at least one prompt name")
        self._pipeline = pipeline
        self._langfuse = langfuse
        self._judge_model = judge_model
        self._eval_model = eval_model
        self._k = k
        self._session_id = session_id or f"judge-{uuid.uuid4()}"
        self._judge_prompts = list(judge_prompts)
        self._quality = LLMQualityJudge(
            openai=openai,
            langfuse=langfuse,
            judge_model=judge_model,
            max_extract_retries=max_extract_retries,
        )
        self._quality_evaluators: dict[str, QualityEvaluator] = {
            _metric_name(name): self._make_quality_evaluator(name)
            for name in self._judge_prompts
        }

    @property
    def session_id(self) -> str:
        return self._session_id

    async def run(
        self,
        question_dataset_names: list[str],
        corpus_dataset_names: list[str],
        max_concurrency: int = 50,
    ) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        timestamp = datetime.now(UTC).strftime("%Y%m%d %H:%M:%S")

        with propagate_attributes(
            session_id=self._session_id,
            tags=["judge", self._eval_model, self._judge_model],
            metadata={
                "eval_model": self._eval_model,
                "judge_model": self._judge_model,
                "k": str(self._k),
            },
        ):
            for index, dataset_name in enumerate(question_dataset_names):
                logger.info(
                    "Running judge experiment on %s (%d/%d)",
                    dataset_name,
                    index + 1,
                    len(question_dataset_names),
                )
                dataset = self._langfuse.get_dataset(dataset_name)
                experiment = await asyncio.to_thread(
                    dataset.run_experiment,
                    name=f"judge-{dataset_name}",
                    run_name=f"{self._eval_model} {timestamp}",
                    description=(
                        f"Judge run for eval={self._eval_model} "
                        f"judge={self._judge_model} k={self._k} "
                        f"corpus={', '.join(corpus_dataset_names)}"
                    ),
                    task=self._task,
                    evaluators=[
                        self._retrieval_evaluator,
                        *self._quality_evaluators.values(),
                    ],
                    metadata={
                        "eval_model": self._eval_model,
                        "judge_model": self._judge_model,
                        "k": str(self._k),
                    },
                    max_concurrency=max_concurrency,
                )
                results.append(experiment)

        self._langfuse.flush()
        return results

    async def _task(self, *, item, **_kwargs) -> dict[str, Any]:
        question = item.input["question"]
        return await self._pipeline.generate(
            query=question,
            model=self._eval_model,
            final_k=self._k,
        )

    async def _retrieval_evaluator(
        self,
        *,
        input: Any,  # noqa: A002 — protocol forces this shadowing
        output: dict[str, Any],
        expected_output: Any,
        metadata: dict[str, Any] | None,
        **_kwargs,
    ) -> list[Evaluation]:
        relevant = self._resolve_relevant_chunks(expected_output)
        retrieved: list[int] = output.get("reference_chunks", [])
        k = self._k

        recall = recall_at_k(retrieved, relevant, k)
        precision = precision_at_k(retrieved, relevant, k)
        f1 = f1_at_k(retrieved, relevant, k)
        reciprocal_rank = mrr(retrieved, relevant)

        comment = f"relevant={len(relevant)} retrieved={retrieved} hit={sorted(set(retrieved) & relevant)}"
        return [
            Evaluation(name=f"recall@{k}", value=recall, comment=comment),
            Evaluation(name=f"precision@{k}", value=precision, comment=comment),
            Evaluation(name=f"f1@{k}", value=f1, comment=comment),
            Evaluation(name="mrr", value=reciprocal_rank, comment=comment),
        ]

    def _make_quality_evaluator(self, prompt_name: str) -> QualityEvaluator:
        metric_name = _metric_name(prompt_name)

        async def evaluator(
            *,
            input: Any,  # noqa: A002
            output: dict[str, Any],
            expected_output: Any,
            metadata: dict[str, Any] | None,
            **_kwargs,
        ) -> Evaluation:
            return await self._quality_evaluation(
                metric_name=metric_name,
                prompt_name=prompt_name,
                input=input,
                output=output,
                expected_output=expected_output,
            )

        evaluator.__name__ = f"_evaluator_{metric_name}"
        return evaluator

    async def _quality_evaluation(
        self,
        *,
        metric_name: str,
        prompt_name: str,
        input: Any,  # noqa: A002
        output: dict[str, Any],
        expected_output: Any,
    ) -> Evaluation:
        question = self._extract_question(input)
        generation = output.get("output_text", "")
        ideal, references = self._extract_ideal_and_references(expected_output)

        score: QualityScore = await self._quality.score(
            prompt_name=prompt_name,
            question=question,
            generation=generation,
            ideal=ideal,
            references=references,
        )
        return Evaluation(
            name=metric_name,
            value=score.value,
            comment=("clean parse" if score.parsed_cleanly else "fallback extract"),
            metadata={
                "raw": score.raw,
                "extract_attempts": score.extract_attempts,
            },
        )

    def _resolve_relevant_chunks(self, expected_output: Any) -> set[int]:
        if not isinstance(expected_output, dict):
            return set()
        raw = expected_output.get("ref_text_coords")
        if raw is None:
            return set()
        coords = raw if isinstance(raw, list) else json.loads(raw)
        relevant: set[int] = set()
        for entry in coords:
            source = entry.get("source")
            pair = entry.get("coords") or []
            if not source or len(pair) != 2:
                continue
            start, end = int(pair[0]), int(pair[1])
            relevant.update(self._pipeline.resolve_chunks(source, start, end))
        return relevant

    @staticmethod
    def _extract_question(input_value: Any) -> str:
        if isinstance(input_value, dict):
            return str(input_value.get("question", ""))
        return str(input_value or "")

    @staticmethod
    def _extract_ideal_and_references(
        expected_output: Any,
    ) -> tuple[str, str]:
        if not isinstance(expected_output, dict):
            return "", ""
        ideal = str(expected_output.get("gold_answer", ""))
        raw_refs = expected_output.get("ref_text", "")
        if isinstance(raw_refs, list):
            ref_list = raw_refs
        elif isinstance(raw_refs, str) and raw_refs:
            try:
                ref_list = json.loads(raw_refs)
            except json.JSONDecodeError:
                ref_list = [raw_refs]
        else:
            ref_list = []
        references = "\n---\n".join(str(r) for r in ref_list)
        return ideal, references


def _metric_name(prompt_name: str) -> str:
    return prompt_name.removeprefix(JUDGE_PROMPT_PREFIX) or prompt_name
