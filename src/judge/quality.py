import logging
from dataclasses import dataclass

from langfuse import Langfuse
from openai import AsyncOpenAI

from judge.config import get_judge_config
from rag.retry import with_openai_retry

logger = logging.getLogger(__name__)

# Re-exported sentinel for downstream callers/tests; sourced from JudgeConfig so
# overriding `JUDGE_FAILED_SCORE` flows through here too.
FAILED_SCORE = get_judge_config().failed_score


@dataclass(frozen=True)
class QualityScore:
    value: float
    raw: str
    extract_attempts: int  # 0 if the judge's own last token parsed cleanly

    @property
    def parsed_cleanly(self) -> bool:
        return self.extract_attempts == 0 and self.value != FAILED_SCORE


class LLMQualityJudge:
    """Run a Langfuse-managed judge prompt and coerce the output into a numeric score.

    First tries the judge's own response (last whitespace-separated token); on parse
    failure, falls back to the `extract-score-from-judgement` prompt for up to
    `max_extract_retries` attempts, finally returning FAILED_SCORE.
    """

    def __init__(
        self,
        *,
        openai: AsyncOpenAI,
        langfuse: Langfuse,
        judge_model: str,
        extract_prompt_name: str | None = None,
        max_extract_retries: int | None = None,
    ) -> None:
        config = get_judge_config()
        self._openai = openai
        self._langfuse = langfuse
        self._judge_model = judge_model
        self._extract_prompt_name = extract_prompt_name or config.extract_prompt_name
        self._max_extract_retries = (
            max_extract_retries
            if max_extract_retries is not None
            else config.max_extract_retries
        )

    async def score(
        self,
        *,
        prompt_name: str,
        question: str,
        generation: str,
        ideal: str,
        references: str,
    ) -> QualityScore:
        prompt = self._langfuse.get_prompt(prompt_name)
        compiled = prompt.compile(
            question=question,
            generation=generation,
            ideal=ideal,
            references=references,
        )

        with self._langfuse.start_as_current_observation(
            name=f"judge-{prompt_name}",
            as_type="generation",
            model=self._judge_model,
            input={
                "question": question,
                "generation": generation,
                "ideal": ideal,
                "references": references,
            },
        ) as span:
            self._langfuse.update_current_generation(prompt=prompt)
            response = await self._chat(system=compiled, user=question)
            judgement = response.choices[0].message.content or ""
            span.update(output=judgement)

        parsed = _parse_last_token(judgement)
        if parsed is not None:
            return QualityScore(value=parsed, raw=judgement, extract_attempts=0)

        return await self._extract_score_with_retry(judgement)

    async def _extract_score_with_retry(self, judgement: str) -> QualityScore:
        extract_prompt = self._langfuse.get_prompt(self._extract_prompt_name)
        compiled = extract_prompt.compile(generation=judgement)

        last_raw = judgement
        for attempt in range(1, self._max_extract_retries + 1):
            with self._langfuse.start_as_current_observation(
                name=f"judge-extract-attempt-{attempt}",
                as_type="generation",
                model=self._judge_model,
                input={"generation": judgement},
            ) as span:
                self._langfuse.update_current_generation(prompt=extract_prompt)
                response = await self._chat(system=compiled, user=judgement)
                extracted = response.choices[0].message.content or ""
                span.update(output=extracted)

            last_raw = extracted
            parsed = _parse_score(extracted)
            if parsed is not None:
                return QualityScore(
                    value=parsed, raw=judgement, extract_attempts=attempt
                )

        logger.warning(
            "Failed to extract score after %d attempts; defaulting to %s",
            self._max_extract_retries,
            FAILED_SCORE,
        )
        return QualityScore(
            value=FAILED_SCORE,
            raw=last_raw,
            extract_attempts=self._max_extract_retries,
        )

    @with_openai_retry()
    async def _chat(self, *, system: str, user: str):
        return await self._openai.chat.completions.create(
            model=self._judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )


def _parse_last_token(text: str) -> float | None:
    tokens = text.strip().split()
    if not tokens:
        return None
    return _parse_score(tokens[-1])


def _parse_score(text: str) -> float | None:
    candidate = text.strip().rstrip(".,;:!?)。，；：！？")
    if not candidate:
        return None
    try:
        return float(candidate)
    except ValueError:
        return None
