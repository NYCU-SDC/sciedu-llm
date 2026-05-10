import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from judge.quality import FAILED_SCORE, LLMQualityJudge


class _FakePrompt:
    def __init__(self, name: str) -> None:
        self.name = name

    def compile(self, **kwargs) -> str:
        return f"{self.name}::" + ",".join(f"{k}={v}" for k, v in kwargs.items())


class _FakeLangfuse:
    def __init__(self) -> None:
        self.prompts_by_name = {
            "judge/factuality": _FakePrompt("judge/factuality"),
            "judge/conciseness": _FakePrompt("judge/conciseness"),
            "extract-score-from-judgement": _FakePrompt("extract-score-from-judgement"),
        }

    def get_prompt(self, name: str):
        return self.prompts_by_name[name]

    @contextmanager
    def start_as_current_observation(self, **_kwargs):
        yield SimpleNamespace(update=lambda **_: None)

    def update_current_generation(self, **_kwargs) -> None:
        pass


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
    )


class _FakeOpenAI:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

        async def create(**_kwargs):
            self.calls += 1
            return _completion(self._responses.pop(0))

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)


@pytest.mark.asyncio
async def test_score_parses_last_token_when_clean():
    openai = _FakeOpenAI(["The answer is good. Score: 8"])
    judge = LLMQualityJudge(
        openai=openai, langfuse=_FakeLangfuse(), judge_model="judge-m"
    )

    result = await judge.score(
        prompt_name="judge/factuality",
        question="q",
        generation="g",
        ideal="i",
        references="r",
    )

    assert result.value == 8.0
    assert result.extract_attempts == 0
    assert result.parsed_cleanly is True
    assert openai.calls == 1


@pytest.mark.asyncio
async def test_score_strips_punctuation_around_token():
    openai = _FakeOpenAI(["...overall the score is 7。"])
    judge = LLMQualityJudge(
        openai=openai, langfuse=_FakeLangfuse(), judge_model="judge-m"
    )

    result = await judge.score(
        prompt_name="judge/factuality",
        question="q",
        generation="g",
        ideal="i",
        references="r",
    )
    assert result.value == 7.0
    assert result.extract_attempts == 0


@pytest.mark.asyncio
async def test_score_falls_back_to_extract_then_succeeds():
    openai = _FakeOpenAI(
        [
            "Honestly the response was mediocre.",  # judge — unparseable
            "garbage",  # extract attempt 1 — unparseable
            "5",  # extract attempt 2 — parseable
        ]
    )
    judge = LLMQualityJudge(
        openai=openai,
        langfuse=_FakeLangfuse(),
        judge_model="judge-m",
        max_extract_retries=10,
    )

    result = await judge.score(
        prompt_name="judge/factuality",
        question="q",
        generation="g",
        ideal="i",
        references="r",
    )
    assert result.value == 5.0
    assert result.extract_attempts == 2
    assert openai.calls == 3


@pytest.mark.asyncio
async def test_score_returns_failed_after_exhausted_retries():
    openai = _FakeOpenAI(["unparseable judgement"] + ["still bad"] * 10)
    judge = LLMQualityJudge(
        openai=openai,
        langfuse=_FakeLangfuse(),
        judge_model="judge-m",
        max_extract_retries=10,
    )

    result = await judge.score(
        prompt_name="judge/factuality",
        question="q",
        generation="g",
        ideal="i",
        references="r",
    )
    assert result.value == FAILED_SCORE
    assert result.extract_attempts == 10
    assert openai.calls == 11  # 1 judge + 10 extract attempts
