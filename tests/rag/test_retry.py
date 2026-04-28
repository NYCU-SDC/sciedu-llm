import asyncio

import httpx
import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from rag.retry import with_openai_retry


def _build_api_status_error(cls, *, status_code: int):
    request = httpx.Request("POST", "https://example.test/v1/x")
    response = httpx.Response(status_code=status_code, request=request)
    return cls(message="boom", response=response, body=None)


@pytest.mark.asyncio
async def test_returns_value_on_first_success():
    calls = 0

    @with_openai_retry()
    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    assert await fn() == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds_on_retryable_exception():
    calls = 0

    @with_openai_retry(max_attempts=3, base_delay=0, max_delay=0, jitter=0)
    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _build_api_status_error(RateLimitError, status_code=429)
        return "ok"

    assert await fn() == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_raises_after_exhausting_retries():
    calls = 0

    @with_openai_retry(max_attempts=2, base_delay=0, max_delay=0, jitter=0)
    async def fn():
        nonlocal calls
        calls += 1
        raise _build_api_status_error(InternalServerError, status_code=500)

    with pytest.raises(InternalServerError):
        await fn()
    assert calls == 2


@pytest.mark.asyncio
async def test_does_not_retry_non_retryable_exception():
    calls = 0

    @with_openai_retry(max_attempts=5, base_delay=0, max_delay=0, jitter=0)
    async def fn():
        nonlocal calls
        calls += 1
        raise _build_api_status_error(AuthenticationError, status_code=401)

    with pytest.raises(AuthenticationError):
        await fn()
    assert calls == 1


@pytest.mark.asyncio
async def test_uses_exponential_backoff(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @with_openai_retry(max_attempts=4, base_delay=1.0, max_delay=100.0, jitter=0)
    async def fn():
        raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    with pytest.raises(APIConnectionError):
        await fn()

    # 3 retries before giving up; delays are 1, 2, 4 (no jitter).
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_max_delay_caps_backoff(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @with_openai_retry(max_attempts=5, base_delay=10.0, max_delay=15.0, jitter=0)
    async def fn():
        raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    with pytest.raises(APIConnectionError):
        await fn()

    # Without cap: 10, 20, 40, 80. Cap at 15: 10, 15, 15, 15.
    assert sleeps == [10.0, 15.0, 15.0, 15.0]


def test_invalid_max_attempts_rejected():
    with pytest.raises(ValueError):
        with_openai_retry(max_attempts=0)


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/rerank")
    response = httpx.Response(status_code=status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.asyncio
async def test_retries_httpx_transport_error():
    from rag.retry import RETRYABLE_HTTPX_EXCEPTIONS, is_retryable_http_error

    calls = 0

    @with_openai_retry(
        max_attempts=3,
        base_delay=0,
        max_delay=0,
        jitter=0,
        retry_on=RETRYABLE_HTTPX_EXCEPTIONS,
        should_retry=is_retryable_http_error,
    )
    async def fn():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise httpx.ConnectError("boom")
        return "ok"

    assert await fn() == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retries_on_429_and_5xx_status():
    from rag.retry import RETRYABLE_HTTPX_EXCEPTIONS, is_retryable_http_error

    statuses = iter([429, 503])
    calls = 0

    @with_openai_retry(
        max_attempts=4,
        base_delay=0,
        max_delay=0,
        jitter=0,
        retry_on=RETRYABLE_HTTPX_EXCEPTIONS,
        should_retry=is_retryable_http_error,
    )
    async def fn():
        nonlocal calls
        calls += 1
        try:
            status = next(statuses)
        except StopIteration:
            return "ok"
        raise _http_status_error(status)

    assert await fn() == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_on_4xx_other_than_429():
    from rag.retry import RETRYABLE_HTTPX_EXCEPTIONS, is_retryable_http_error

    calls = 0

    @with_openai_retry(
        max_attempts=5,
        base_delay=0,
        max_delay=0,
        jitter=0,
        retry_on=RETRYABLE_HTTPX_EXCEPTIONS,
        should_retry=is_retryable_http_error,
    )
    async def fn():
        nonlocal calls
        calls += 1
        raise _http_status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await fn()
    assert calls == 1


def test_is_retryable_http_error_classification():
    from rag.retry import is_retryable_http_error

    assert is_retryable_http_error(httpx.ConnectError("x")) is True
    assert is_retryable_http_error(httpx.ReadTimeout("x")) is True
    assert is_retryable_http_error(_http_status_error(429)) is True
    assert is_retryable_http_error(_http_status_error(500)) is True
    assert is_retryable_http_error(_http_status_error(503)) is True
    assert is_retryable_http_error(_http_status_error(400)) is False
    assert is_retryable_http_error(_http_status_error(401)) is False
    assert is_retryable_http_error(_http_status_error(404)) is False
