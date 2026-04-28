import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

RETRYABLE_OPENAI_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

RETRYABLE_HTTPX_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    httpx.HTTPStatusError,
)


def is_retryable_http_error(exc: BaseException) -> bool:
    """Predicate: transport errors always retry; HTTP errors retry only on 429/5xx."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return True


def with_openai_retry(
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE_OPENAI_EXCEPTIONS,
    should_retry: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Retry an async API call with exponential backoff and bounded jitter.

    Delay before attempt n (1-indexed retry) is `base_delay * 2**(n-1)` capped at
    `max_delay`, then multiplied by a uniform factor in `[1 - jitter, 1 + jitter]`.
    Exceptions outside `retry_on` propagate immediately. If `should_retry` is
    provided, it acts as an additional filter on caught exceptions: returning
    False makes the exception propagate even though its type matched `retry_on`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    if should_retry is not None and not should_retry(exc):
                        raise
                    if attempt == max_attempts:
                        logger.warning(
                            "%s exhausted retries (%d) — re-raising %s",
                            fn.__qualname__,
                            max_attempts,
                            type(exc).__name__,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay *= 1 + random.uniform(-jitter, jitter)
                    delay = max(0.0, delay)
                    logger.warning(
                        "%s failed with %s — retry %d/%d in %.2fs",
                        fn.__qualname__,
                        type(exc).__name__,
                        attempt,
                        max_attempts - 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
