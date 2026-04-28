import argparse
import logging
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from langfuse.api.commons.errors.not_found_error import NotFoundError
from tqdm import tqdm

T = TypeVar("T")

load_dotenv()

DATA_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = DATA_ROOT / "corpus"
QUESTIONS_ROOT = DATA_ROOT / "questions"

CORPUS_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<grade>\d+)(?:_(?P<semester>\d+))?_ch(?P<chapter>\d+)\.txt$"
)


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("seed")


def get_langfuse_client() -> Langfuse:
    return get_client()


def dataset_exists(client: Langfuse, name: str) -> bool:
    try:
        client.get_dataset(name)
    except NotFoundError:
        return False
    return True


def clear_dataset_items(client: Langfuse, name: str, logger: logging.Logger) -> int:
    """Delete every item in `name`. Dataset itself (and its runs) survive."""
    dataset = client.get_dataset(name)
    deleted = 0
    for item in tqdm(dataset.items, desc=f"Clearing {name}", unit="item", leave=False):
        retry_on_transport_error(
            lambda item_id=item.id: client.api.dataset_items.delete(id=item_id),
            logger=logger,
        )
        deleted += 1
    if deleted:
        logger.info("Cleared %d existing item(s) from dataset '%s'", deleted, name)
    return deleted


def retry_on_transport_error(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.25,
    logger: logging.Logger | None = None,
) -> T:
    """Call `fn()` with exponential backoff on httpx transport errors and 5xx/429.

    The Langfuse SDK retries on HTTP status codes but not on transport-level
    failures like ReadTimeout, so wrap any single-item write that needs to be
    resilient. `fn` should be idempotent (use a deterministic dataset-item id
    so the upsert behaviour kicks in if a request succeeded but the response
    was lost).
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status != 429 and status < 500:
                raise
            _backoff(attempt, attempts, exc, base_delay, max_delay, jitter, logger)
        except httpx.TransportError as exc:
            _backoff(attempt, attempts, exc, base_delay, max_delay, jitter, logger)
    raise RuntimeError("unreachable")  # pragma: no cover


def _backoff(
    attempt: int,
    attempts: int,
    exc: BaseException,
    base_delay: float,
    max_delay: float,
    jitter: float,
    logger: logging.Logger | None,
) -> None:
    if attempt == attempts:
        if logger:
            logger.error(
                "Gave up after %d attempt(s); last error: %s",
                attempts,
                type(exc).__name__,
            )
        raise exc
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    delay *= 1 + random.uniform(-jitter, jitter)
    delay = max(0.0, delay)
    if logger:
        logger.warning(
            "%s on attempt %d/%d — retrying in %.1fs",
            type(exc).__name__,
            attempt,
            attempts,
            delay,
        )
    time.sleep(delay)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing items in matching datasets before re-seeding.",
    )
    return parser.parse_args()


def parse_corpus_filename(filename: str) -> dict | None:
    match = CORPUS_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {
        "subject": match.group("subject"),
        "grade": match.group("grade"),
        "semester": match.group("semester"),
        "chapter": int(match.group("chapter")),
    }


def corpus_dataset_name(subject: str, grade: str, semester: str | None) -> str:
    if semester:
        return f"corpus-{subject}-{grade}-{semester}"
    return f"corpus-{subject}-{grade}"
