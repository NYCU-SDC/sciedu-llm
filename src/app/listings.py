"""Upstream listings backing the `/admin` metadata endpoints.

Everything here talks to a third-party API (Langfuse or the OpenAI-compatible
server) and every function **raises** on failure — the routers translate that
into a 502 so an admin UI shows "upstream is down" instead of an empty picker
that looks like "you have no datasets".

The Langfuse SDK surface used here (`langfuse.api.*`) is synchronous, so each
call is pushed onto a worker thread with `asyncio.to_thread` to keep the event
loop free.
"""

import asyncio
import logging
import urllib.parse

from langfuse import Langfuse
from openai import AsyncOpenAI

from judge.config import get_judge_config

logger = logging.getLogger(__name__)

#: Langfuse list endpoints cap out at 100 items per page.
_PAGE_LIMIT = 100

#: A `(label, full_name)` pair — the label is the name with its folder prefix
#: stripped (what a UI shows), the full name is the canonical Langfuse name
#: (what downstream API calls must use).
type NamePair = tuple[str, str]


def _filter_by_folder(names: list[str], folder: str) -> list[NamePair]:
    prefix = f"{folder}/"
    return sorted(
        (name.removeprefix(prefix), name) for name in names if name.startswith(prefix)
    )


async def _list_all_names(list_page, resource: str) -> list[str]:
    """Page through a Langfuse list endpoint, collecting every `.name`.

    ``list_page`` is the SDK's blocking ``list(page=..., limit=...)`` callable;
    each call is run off-loop. Any SDK error propagates to the caller.
    """
    names: list[str] = []
    page = 1
    while True:
        response = await asyncio.to_thread(list_page, page=page, limit=_PAGE_LIMIT)
        names.extend(item.name for item in response.data)
        if page >= response.meta.total_pages:
            break
        page += 1
    logger.debug("Listed %d Langfuse %s over %d page(s)", len(names), resource, page)
    return names


async def list_dataset_names(
    langfuse: Langfuse, *, corpus_folder: str, questions_folder: str
) -> tuple[list[NamePair], list[NamePair]]:
    """Return `(corpus, questions)` dataset name pairs, sorted by label.

    Both lists are filtered from a single full listing of the project's Langfuse
    datasets — one pass, two folder filters. Raises on any Langfuse failure.
    """
    names = await _list_all_names(langfuse.api.datasets.list, "datasets")
    return (
        _filter_by_folder(names, corpus_folder),
        _filter_by_folder(names, questions_folder),
    )


async def list_judge_prompt_names(langfuse: Langfuse) -> list[NamePair]:
    """Return judge prompt name pairs, sorted by label.

    Filtered to the folder configured as ``JUDGE_PROMPT_FOLDER``. Raises on any
    Langfuse failure.
    """
    names = await _list_all_names(langfuse.api.prompts.list, "prompts")
    return _filter_by_folder(names, get_judge_config().prompt_folder)


async def list_model_ids(openai: AsyncOpenAI) -> list[str]:
    """Return the model ids served by `OPENAI_BASE_URL/models`, sorted.

    `models.list()` returns an `AsyncPaginator`, not a coroutine — it is iterated
    with `async for`. Raises on any upstream failure.
    """
    return sorted([model.id async for model in openai.models.list()])


async def list_experiment_runs(langfuse: Langfuse, dataset_name: str) -> list:
    """Return the Langfuse dataset (experiment) runs for `dataset_name`, newest first.

    Each element is the SDK's ``DatasetRun`` — the caller picks the fields it
    needs. Raises on any Langfuse failure.

    The name is percent-encoded first because the generated ``langfuse.api.*``
    clients interpolate path parameters raw, unlike the high-level client, which
    runs them through its own encoder. Every dataset here is foldered
    (``questions/biology``), so without this the request addresses
    ``/api/public/datasets/questions/biology/runs`` and collects the web app's
    HTML 404.
    """
    quoted = urllib.parse.quote(dataset_name, safe="")
    runs = []
    page = 1
    while True:
        response = await asyncio.to_thread(
            langfuse.api.datasets.get_runs,
            dataset_name=quoted,
            page=page,
            limit=_PAGE_LIMIT,
        )
        runs.extend(response.data)
        if page >= response.meta.total_pages:
            break
        page += 1
    return sorted(runs, key=lambda run: run.created_at, reverse=True)
