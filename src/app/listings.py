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
from dataclasses import dataclass
from datetime import datetime

import httpx
from langfuse import Langfuse
from openai import AsyncOpenAI

from judge.config import get_judge_config

logger = logging.getLogger(__name__)

#: Langfuse list endpoints cap out at 100 items per page.
_PAGE_LIMIT = 100

#: Pages of experiments to walk before giving up and saying so. 100 per page, so
#: this is 2000 recorded runs for one dataset — far past anything a history
#: listing can usefully show, and a bound on a cursor loop we do not control.
_MAX_EXPERIMENT_PAGES = 20

#: Timeout for the hand-rolled Langfuse REST calls below.
_HTTP_TIMEOUT_SECONDS = 30.0

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


@dataclass(frozen=True)
class ExperimentRun:
    """One recorded judge run, whichever way this Langfuse stores them.

    Langfuse v4 replaced dataset runs with *experiments*, and the two endpoints
    answer with different field names; this is the shape the admin API reports,
    so the caller does not have to care which one served it.
    """

    dataset_name: str
    name: str
    created_at: datetime
    description: str | None = None


async def list_experiment_runs(
    langfuse: Langfuse,
    dataset_name: str,
    *,
    since: datetime,
    http: httpx.AsyncClient | None = None,
) -> list[ExperimentRun]:
    """Return the recorded judge runs for `dataset_name`, newest first.

    Reads `GET /api/public/experiments`, which is where Langfuse v4 keeps them —
    a v4 deployment in ``events_only`` mode answers the older
    ``/datasets/{name}/runs`` with a 404 explaining exactly that. That older
    endpoint is still tried as a fallback, so a deployment predating experiments
    keeps working; whichever answers, the result is `ExperimentRun` objects.

    ``since`` is not optional upstream: the experiments endpoint requires
    ``fromStartTime``, so a history listing is always a window rather than "all
    of it". Raises on any Langfuse failure.
    """
    transport = _rest_transport(langfuse)
    if transport is not None:
        try:
            return await _list_experiments(
                langfuse, dataset_name, since=since, transport=transport, http=http
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            logger.info(
                "This Langfuse does not serve /api/public/experiments (404); "
                "falling back to the deprecated dataset-runs endpoint"
            )

    return await _list_dataset_runs(langfuse, dataset_name)


def _rest_transport(langfuse: Langfuse) -> tuple[str, dict[str, str]] | None:
    """Base URL + auth headers for a Langfuse REST call the SDK does not wrap.

    ``langfuse`` 4.5.1 generates no client for ``/api/public/experiments``, so
    that call is made by hand — reusing the SDK's own resolved base URL and
    credentials rather than re-reading the environment, so a client configured in
    code is still followed. ``_client_wrapper`` is a Fern-generated internal: if
    a future SDK moves it (or the caller passed a stand-in), this answers
    ``None`` and the caller falls back to a wrapped endpoint instead of failing.
    """
    wrapper = getattr(getattr(langfuse, "api", None), "_client_wrapper", None)
    if wrapper is None:
        return None
    try:
        base_url = wrapper.get_base_url().rstrip("/")
        headers = dict(wrapper.get_headers())
    except Exception:
        logger.warning(
            "Could not read the Langfuse SDK's base URL and credentials; the "
            "experiments endpoint cannot be called directly",
            exc_info=True,
        )
        return None
    return (base_url, headers) if base_url else None


async def _list_experiments(
    langfuse: Langfuse,
    dataset_name: str,
    *,
    since: datetime,
    transport: tuple[str, dict[str, str]],
    http: httpx.AsyncClient | None = None,
) -> list[ExperimentRun]:
    """Walk `GET /api/public/experiments` for one dataset, newest first.

    The endpoint filters by dataset *id*, not name, so the name is resolved
    first. Pagination is cursor-based: ``meta.cursor`` is absent on the last
    page. The experiments never carry the dataset name, so the requested one is
    reported back — it is the dataset that was asked about by definition.
    """
    base_url, headers = transport
    dataset_id = await _dataset_id(langfuse, dataset_name)

    client = http or httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)
    runs: list[ExperimentRun] = []
    try:
        cursor: str | None = None
        for page in range(1, _MAX_EXPERIMENT_PAGES + 1):
            params: dict[str, str | int] = {
                "fields": "core",
                "datasetId": dataset_id,
                "fromStartTime": since.isoformat(),
                "limit": _PAGE_LIMIT,
            }
            if cursor is not None:
                params["cursor"] = cursor
            response = await client.get(
                f"{base_url}/api/public/experiments", params=params, headers=headers
            )
            response.raise_for_status()
            body = response.json()
            runs.extend(
                _experiment_run(entry, dataset_name) for entry in body.get("data") or []
            )
            cursor = (body.get("meta") or {}).get("cursor")
            if not cursor:
                logger.debug(
                    "Listed %d experiment(s) for '%s' over %d page(s)",
                    len(runs),
                    dataset_name,
                    page,
                )
                break
        else:
            logger.warning(
                "Stopped after %d pages of experiments for '%s'; anything older "
                "than the %d listed is not reported",
                _MAX_EXPERIMENT_PAGES,
                dataset_name,
                len(runs),
            )
    finally:
        if http is None:
            await client.aclose()

    return sorted(runs, key=lambda run: run.created_at, reverse=True)


def _experiment_run(entry: dict, dataset_name: str) -> ExperimentRun:
    """Map one `Experiment` onto `ExperimentRun`.

    `startTime` stands in for "when this ran": an experiment has no creation
    timestamp of its own, and Langfuse documents it as the earliest event in the
    requested window — which for a judge run is when the run began, unless the
    window itself starts later, in which case it is clipped to the window.
    """
    return ExperimentRun(
        dataset_name=dataset_name,
        name=entry["name"],
        created_at=datetime.fromisoformat(entry["startTime"]),
        description=entry.get("description"),
    )


async def _dataset_id(langfuse: Langfuse, dataset_name: str) -> str:
    dataset = await asyncio.to_thread(
        langfuse.api.datasets.get, urllib.parse.quote(dataset_name, safe="")
    )
    return dataset.id


async def _list_dataset_runs(
    langfuse: Langfuse, dataset_name: str
) -> list[ExperimentRun]:
    """The pre-v4 listing: `GET /api/public/datasets/{name}/runs`, paged by number.

    Deprecated upstream and answered with a 404 by a v4 deployment running in
    ``events_only`` mode, which is why it is the fallback rather than the path.

    The name is percent-encoded first because the generated ``langfuse.api.*``
    clients interpolate path parameters raw, unlike the high-level client, which
    runs them through its own encoder. Every dataset here is foldered
    (``questions/biology``), so without this the request addresses
    ``/api/public/datasets/questions/biology/runs`` and collects the web app's
    HTML 404.
    """
    quoted = urllib.parse.quote(dataset_name, safe="")
    runs: list[ExperimentRun] = []
    page = 1
    while True:
        response = await asyncio.to_thread(
            langfuse.api.datasets.get_runs,
            dataset_name=quoted,
            page=page,
            limit=_PAGE_LIMIT,
        )
        runs.extend(
            ExperimentRun(
                dataset_name=getattr(run, "dataset_name", dataset_name),
                name=run.name,
                created_at=run.created_at,
                description=getattr(run, "description", None),
            )
            for run in response.data
        )
        if page >= response.meta.total_pages:
            break
        page += 1
    return sorted(runs, key=lambda run: run.created_at, reverse=True)
