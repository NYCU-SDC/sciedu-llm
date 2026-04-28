import asyncio

import httpx
import pytest

from rag import reranker as reranker_module
from rag.reranker import Reranker


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that yields scripted responses or raises errors."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # noqa: D401
        self.calls += 1
        action = self._script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


@pytest.fixture
def patched_async_client(monkeypatch):
    """Replace httpx.AsyncClient inside reranker with one that uses a scripted transport."""
    transports: list[_ScriptedTransport] = []

    def install(script):
        transport = _ScriptedTransport(script)
        transports.append(transport)

        original = reranker_module.httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(reranker_module.httpx, "AsyncClient", factory)
        return transport

    return install


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "r-1",
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ],
        },
    )


def _status_response(status: int) -> httpx.Response:
    return httpx.Response(status, json={"error": "boom"})


@pytest.mark.asyncio
async def test_rerank_returns_index_score_pairs(patched_async_client):
    transport = patched_async_client([_ok_response()])
    rr = Reranker(base_url="https://example.test/v1", api_key="key")

    result = await rr.rerank(query="q", documents=["a", "b", "c"], top_n=2)

    assert result == [(2, 0.91), (0, 0.42)]
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_rerank_retries_then_succeeds_on_503(patched_async_client):
    transport = patched_async_client(
        [_status_response(503), _status_response(503), _ok_response()]
    )
    rr = Reranker(base_url="https://example.test/v1", api_key="key")

    result = await rr.rerank(query="q", documents=["a", "b"], top_n=2)

    assert result == [(2, 0.91), (0, 0.42)]
    assert transport.calls == 3


@pytest.mark.asyncio
async def test_rerank_retries_on_transport_error(patched_async_client):
    transport = patched_async_client([httpx.ConnectError("boom"), _ok_response()])
    rr = Reranker(base_url="https://example.test/v1", api_key="key")

    result = await rr.rerank(query="q", documents=["a"], top_n=1)

    assert result == [(2, 0.91), (0, 0.42)]
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_rerank_does_not_retry_on_400(patched_async_client):
    transport = patched_async_client([_status_response(400)])
    rr = Reranker(base_url="https://example.test/v1", api_key="key")

    with pytest.raises(httpx.HTTPStatusError):
        await rr.rerank(query="q", documents=["a"], top_n=1)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_rerank_short_circuits_on_empty_documents(patched_async_client):
    transport = patched_async_client([])  # transport never called
    rr = Reranker(base_url="https://example.test/v1", api_key="key")

    assert await rr.rerank(query="q", documents=[], top_n=5) == []
    assert transport.calls == 0
