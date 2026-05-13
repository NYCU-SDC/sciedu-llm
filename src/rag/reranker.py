import httpx

from rag.retry import (
    RETRYABLE_HTTPX_EXCEPTIONS,
    is_retryable_http_error,
    with_openai_retry,
)


class Reranker:
    """Async client for an OpenAI-compatible `/rerank` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "BGE-Reranker-V2-M3",
        timeout: float = 30.0,
    ) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._timeout = timeout

    async def rerank(
        self, *, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "top_n": top_n,
            "documents": documents,
        }
        data = await self._post(payload)
        return [
            (result["index"], result["relevance_score"]) for result in data["results"]
        ]

    @with_openai_retry(
        retry_on=RETRYABLE_HTTPX_EXCEPTIONS,
        should_retry=is_retryable_http_error,
    )
    async def _post(self, payload: dict) -> dict:
        async def _send() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                return await client.post(self._url, headers=self._headers, json=payload)

        response = await _send()

        response.raise_for_status()
        return response.json()
