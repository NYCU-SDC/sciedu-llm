"""Thin synchronous HTTP client for the FastAPI admin RAG endpoints.

The admin router mutates the *live* pipeline inside the running FastAPI process,
so the UI must reach it over HTTP rather than building its own pipeline. Handlers
in `admin_ui.main` are synchronous, so a blocking `httpx.Client` keeps things
simple and sidesteps event-loop concerns.

Every call returns a normalized ``(config, rebuilt)`` tuple: ``config`` is the
``RAGConfigResponse`` dict (the 13 tunable fields plus ``is_built`` /
``corpus_datasets``) and ``rebuilt`` is whether the call triggered an index
rebuild (``None`` for endpoints that don't report it). Failures surface as
``AdminAPIError`` carrying a human-readable message.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# A RAGConfigResponse payload: the 13 tunable fields (str/int) plus `is_built`
# (bool) and `corpus_datasets` (list[str]) — hence heterogeneous values.
Config = dict[str, Any]


class AdminAPIError(Exception):
    """A user-facing failure talking to the admin API (transport or HTTP error)."""


class AdminClient:
    def __init__(self, base_url: str) -> None:
        # Rebuilds re-embed the whole corpus, so allow a generous read timeout.
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_base_url(self, base_url: str) -> None:
        """Point subsequent requests at a different API server.

        Lets the admin panel switch the endpoint it drives at runtime (see the
        endpoint controls in ``admin_ui.main``) without rebuilding the client.
        """
        self._base_url = base_url.rstrip("/")
        self._client.base_url = self._base_url

    def healthz(self) -> Config:
        """Probe the server's ``/healthz`` liveness endpoint.

        Uses a short timeout so an unreachable/misconfigured endpoint fails fast
        rather than waiting out the rebuild-sized read timeout.
        """
        return self._request("GET", "/healthz", timeout=httpx.Timeout(5.0))

    def get_config(self) -> tuple[Config, bool | None]:
        return self._normalize(self._request("GET", "/admin/rag/config"))

    def update_config(
        self, overrides: Config, *, rebuild: bool
    ) -> tuple[Config, bool | None]:
        body = {**overrides, "rebuild": rebuild}
        return self._normalize(self._request("PATCH", "/admin/rag/config", json=body))

    def rebuild(self) -> tuple[Config, bool | None]:
        return self._normalize(self._request("POST", "/admin/rag/rebuild"))

    def reset(self) -> tuple[Config, bool | None]:
        return self._normalize(self._request("POST", "/admin/rag/reset"))

    @staticmethod
    def _normalize(payload: Config) -> tuple[Config, bool | None]:
        """Flatten both response shapes to ``(config_dict, rebuilt_or_None)``.

        ``GET``/``rebuild`` return the config directly; ``PATCH``/``reset`` wrap it
        as ``{config: {...}, rebuilt: bool}``.
        """
        if "config" in payload:
            return payload["config"], payload.get("rebuilt")
        return payload, None

    def _request(self, method: str, path: str, **kwargs: Any) -> Config:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            logger.warning("Admin API request failed: %s %s (%s)", method, path, exc)
            raise AdminAPIError(
                f"Cannot reach API server at {self._base_url} ({exc})"
            ) from exc

        if response.is_success:
            return response.json()

        raise AdminAPIError(self._describe_error(response))

    @staticmethod
    def _describe_error(response: httpx.Response) -> str:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        detail = detail or response.text or "no detail"

        prefixes = {
            422: "Invalid config value",
            502: "Rebuild failed",
            503: "RAG not enabled on the server",
        }
        prefix = prefixes.get(response.status_code)
        if prefix:
            return f"{prefix}: {detail}"
        return f"API error ({response.status_code}): {detail}"
