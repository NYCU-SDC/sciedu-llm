"""Tests for the runtime-endpoint and health-probe additions to AdminClient."""

import httpx
import pytest

from admin_ui.client import AdminAPIError, AdminClient


def _client_with_transport(handler) -> AdminClient:
    client = AdminClient("http://localhost:8080")
    # Swap in a mock transport so requests never touch the network.
    client._client = httpx.Client(
        base_url=client.base_url, transport=httpx.MockTransport(handler)
    )
    return client


def test_set_base_url_strips_trailing_slash_and_rebinds():
    client = AdminClient("http://localhost:8080")
    client.set_base_url("http://api.example.com:9000/")
    assert client.base_url == "http://api.example.com:9000"
    # Subsequent requests go to the new host.
    assert str(client._client.base_url) == "http://api.example.com:9000"


def test_healthz_hits_healthz_and_returns_payload():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"status": "ok"})

    client = _client_with_transport(handler)
    assert client.healthz() == {"status": "ok"}
    assert seen["url"] == "http://localhost:8080/healthz"


def test_healthz_raises_admin_error_on_transport_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client_with_transport(handler)
    with pytest.raises(AdminAPIError, match="Cannot reach API server"):
        client.healthz()
