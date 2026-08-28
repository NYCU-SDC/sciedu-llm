"""Tests for `/admin/presets`.

The router is mounted on a bare ``FastAPI()`` rather than on ``app.main``: the
preset admin surface has no dependency on the rest of the app, and building the
whole app here would drag in a lifespan (and its startup requirements) that none
of these assertions care about.

The Langfuse stand-in below stores dataset items in a dict, so a PUT followed by
a registry refresh really does read back what was written — the write path and
the load path are exercised against the same fake, which is where the
interesting bugs (id reuse, shadowing, delete-reverts-to-builtin) live.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "mock_key")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langfuse.api.commons.errors.not_found_error import NotFoundError

from app.dependencies import get_langfuse_client, get_settings
from app.presets import DEFAULT_PRESETS, PresetRegistry
from app.routers.admin import presets as presets_router

DATASET = "config/presets"


def _item(id_: str, document):
    return SimpleNamespace(id=id_, input=document)


def _document(**overrides) -> dict:
    document = {
        "name": "socratic",
        "description": "asks questions back",
        "characters": [{"id": "assistant", "display_name": "助教"}],
    }
    document.update(overrides)
    return document


class _FakeLangfuse:
    """A Langfuse whose preset dataset is an in-memory list of items."""

    def __init__(self, items=(), *, dataset_exists: bool = True):
        self.datasets: dict[str, list] = (
            {DATASET: list(items)} if dataset_exists else {}
        )
        self.created_datasets: list[str] = []
        self.written: list[tuple[str, str, dict]] = []
        self.deleted: list[str] = []
        self.api = SimpleNamespace(
            dataset_items=SimpleNamespace(delete=self._delete_item)
        )

    def get_dataset(self, name):
        if name not in self.datasets:
            raise NotFoundError(body=f"no dataset '{name}'")
        return SimpleNamespace(items=list(self.datasets[name]))

    def create_dataset(self, *, name, description=None, **_kwargs):
        self.created_datasets.append(name)
        self.datasets.setdefault(name, [])

    def create_dataset_item(self, *, dataset_name, input, id=None, **_kwargs):
        self.written.append((dataset_name, id, input))
        items = self.datasets.setdefault(dataset_name, [])
        for index, existing in enumerate(items):
            if existing.id == id:
                # Langfuse upserts on a reused item id; mirror that here.
                items[index] = _item(id, input)
                return
        items.append(_item(id, input))

    def _delete_item(self, *, id):
        self.deleted.append(id)
        for name, items in self.datasets.items():
            self.datasets[name] = [item for item in items if item.id != id]


def _settings():
    return SimpleNamespace(
        presets_dataset_name=DATASET, presets_cache_ttl_seconds=300.0
    )


@pytest.fixture
def build_client():
    """Mount the presets router alone, over a scripted Langfuse."""

    def _build(langfuse=None, registry=None):
        langfuse = langfuse if langfuse is not None else _FakeLangfuse()
        settings = _settings()
        registry = (
            registry
            if registry is not None
            else PresetRegistry(langfuse=langfuse, settings=settings)
        )

        app = FastAPI()
        app.include_router(presets_router.router)
        app.state.preset_registry = registry
        app.dependency_overrides[get_langfuse_client] = lambda: langfuse
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app), langfuse, registry

    return _build


# --- listing ----------------------------------------------------------------


def test_list_reports_the_code_defaults_as_builtin_and_unshadowed(build_client):
    client, _langfuse, _registry = build_client()

    body = client.get("/presets").json()

    assert [entry["name"] for entry in body] == sorted(DEFAULT_PRESETS)
    assert all(entry["builtin"] for entry in body)
    assert not any(entry["shadowed_builtin"] for entry in body)


def test_list_marks_a_dataset_preset_as_neither_builtin_nor_shadowing(build_client):
    langfuse = _FakeLangfuse([_item("socratic", _document())])
    client, _langfuse, registry = build_client(langfuse)
    client.post("/presets/refresh")

    body = {entry["name"]: entry for entry in client.get("/presets").json()}

    assert body["socratic"] == {
        "name": "socratic",
        "description": "asks questions back",
        "builtin": False,
        "shadowed_builtin": False,
    }
    assert registry.names() == sorted([*DEFAULT_PRESETS, "socratic"])


def test_list_marks_a_shadowed_builtin(build_client):
    langfuse = _FakeLangfuse(
        [_item("default-chat", _document(name="default-chat", description="tuned"))]
    )
    client, _langfuse, _registry = build_client(langfuse)
    client.post("/presets/refresh")

    body = {entry["name"]: entry for entry in client.get("/presets").json()}

    assert body["default-chat"]["builtin"] is True
    assert body["default-chat"]["shadowed_builtin"] is True
    assert body["default-chat"]["description"] == "tuned"
    # Shadowing replaces rather than adds.
    assert sorted(body) == sorted(DEFAULT_PRESETS)


# --- get --------------------------------------------------------------------


def test_get_returns_the_round_trippable_document(build_client):
    client, _langfuse, _registry = build_client()

    body = client.get("/presets/default-chat-plain").json()

    assert body["builtin"] is True
    assert body["definition"]["name"] == "default-chat-plain"
    assert body["definition"]["max_steps"] == 1
    # The definition is exactly what PUT accepts back.
    assert (
        client.put("/presets/default-chat-plain", json=body["definition"]).status_code
        == 200
    )


def test_get_unknown_preset_is_404(build_client):
    client, _langfuse, _registry = build_client()

    response = client.get("/presets/nope")

    assert response.status_code == 404
    assert "Unknown preset 'nope'" in response.json()["detail"]


# --- upsert -----------------------------------------------------------------


def test_put_writes_the_item_and_serves_it_immediately(build_client):
    client, langfuse, registry = build_client()

    response = client.put("/presets/socratic", json=_document())

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "socratic"
    assert body["builtin"] is False
    assert body["definition"]["max_steps"] == 8
    assert langfuse.written == [(DATASET, "socratic", body["definition"])]
    # Refreshed inside the request, so the preset is live already.
    assert registry.names() == sorted([*DEFAULT_PRESETS, "socratic"])


def test_put_reuses_the_item_id_so_an_edit_upserts(build_client):
    client, langfuse, registry = build_client()

    client.put("/presets/socratic", json=_document())
    client.put("/presets/socratic", json=_document(description="reworded"))

    assert [item_id for _dataset, item_id, _doc in langfuse.written] == [
        "socratic",
        "socratic",
    ]
    assert len(langfuse.datasets[DATASET]) == 1
    assert registry.snapshot()["socratic"].description == "reworded"


def test_put_creates_the_dataset_when_it_does_not_exist_yet(build_client):
    langfuse = _FakeLangfuse(dataset_exists=False)
    client, _langfuse, _registry = build_client(langfuse)

    assert client.put("/presets/socratic", json=_document()).status_code == 200
    assert langfuse.created_datasets == [DATASET]


def test_put_can_shadow_a_builtin(build_client):
    client, _langfuse, registry = build_client()

    response = client.put(
        "/presets/default-chat",
        json=_document(name="default-chat", description="tuned"),
    )

    assert response.status_code == 200
    assert response.json()["shadowed_builtin"] is True
    assert registry.snapshot()["default-chat"].description == "tuned"


def test_put_rejects_an_invalid_document_before_writing(build_client):
    client, langfuse, _registry = build_client()

    response = client.put("/presets/socratic", json={"name": "socratic"})

    assert response.status_code == 422
    assert langfuse.written == []


def test_put_surfaces_the_semantic_validation_error(build_client):
    client, langfuse, _registry = build_client()

    response = client.put(
        "/presets/socratic",
        json=_document(
            characters=[
                {"id": "assistant", "display_name": "助教", "tools": ["nonesuch"]}
            ]
        ),
    )

    assert response.status_code == 422
    assert "unknown tool(s): nonesuch" in str(response.json()["detail"])
    assert langfuse.written == []


def test_put_rejects_a_document_whose_name_differs_from_the_path(build_client):
    client, langfuse, _registry = build_client()

    response = client.put("/presets/socratic", json=_document(name="other"))

    assert response.status_code == 422
    assert "does not match the path name" in response.json()["detail"]
    assert langfuse.written == []


def test_put_reports_a_failed_write_as_a_bad_gateway(build_client):
    langfuse = _FakeLangfuse()

    def _boom(**_kwargs):
        raise RuntimeError("langfuse is down")

    langfuse.create_dataset_item = _boom
    client, _langfuse, _registry = build_client(langfuse)

    response = client.put("/presets/socratic", json=_document())

    assert response.status_code == 502
    assert "Failed to write preset 'socratic'" in response.json()["detail"]


# --- delete -----------------------------------------------------------------


def test_delete_removes_the_dataset_item(build_client):
    langfuse = _FakeLangfuse([_item("socratic", _document())])
    client, _langfuse, registry = build_client(langfuse)
    client.post("/presets/refresh")

    response = client.delete("/presets/socratic")

    assert response.status_code == 204
    assert langfuse.deleted == ["socratic"]
    assert registry.names() == sorted(DEFAULT_PRESETS)


def test_deleting_a_shadowing_preset_puts_the_code_default_back(build_client):
    langfuse = _FakeLangfuse(
        [_item("default-chat", _document(name="default-chat", description="tuned"))]
    )
    client, _langfuse, registry = build_client(langfuse)
    client.post("/presets/refresh")
    assert registry.snapshot()["default-chat"].description == "tuned"

    assert client.delete("/presets/default-chat").status_code == 204

    assert registry.snapshot()["default-chat"] is DEFAULT_PRESETS["default-chat"]
    assert client.get("/presets/default-chat").json()["shadowed_builtin"] is False


def test_delete_of_an_unseeded_code_default_is_409(build_client):
    client, langfuse, _registry = build_client()

    response = client.delete("/presets/default-chat")

    assert response.status_code == 409
    assert "is built in and cannot be deleted" in response.json()["detail"]
    assert langfuse.deleted == []


def test_delete_of_an_unknown_preset_is_404(build_client):
    client, langfuse, _registry = build_client()

    response = client.delete("/presets/nope")

    assert response.status_code == 404
    assert langfuse.deleted == []


def test_delete_finds_an_item_whose_id_is_not_the_preset_name(build_client):
    # Hand-created in the Langfuse UI: arbitrary item id, correct document name.
    langfuse = _FakeLangfuse([_item("cm-generated-id", _document())])
    client, _langfuse, _registry = build_client(langfuse)

    assert client.delete("/presets/socratic").status_code == 204
    assert langfuse.deleted == ["cm-generated-id"]


# --- refresh ----------------------------------------------------------------


def test_refresh_reports_what_is_served_and_what_failed(build_client):
    langfuse = _FakeLangfuse(
        [
            _item("good", _document()),
            _item("broken", _document(name="broken", orchestrator="nobody")),
        ]
    )
    client, _langfuse, _registry = build_client(langfuse)

    body = client.post("/presets/refresh").json()

    assert body["loaded"] == sorted([*DEFAULT_PRESETS, "socratic"])
    assert "is not one of the characters" in body["errors"]["broken"]
    assert body["fetched_at"] is not None


def test_refresh_keeps_serving_the_builtins_when_langfuse_is_down(build_client):
    langfuse = _FakeLangfuse(dataset_exists=False)

    def _boom(_name, **_kwargs):
        raise RuntimeError("langfuse is down")

    langfuse.get_dataset = _boom
    client, _langfuse, _registry = build_client(langfuse)

    body = client.post("/presets/refresh").json()

    assert body["loaded"] == sorted(DEFAULT_PRESETS)
    assert body["fetched_at"] is None
    assert "langfuse is down" in "".join(body["errors"].values())


# --- registry wiring --------------------------------------------------------


def test_the_registry_is_built_on_demand_when_the_lifespan_left_none(build_client):
    langfuse = _FakeLangfuse()
    app = FastAPI()
    app.include_router(presets_router.router)
    app.dependency_overrides[get_langfuse_client] = lambda: langfuse
    app.dependency_overrides[get_settings] = _settings
    client = TestClient(app)

    response = client.get("/presets")

    assert response.status_code == 200
    assert [entry["name"] for entry in response.json()] == sorted(DEFAULT_PRESETS)
