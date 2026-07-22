"""Gradio admin panel for the RAG pipeline of a running FastAPI server.

Runs as a separate process from the API (`uv run poe admin-ui`, port 7861) and
drives the `/admin/rag/*` endpoints over HTTP — inspect the live config/status,
override the tunable knobs, force a rebuild, or reset to environment defaults.

The endpoints mutate the pipeline living *inside* the API process, so this UI is
a thin HTTP client (see `admin_ui.client`) rather than an in-process pipeline.
"""

import logging
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langfuse import Langfuse

from admin_ui.client import AdminAPIError, AdminClient, Config
from admin_ui.config import get_admin_ui_config
from eval_ui.config import get_eval_ui_config
from observability import init_langfuse_client
from rag.config import RAGConfig

logger = logging.getLogger(__name__)

# The 13 tunable config fields, in display order. The first four are free-text
# strings; the rest are integers (`chunk_overlap` allows 0, the others are > 0).
FIELD_ORDER = [
    "embedding_model",
    "rerank_model",
    "generator_system_prompt_name",
    "generator_user_prompt_name",
    "embedding_batch_size",
    "max_concurrency",
    "chunk_size",
    "chunk_overlap",
    "bm25_top_n",
    "dense_top_n",
    "rrf_k",
    "rerank_pool_size",
    "final_k",
]
STRING_FIELDS = {
    "embedding_model",
    "rerank_model",
    "generator_system_prompt_name",
    "generator_user_prompt_name",
}


def _error(message: str) -> str:
    return f'<span style="color: var(--color-red-600, #dc2626)"><b>Error:</b> {message}</span>'


def _success(message: str) -> str:
    return f'<span style="color: var(--color-green-600, #16a34a)"><b>✓</b> {message}</span>'


def _fallback_config() -> Config:
    """Env-derived defaults used as placeholders when the server is unreachable."""
    config = RAGConfig().model_dump()
    config["is_built"] = False
    config["corpus_datasets"] = []
    return config


def _status_md(config: Config) -> str:
    built = config.get("is_built")
    datasets = config.get("corpus_datasets") or []
    badge = "🟢 **Indexes built**" if built else "🔴 **Indexes not built**"
    joined = ", ".join(f"`{name}`" for name in datasets) if datasets else "_none_"
    return f"{badge} &nbsp;·&nbsp; Corpus datasets: {joined}"


def _read_form(values: list[Any]) -> Config:
    """Coerce the raw form values (in FIELD_ORDER) into an override-ready dict."""
    result: Config = {}
    for name, value in zip(FIELD_ORDER, values):
        if name in STRING_FIELDS:
            result[name] = (value or "").strip()
        else:
            result[name] = int(value)
    return result


def _form_updates(config: Config) -> list[Any]:
    return [gr.update(value=config[name]) for name in FIELD_ORDER]


def list_corpus_datasets(langfuse: Langfuse) -> list[tuple[str, str]]:
    """Return corpus dataset choices as (display_label, full_name) tuples, sorted.

    Mirrors ``eval_ui.main.list_dataset_names``: pages through the Langfuse
    datasets and keeps those under the configured corpus folder, showing the label
    without the folder prefix while passing the canonical name back as the value.
    Falls back to ``[]`` on any API failure so the UI still renders.
    """
    prefix = f"{get_eval_ui_config().corpus_dataset_folder}/"
    try:
        names: list[str] = []
        page = 1
        while True:
            response = langfuse.api.datasets.list(page=page, limit=100)
            names.extend(dataset.name for dataset in response.data)
            if page >= response.meta.total_pages:
                break
            page += 1
    except Exception:
        logger.exception("Failed to list Langfuse datasets")
        return []

    return sorted(
        (name.removeprefix(prefix), name) for name in names if name.startswith(prefix)
    )


def _merge_choices(
    choices: list[tuple[str, str]], selected: list[str]
) -> list[tuple[str, str]]:
    """Ensure every currently-selected corpus name appears as a choice.

    The pipeline may be built from datasets outside the corpus folder (or the
    listing may have failed), so append any selected names missing from the listed
    choices — otherwise the folder tree would have no row to un-check them from.
    Selected names outside the corpus folder keep their full name as the label so
    they still slot into the tree under their own path.
    """
    prefix = f"{get_eval_ui_config().corpus_dataset_folder}/"
    known = {full for _, full in choices}
    extra = [
        (full.removeprefix(prefix), full) for full in selected if full not in known
    ]
    return [*choices, *extra]


# A folder node in the corpus tree: named sub-folders plus the datasets that live
# directly at this level as (leaf_label, full_name) pairs.
CorpusTree = dict[str, Any]


def build_corpus_tree(choices: list[tuple[str, str]]) -> CorpusTree:
    """Group flat ``(label, full_name)`` choices into a nested folder tree.

    ``label`` is the corpus name with the folder prefix already stripped, so a
    ``/`` in it denotes nesting (``a/b/dataset`` → folder ``a`` → folder ``b`` →
    dataset ``dataset``). Labels with no ``/`` are datasets at the root. Returns a
    node ``{"folders": {name: node}, "datasets": [(leaf, full)]}``.
    """
    root: CorpusTree = {"folders": {}, "datasets": []}
    for label, full in choices:
        *folders, leaf = label.split("/")
        node = root
        for folder in folders:
            node = node["folders"].setdefault(folder, {"folders": {}, "datasets": []})
        node["datasets"].append((leaf, full))
    return root


def tree_dataset_names(node: CorpusTree) -> list[str]:
    """Every dataset full-name under ``node``, including nested folders."""
    names = [full for _, full in node["datasets"]]
    for child in node["folders"].values():
        names.extend(tree_dataset_names(child))
    return names


def apply_toggle(current: list[str], names: list[str], checked: bool) -> list[str]:
    """Add or remove ``names`` from the ``current`` selection, returned sorted.

    Shared by the per-dataset checkboxes (``names`` is one name) and the folder
    "select all" checkboxes (``names`` is every dataset under that folder).
    """
    chosen = set(current or [])
    chosen.update(names) if checked else chosen.difference_update(names)
    return sorted(chosen)


def build_demo(client: AdminClient, langfuse: Langfuse) -> gr.Blocks:
    try:
        initial_config, _ = client.get_config()
        initial_error: str | None = None
    except AdminAPIError as exc:
        initial_config = _fallback_config()
        initial_error = f"{exc} — showing defaults; click “Reload from server”."

    initial_corpus = initial_config.get("corpus_datasets") or []
    initial_choices = _merge_choices(list_corpus_datasets(langfuse), initial_corpus)

    def _synced(new_config: Config, feedback_html: str) -> list[Any]:
        selected = new_config.get("corpus_datasets") or []
        return [
            *_form_updates(new_config),
            selected,  # selected_state (drives the corpus tree re-render)
            gr.update(value=_status_md(new_config)),
            gr.update(value=feedback_html, visible=True),
            new_config,
        ]

    def _unchanged(feedback_html: str) -> list[Any]:
        # gr.skip() leaves an output untouched — required for the gr.State slots
        # (selected_state, snapshot_state), which take raw values, not gr.update().
        return [
            *[gr.skip() for _ in FIELD_ORDER],
            gr.skip(),  # corpus selection unchanged
            gr.skip(),  # status unchanged
            gr.update(value=feedback_html, visible=True),
            gr.skip(),  # snapshot unchanged
        ]

    def _use_endpoint(endpoint: str) -> None:
        """Point the shared client at the endpoint currently entered in the UI."""
        endpoint = (endpoint or "").strip()
        if endpoint and endpoint != client.base_url:
            client.set_base_url(endpoint)

    with gr.Blocks(title="sciedu-llm — RAG Admin") as demo:
        gr.Markdown("# sciedu-llm — RAG Admin")
        gr.Markdown(
            "Tune the live RAG pipeline. Retrieval knobs apply to the next query "
            "immediately; build-time fields take effect on the next rebuild."
        )

        with gr.Accordion("API endpoint", open=True):
            with gr.Row():
                endpoint_box = gr.Textbox(
                    label="API base URL",
                    value=client.base_url,
                    scale=4,
                    info="The FastAPI server this panel drives. No trailing /admin.",
                )
                test_btn = gr.Button("Test connection", scale=1)

        status_md = gr.Markdown(_status_md(initial_config))
        feedback = gr.Markdown(
            value=_error(initial_error) if initial_error else "",
            visible=bool(initial_error),
        )

        snapshot_state = gr.State(initial_config)
        selected_state = gr.State(initial_corpus)
        choices_state = gr.State(initial_choices)
        # Bumped by server ops (reload/reset/apply/…) to force the corpus tree to
        # re-render from the refreshed selection. The tree deliberately does NOT
        # re-render on selected_state, so per-checkbox toggles don't tear down and
        # recreate the box being clicked (which would swallow an un-check).
        tree_nonce_state = gr.State(0)
        inputs: dict[str, gr.components.Component] = {}

        # Build-time settings and retrieval knobs sit side by side (2:1 width).
        # The retrieval column is the narrower third, so its numbers stack rather
        # than crowd into a single row.
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Accordion("Build-time settings", open=True):
                    with gr.Row():
                        inputs["embedding_model"] = gr.Textbox(
                            label="Embedding model",
                            value=initial_config["embedding_model"],
                        )
                        inputs["rerank_model"] = gr.Textbox(
                            label="Rerank model",
                            value=initial_config["rerank_model"],
                        )
                    with gr.Row():
                        inputs["generator_system_prompt_name"] = gr.Textbox(
                            label="Generator system prompt name",
                            value=initial_config["generator_system_prompt_name"],
                        )
                        inputs["generator_user_prompt_name"] = gr.Textbox(
                            label="Generator user prompt name",
                            value=initial_config["generator_user_prompt_name"],
                        )
                    with gr.Row():
                        inputs["embedding_batch_size"] = gr.Number(
                            label="Embedding batch size",
                            value=initial_config["embedding_batch_size"],
                            precision=0,
                            minimum=1,
                        )
                        inputs["max_concurrency"] = gr.Number(
                            label="Max concurrency",
                            value=initial_config["max_concurrency"],
                            precision=0,
                            minimum=1,
                        )
                        inputs["chunk_size"] = gr.Number(
                            label="Chunk size",
                            value=initial_config["chunk_size"],
                            precision=0,
                            minimum=1,
                        )
                        inputs["chunk_overlap"] = gr.Number(
                            label="Chunk overlap",
                            value=initial_config["chunk_overlap"],
                            precision=0,
                            minimum=0,
                        )

            with gr.Column(scale=1):
                with gr.Accordion("Retrieval knobs", open=True):
                    inputs["bm25_top_n"] = gr.Number(
                        label="BM25 top N",
                        value=initial_config["bm25_top_n"],
                        precision=0,
                        minimum=1,
                    )
                    inputs["dense_top_n"] = gr.Number(
                        label="Dense top N",
                        value=initial_config["dense_top_n"],
                        precision=0,
                        minimum=1,
                    )
                    inputs["rrf_k"] = gr.Number(
                        label="RRF k",
                        value=initial_config["rrf_k"],
                        precision=0,
                        minimum=1,
                    )
                    inputs["rerank_pool_size"] = gr.Number(
                        label="Rerank pool size",
                        value=initial_config["rerank_pool_size"],
                        precision=0,
                        minimum=1,
                    )
                    inputs["final_k"] = gr.Number(
                        label="Final k",
                        value=initial_config["final_k"],
                        precision=0,
                        minimum=1,
                    )

        with gr.Accordion("Corpus datasets", open=True):
            gr.Markdown(
                "Select the Langfuse corpus datasets to index. Folders group nested "
                "dataset names (`a/b/dataset`); toggling a folder selects every "
                "dataset under it. Changing the selection rebuilds the indexes."
            )

            def _leaf_toggle(full: str):
                def _handler(checked: bool, current: list[str]) -> list[str]:
                    return apply_toggle(current, [full], checked)

                return _handler

            def _folder_toggle(names: list[str]):
                # Returns the new selection plus a value update per descendant leaf
                # so "select all" visually flips every box under the folder without
                # a full re-render (outputs = [selected_state, *leaf_boxes]).
                def _handler(checked: bool, current: list[str]) -> list[Any]:
                    updated = apply_toggle(current, names, checked)
                    return [updated, *(gr.update(value=checked) for _ in names)]

                return _handler

            # Re-render only when the dataset list changes or a server op bumps the
            # nonce — never on selected_state (see tree_nonce_state above).
            @gr.render(
                inputs=[choices_state, selected_state],
                triggers=[choices_state.change, tree_nonce_state.change],
            )
            def render_corpus_tree(choices, selected):
                selected_set = set(selected or [])
                tree = build_corpus_tree(choices or [])
                # full_name -> its leaf Checkbox, filled depth-first so a folder can
                # wire its "select all" to every descendant box created below it.
                leaf_boxes: dict[str, gr.Checkbox] = {}

                def render_node(node: CorpusTree) -> None:
                    for leaf, full in sorted(node["datasets"]):
                        box = gr.Checkbox(
                            label=leaf, value=full in selected_set, container=False
                        )
                        leaf_boxes[full] = box
                        # .input = user clicks only, so folder-driven value updates
                        # (below) don't re-fire this and double-toggle the state.
                        box.input(
                            _leaf_toggle(full),
                            inputs=[box, selected_state],
                            outputs=[selected_state],
                        )
                    for name in sorted(node["folders"]):
                        child = node["folders"][name]
                        under = tree_dataset_names(child)
                        chosen = sum(1 for n in under if n in selected_set)
                        with gr.Accordion(
                            f"📁 {name} ({chosen}/{len(under)} selected)", open=False
                        ):
                            all_box = gr.Checkbox(
                                label=f"Select all in {name}",
                                value=bool(under) and chosen == len(under),
                                container=False,
                            )
                            render_node(child)  # creates the descendant leaf boxes
                            all_box.input(
                                _folder_toggle(under),
                                inputs=[all_box, selected_state],
                                outputs=[selected_state, *(leaf_boxes[n] for n in under)],
                            )

                if not tree["datasets"] and not tree["folders"]:
                    gr.Markdown("_No corpus datasets found._")
                else:
                    render_node(tree)

            refresh_btn = gr.Button("Refresh dataset list")

        rebuild_checkbox = gr.Checkbox(
            value=True, label="Rebuild indexes after applying"
        )

        with gr.Row():
            apply_btn = gr.Button("Apply changes", variant="primary")
            rebuild_btn = gr.Button("Rebuild now")
            reset_btn = gr.Button("Reset to defaults")
            reload_btn = gr.Button("Reload from server")

        input_list = [inputs[name] for name in FIELD_ORDER]
        outputs = [*input_list, selected_state, status_md, feedback, snapshot_state]

        def on_test(endpoint):
            """Probe the entered endpoint and, if healthy, load its config."""
            _use_endpoint(endpoint)
            try:
                client.healthz()
            except AdminAPIError as exc:
                return _unchanged(_error(f"{client.base_url} is unhealthy: {exc}"))
            try:
                new_config, _ = client.get_config()
            except AdminAPIError as exc:
                # Reachable but RAG may be disabled — still a useful health signal.
                return _unchanged(_error(f"{client.base_url} reachable, but: {exc}"))
            return _synced(
                new_config, _success(f"Connected to {client.base_url} — config loaded.")
            )

        def on_apply(*args):
            *form_values, endpoint, corpus_sel, rebuild, snapshot = args
            _use_endpoint(endpoint)
            snapshot = snapshot or {}
            corpus_sel = corpus_sel or []
            current = _read_form(form_values)
            overrides = {
                name: current[name]
                for name in FIELD_ORDER
                if current[name] != snapshot.get(name)
            }
            corpus_changed = sorted(corpus_sel) != sorted(
                snapshot.get("corpus_datasets") or []
            )
            if corpus_changed and not corpus_sel:
                return _unchanged(_error("select at least one corpus dataset."))
            if corpus_changed:
                overrides["corpus_datasets"] = corpus_sel
            if not overrides and not rebuild:
                return _unchanged("Nothing to apply — no fields changed.")
            try:
                new_config, rebuilt = client.update_config(overrides, rebuild=rebuild)
            except AdminAPIError as exc:
                return _unchanged(_error(str(exc)))

            field_changes = len(overrides) - (1 if corpus_changed else 0)
            parts: list[str] = []
            if field_changes:
                noun = "change" if field_changes == 1 else "changes"
                parts.append(f"{field_changes} field {noun}")
            if corpus_changed:
                parts.append("corpus updated")
            message = f"Applied {', '.join(parts)}." if parts else "No changes."
            if rebuilt:
                message += " Indexes rebuilt."
            elif rebuild:
                message += " Rebuild requested."
            return _synced(new_config, _success(message))

        def on_rebuild(endpoint):
            _use_endpoint(endpoint)
            try:
                new_config, _ = client.rebuild()
            except AdminAPIError as exc:
                return _unchanged(_error(str(exc)))
            return _synced(new_config, _success("Indexes rebuilt."))

        def on_reset(endpoint):
            _use_endpoint(endpoint)
            try:
                new_config, _ = client.reset()
            except AdminAPIError as exc:
                return _unchanged(_error(str(exc)))
            return _synced(
                new_config, _success("Reset to environment defaults and rebuilt.")
            )

        def on_reload(endpoint):
            _use_endpoint(endpoint)
            try:
                new_config, _ = client.get_config()
            except AdminAPIError as exc:
                return _unchanged(_error(str(exc)))
            return _synced(new_config, _success("Loaded current config from server."))

        def on_refresh(current):
            listed = list_corpus_datasets(langfuse)
            choices = _merge_choices(listed, current or [])
            warning = (
                None
                if listed
                else "could not list datasets — check Langfuse credentials"
            )
            return (
                choices,
                gr.update(
                    value=_error(warning) if warning else "", visible=bool(warning)
                ),
            )

        def _bump(nonce: int) -> int:
            """Increment the re-render nonce so the corpus tree redraws from the
            selection a server op just loaded (selected_state is not a trigger)."""
            return (nonce or 0) + 1

        # Server ops load a fresh selection into selected_state; chain a nonce bump
        # so the tree re-renders to match. refresh_btn instead updates choices_state,
        # which already triggers the render on its own.
        for button, handler, extra_inputs in (
            (test_btn, on_test, []),
            (rebuild_btn, on_rebuild, []),
            (reset_btn, on_reset, []),
            (reload_btn, on_reload, []),
        ):
            button.click(
                handler, inputs=[endpoint_box, *extra_inputs], outputs=outputs
            ).then(_bump, inputs=[tree_nonce_state], outputs=[tree_nonce_state])

        apply_btn.click(
            on_apply,
            inputs=[
                *input_list,
                endpoint_box,
                selected_state,
                rebuild_checkbox,
                snapshot_state,
            ],
            outputs=outputs,
        ).then(_bump, inputs=[tree_nonce_state], outputs=[tree_nonce_state])
        refresh_btn.click(
            on_refresh, inputs=[selected_state], outputs=[choices_state, feedback]
        )

    return demo


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = get_admin_ui_config()
    client = AdminClient(config.api_base_url)
    langfuse = init_langfuse_client()
    demo = build_demo(client, langfuse)
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
