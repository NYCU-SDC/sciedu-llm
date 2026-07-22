"""Tests for the corpus-folder tree helpers backing the admin UI."""

from admin_ui.main import (
    _merge_choices,
    apply_toggle,
    build_corpus_tree,
    tree_dataset_names,
)


def _choices() -> list[tuple[str, str]]:
    # (prefix-stripped label, canonical Langfuse name) — nesting via `/`.
    return [
        ("alpha", "corpus/alpha"),
        ("law/criminal", "corpus/law/criminal"),
        ("law/civil/torts", "corpus/law/civil/torts"),
        ("law/civil/contracts", "corpus/law/civil/contracts"),
    ]


def test_build_corpus_tree_nests_folders_by_slash():
    tree = build_corpus_tree(_choices())

    # Root has the un-nested dataset and one top-level folder.
    assert tree["datasets"] == [("alpha", "corpus/alpha")]
    assert sorted(tree["folders"]) == ["law"]

    law = tree["folders"]["law"]
    assert law["datasets"] == [("criminal", "corpus/law/criminal")]
    assert sorted(law["folders"]) == ["civil"]

    civil = law["folders"]["civil"]
    assert sorted(civil["datasets"]) == [
        ("contracts", "corpus/law/civil/contracts"),
        ("torts", "corpus/law/civil/torts"),
    ]
    assert civil["folders"] == {}


def test_build_corpus_tree_empty():
    assert build_corpus_tree([]) == {"folders": {}, "datasets": []}


def test_tree_dataset_names_collects_all_descendants():
    tree = build_corpus_tree(_choices())
    assert sorted(tree_dataset_names(tree)) == [
        "corpus/alpha",
        "corpus/law/civil/contracts",
        "corpus/law/civil/torts",
        "corpus/law/criminal",
    ]
    # A sub-folder only reports its own subtree.
    civil = tree["folders"]["law"]["folders"]["civil"]
    assert sorted(tree_dataset_names(civil)) == [
        "corpus/law/civil/contracts",
        "corpus/law/civil/torts",
    ]


def test_apply_toggle_adds_and_removes_and_sorts():
    # Selecting a whole folder's datasets, then de-selecting one leaf.
    under_civil = ["corpus/law/civil/contracts", "corpus/law/civil/torts"]
    selected = apply_toggle(["corpus/alpha"], under_civil, checked=True)
    assert selected == [
        "corpus/alpha",
        "corpus/law/civil/contracts",
        "corpus/law/civil/torts",
    ]

    selected = apply_toggle(selected, ["corpus/law/civil/torts"], checked=False)
    assert selected == ["corpus/alpha", "corpus/law/civil/contracts"]


def test_apply_toggle_is_idempotent_and_handles_empty_current():
    names = ["corpus/a", "corpus/b"]
    assert apply_toggle(None, names, checked=True) == names
    # Re-adding already-selected names does not duplicate them.
    assert apply_toggle(names, names, checked=True) == names
    # Removing names not present is a no-op.
    assert apply_toggle(["corpus/a"], ["corpus/z"], checked=False) == ["corpus/a"]


def test_merge_choices_appends_selected_outside_listing(monkeypatch):
    listed = [("alpha", "corpus/alpha")]
    # A built-from name absent from the listing (e.g. outside the corpus folder,
    # or the listing failed) must still get a tree row so it can be un-checked.
    merged = _merge_choices(listed, ["corpus/alpha", "corpus/law/criminal"])
    assert sorted(merged) == [
        ("alpha", "corpus/alpha"),
        ("law/criminal", "corpus/law/criminal"),
    ]

    # A selected name entirely outside the corpus folder keeps its full name as
    # the label (only the `corpus/` prefix is stripped).
    merged = _merge_choices([], ["other/thing"])
    assert merged == [("other/thing", "other/thing")]
