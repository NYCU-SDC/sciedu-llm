"""Seed Langfuse with corpus datasets.

One dataset per (subject, grade, semester). Each dataset item is one chapter.
Re-running is safe: any dataset that already exists on Langfuse is skipped
with a warning, unless `--overwrite` is passed (in which case existing items
are cleared and the dataset is re-populated).
"""

from collections import defaultdict
from pathlib import Path

from _common import (  # noqa: E402
    CORPUS_ROOT,
    clear_dataset_items,
    configure_logging,
    corpus_dataset_name,
    dataset_exists,
    get_langfuse_client,
    parse_args,
    parse_corpus_filename,
    retry_on_transport_error,
)


def main() -> None:
    args = parse_args("Seed Langfuse corpus datasets from data/corpus/")
    logger = configure_logging()
    client = get_langfuse_client()

    groups: dict[tuple[str, str, str | None], list[tuple[int, Path]]] = defaultdict(
        list
    )
    for subject_dir in sorted(p for p in CORPUS_ROOT.iterdir() if p.is_dir()):
        for txt_path in subject_dir.glob("*.txt"):
            parsed = parse_corpus_filename(txt_path.name)
            if not parsed:
                logger.warning("Unparsable corpus filename, skipping: %s", txt_path)
                continue
            key = (parsed["subject"], parsed["grade"], parsed["semester"])
            groups[key].append((parsed["chapter"], txt_path))

    if not groups:
        logger.warning("No corpus files found under %s", CORPUS_ROOT)
        return

    for (subject, grade, semester), entries in sorted(groups.items()):
        name = corpus_dataset_name(subject, grade, semester)

        entries.sort(key=lambda e: e[0])

        description = f"Corpus for {subject} grade {grade}"
        if semester:
            description += f" semester {semester}"

        if dataset_exists(client, name):
            if not args.overwrite:
                logger.warning(
                    "Dataset '%s' already exists on Langfuse, skipping. Pass --overwrite to re-seed.",
                    name,
                )
                continue
            clear_dataset_items(client, name, logger)
        else:
            client.create_dataset(name=name, description=description)

        for _chapter, path in entries:
            content = path.read_text(encoding="utf-8")
            item_id = f"{name}-{path.stem}"
            retry_on_transport_error(
                lambda content=content,
                path=path,
                item_id=item_id: client.create_dataset_item(
                    id=item_id,
                    dataset_name=name,
                    input={"content": content},
                    expected_output=None,
                    metadata={"chapter": path.stem},
                ),
                logger=logger,
            )
            logger.info("Added chapter '%s' to dataset '%s'", path.stem, name)

        logger.info("Seeded dataset '%s' with %d chapter(s)", name, len(entries))

    client.flush()


if __name__ == "__main__":
    main()
