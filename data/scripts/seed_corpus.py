"""Seed Langfuse with corpus datasets.

One dataset per (subject, grade, semester). Each dataset item is one chapter.
Re-running is safe: any dataset that already exists on Langfuse is skipped
with a warning.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CORPUS_ROOT,
    configure_logging,
    corpus_dataset_name,
    dataset_exists,
    get_langfuse_client,
    parse_corpus_filename,
)


def main() -> None:
    logger = configure_logging()
    client = get_langfuse_client()

    groups: dict[tuple[str, str, str | None], list[Path]] = defaultdict(list)
    for subject_dir in sorted(p for p in CORPUS_ROOT.iterdir() if p.is_dir()):
        for txt_path in subject_dir.glob("*.txt"):
            parsed = parse_corpus_filename(txt_path.name)
            if not parsed:
                logger.warning("Unparsable corpus filename, skipping: %s", txt_path)
                continue
            key = (parsed["subject"], parsed["grade"], parsed["semester"])
            groups[key].append(txt_path)

    if not groups:
        logger.warning("No corpus files found under %s", CORPUS_ROOT)
        return

    for (subject, grade, semester), files in sorted(groups.items()):
        name = corpus_dataset_name(subject, grade, semester)

        if dataset_exists(client, name):
            logger.warning("Dataset '%s' already exists on Langfuse, skipping.", name)
            continue

        files.sort(key=lambda p: parse_corpus_filename(p.name)["chapter"])

        description = f"Corpus for {subject} grade {grade}"
        if semester:
            description += f" semester {semester}"
        client.create_dataset(name=name, description=description)

        for path in files:
            content = path.read_text(encoding="utf-8")
            client.create_dataset_item(
                dataset_name=name,
                input={"content": content},
                expected_output=None,
                metadata={"chapter": path.stem},
            )
            logger.info("Added chapter '%s' to dataset '%s'", path.stem, name)

        logger.info("Seeded dataset '%s' with %d chapter(s)", name, len(files))

    client.flush()


if __name__ == "__main__":
    main()
