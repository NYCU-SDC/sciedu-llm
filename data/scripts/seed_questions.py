"""Seed Langfuse with question datasets, one per subject.

Only rows whose 標記狀態 is "成功" are uploaded; counts of any filtered rows
are reported. ref_text and ref_text_coords are stored as JSON-stringified
lists. Re-running is safe: any dataset that already exists on Langfuse is
skipped with a warning.
"""

import json
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    QUESTIONS_ROOT,
    configure_logging,
    dataset_exists,
    get_langfuse_client,
)

QUESTION_COL = "題目內容"
ANSWER_COL = "答案分析"
REF_TEXT_COL = "參考段落"
REF_COORDS_COL = "絕對座標"
STATUS_COL = "標記狀態"
STATUS_OK = "成功"


def split_ref_text(raw: str) -> list[str]:
    """Split ref text on lines containing only '---'."""
    sections: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        if line.strip() == "---":
            if current:
                sections.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


COORD_PATTERN = re.compile(r"^(?P<source>.+?)\.txt\((?P<start>\d+)-(?P<end>\d+)\)$")


def split_coords(raw: str, logger) -> list[dict]:
    """Parse "source_file_name.txt(start_num-end_num); ..." into structured coord dicts.

    Each entry becomes {"source": "source_file_name", "coords": [start, end]}.
    Malformed entries are logged and skipped.
    """
    parsed: list[dict] = []
    for raw_entry in raw.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        match = COORD_PATTERN.match(entry)
        if not match:
            logger.warning("Unparsable ref_text_coord entry, skipping: %r", entry)
            continue
        parsed.append(
            {
                "source": match.group("source"),
                "coords": [int(match.group("start")), int(match.group("end"))],
            }
        )
    return parsed


def parse_xlsx(path: Path, logger) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    headers = [str(h) if h is not None else "" for h in next(rows)]

    required = [QUESTION_COL, ANSWER_COL, REF_TEXT_COL, REF_COORDS_COL, STATUS_COL]
    try:
        idx = {col: headers.index(col) for col in required}
    except ValueError as e:
        raise RuntimeError(f"Missing required column in {path.name}: {e}") from e

    items: list[dict] = []
    skipped_status = 0
    skipped_empty = 0
    for row in rows:
        question = row[idx[QUESTION_COL]]
        if question is None or not str(question).strip():
            skipped_empty += 1
            continue

        status = row[idx[STATUS_COL]]
        if status is None or str(status).strip() != STATUS_OK:
            skipped_status += 1
            continue

        items.append(
            {
                "question": str(question).strip(),
                "gold_answer": str(row[idx[ANSWER_COL]] or "").strip(),
                "ref_text": split_ref_text(str(row[idx[REF_TEXT_COL]] or "")),
                "ref_text_coords": split_coords(
                    str(row[idx[REF_COORDS_COL]] or ""), logger
                ),
            }
        )

    if skipped_empty:
        logger.warning(
            "%s: skipped %d row(s) with empty question", path.name, skipped_empty
        )
    if skipped_status:
        logger.warning(
            "%s: skipped %d row(s) where %s != %r",
            path.name,
            skipped_status,
            STATUS_COL,
            STATUS_OK,
        )
    return items


def main() -> None:
    logger = configure_logging()
    client = get_langfuse_client()

    xlsx_files = sorted(QUESTIONS_ROOT.glob("*_questions.xlsx"))
    if not xlsx_files:
        logger.warning("No question files found under %s", QUESTIONS_ROOT)
        return

    for xlsx_path in xlsx_files:
        subject = xlsx_path.stem.removesuffix("_questions")
        name = f"questions-{subject}"

        if dataset_exists(client, name):
            logger.warning("Dataset '%s' already exists on Langfuse, skipping.", name)
            continue

        items = parse_xlsx(xlsx_path, logger)
        if not items:
            logger.warning(
                "No usable rows in %s, skipping dataset creation.", xlsx_path.name
            )
            continue

        client.create_dataset(
            name=name,
            description=f"Benchmark questions for {subject}",
        )
        for item in items:
            client.create_dataset_item(
                dataset_name=name,
                input={"question": item["question"]},
                expected_output={
                    "gold_answer": item["gold_answer"],
                    "ref_text": json.dumps(item["ref_text"], ensure_ascii=False),
                },
                metadata={
                    "ref_text_coords": json.dumps(
                        item["ref_text_coords"], ensure_ascii=False
                    ),
                },
            )
        logger.info("Seeded dataset '%s' with %d question(s)", name, len(items))

    client.flush()


if __name__ == "__main__":
    main()
