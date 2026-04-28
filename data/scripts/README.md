# Data seeding scripts

These scripts upload the lab's corpus and benchmark questions to Langfuse as
datasets. The actual `data/corpus/` and `data/questions/` files are **not**
checked into this repository because of textbook/question-bank copyright. You
must drop the files into place locally before running anything here.

## Required environment

Set in `.env` at the repo root (or as real env vars):

- `LANGFUSE_BASE_URL`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`

## Expected directory layout

```
data/
├── corpus/
│   ├── biology/
│   │   ├── biology_10_ch1.txt
│   │   ├── biology_7_1_ch1.txt
│   │   └── ...
│   ├── chemical/
│   │   └── chemical_10_chN.txt
│   └── physical/
│       └── physical_<grade>[_<semester>]_chN.txt
└── questions/
    ├── biology_questions.xlsx
    ├── chemical_questions.xlsx
    └── physical_questions.xlsx
```

### Corpus text files

- One plain-text file per chapter, UTF-8.
- Filename pattern: `<subject>_<grade>[_<semester>]_ch<N>.txt`.
  - `subject` is lowercase ASCII (`biology`, `chemical`, `physical`).
  - `grade` is an integer (e.g. `7`, `8`, `9`, `10`).
  - `semester` is optional and only present for grades that span two
    semester volumes (e.g. `physical_9_1_ch2.txt`). Single-volume grades
    omit it (e.g. `physical_10_ch1.txt`).
  - `N` is an integer chapter number.
- Files that don't match this pattern are skipped with a warning.

### Question xlsx files

One workbook per subject, named `<subject>_questions.xlsx`. The first row is
the header. Required columns (Traditional Chinese):

| Column     | Meaning                                                           |
| ---------- | ----------------------------------------------------------------- |
| `題目內容` | Question text                                                     |
| `答案分析` | Reference / gold answer                                           |
| `參考段落` | Source passages, multiple sections separated by a line of `---`   |
| `絕對座標` | Source spans, formatted `<source>.txt(<start>-<end>)`, joined by `;` |
| `標記狀態` | Row status; only rows equal to `成功` are uploaded                |

Example `絕對座標` cell:

```
physical_10_ch1.txt(2674-2761); physical_10_ch1.txt(2794-2862); physical_10_ch1.txt(4383-4437)
```

The `<source>` portion (filename without `.txt`) must match the corpus file
stem so the RAG pipeline can later resolve coords to chunk IDs.

## What gets created on Langfuse

Both scripts skip with a warning if a dataset of the same name already exists.

### `seed_corpus.py`

One dataset per `(subject, grade, semester)` group:

- Dataset name: `corpus-<subject>-<grade>` or `corpus-<subject>-<grade>-<semester>`.
- One item per chapter file:
  - `input = {"content": <full file text>}`
  - `expected_output = None`
  - `metadata = {"chapter": "<filename stem>"}` (e.g. `physical_10_ch1`)

### `seed_questions.py`

One dataset per subject (`questions-<subject>`). Per row:

- `input = {"question": <題目內容>}`
- `expected_output = {"gold_answer": <答案分析>, "ref_text": <JSON string>}`
- `metadata = {"ref_text_coords": <JSON string>}`

`ref_text` is a JSON-stringified list of passage strings split on the `---`
delimiter. `ref_text_coords` is a JSON-stringified list of
`{"source": "<filename stem>", "coords": [start, end]}` objects.

Rows are filtered to `標記狀態 == "成功"`; the count of dropped rows is logged
as a warning. Coord entries that don't match the expected pattern are also
logged and skipped individually.

## Running

From the repo root:

```bash
uv run python data/scripts/seed_corpus.py
uv run python data/scripts/seed_questions.py
```

The scripts call `langfuse.flush()` at the end to ensure all queued writes
are sent before exit.
