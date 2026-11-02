# SciEdu LLM

## Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```shell
# Clone the repository
git clone https://github.com/NYCU-SDC/sciedu-llm.git
cd sciedu-llm

# Install dependencies
uv sync

# Start development server
uv run poe dev
```

The application will be available at `http://localhost:8080`.

### Development Scripts

| Command                              | Description                     |
| ------------------------------------ | ------------------------------- |
| `uv run poe dev`                     | Start dev server with hot reload |
| `uv run poe test`                    | Run tests                       |
| `uv run poe lint`                    | Run Ruff code analysis          |
| `uv run poe format`                  | Format code with Ruff           |

### Administration

Everything an operator can change at runtime — RAG configuration, presets, and
evaluation runs — is served by the same FastAPI process under `/admin`, and is
browsable at `http://localhost:8080/docs`. The admin panel that drives it is a
separately built frontend living in `ui/`; see [`docs/admin-ui-spec.md`](docs/admin-ui-spec.md)
for the features it must cover.

### Preset RAG indexes

Choose the corpus in each preset's `tools.rag.corpus_datasets`, for example:

```json
"rag": {
  "enable_tool": true,
  "force": false,
  "corpus_datasets": ["corpus/biology-10"]
}
```

The preset editor provides a corpus picker. RAG settings remain global on the
retrieval settings page, which also lets you select a preset to inspect,
rebuild, or cancel its index build. The API uses
`GET /admin/rag/config?preset=<name>` and the same query parameter for
`POST /admin/rag/rebuild` and `POST /admin/rag/rebuild/cancel`. Omitting it
selects `AGENTS_DEFAULT_PRESET`. `PATCH /admin/rag/config` and
`POST /admin/rag/reset` update global settings; they never change preset corpora.

Indexes are cached in memory for the life of the server process. The cache key
contains the sorted, deduplicated corpus names, embedding model, embedding
batch size, chunk size, and overlap. Matching presets share completed indexes
and in-flight builds. Retrieval settings apply globally without re-embedding;
changing build settings selects a different cache entry. Reverting settings
can reuse an earlier entry. Startup, preset saves, and preset refreshes schedule
missing builds in the background. Requests also schedule missing entries and
return 503 until their preset's index is ready. Failed or cancelled builds need
an explicit rebuild; polling does not retry them. A forced rebuild refreshes a
shared entry after dataset contents change in Langfuse. Cancelling that build
affects every preset sharing it; other entries keep running.

**Migration:** `RAG_CORPUS_DATASETS` and automatic corpus-folder discovery no
longer select retrieval data. Existing preset documents still load, with an
empty corpus by default. Select and save the intended datasets for each
RAG-enabled preset, including `default-chat` and `default-agents`, before using
retrieval. An empty corpus never falls back to another preset's index.

### Seeding Langfuse

The service reads its corpus, questions, and presets out of Langfuse. The
seeders under `data/scripts/` push the contents of `data/` up to the configured
project:

```shell
uv run python data/scripts/seed_corpus.py
uv run python data/scripts/seed_questions.py
uv run python data/scripts/seed_presets.py
```

The presets the service ships with (`default-agents`, `default-chat`,
`default-chat-plain`) seed themselves: startup writes any of them the
`config/presets` dataset does not already hold, and never touches one that is
already there. What `seed_presets.py` is still needed for is the Langfuse
**prompts** those presets reference — it has to have run at least once before
`default-agents` can serve a request, because without `agents/teacher-system`
and `agents/student` in Langfuse the agentic endpoint answers 502 — plus the
example presets under `data/presets/` that the service does not ship
(`teacher-student`, `rag-agent`). Pass `--dry-run` to validate every file under
`data/presets/` without contacting Langfuse.
