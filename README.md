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
