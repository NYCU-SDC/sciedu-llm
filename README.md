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
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

The application will be available at `http://localhost:8080`.

### Development Scripts

| Command                              | Description                     |
| ------------------------------------ | ------------------------------- |
| `uv run uvicorn src.main:app --reload` | Start dev server with hot reload |
| `uv run pytest`                      | Run tests                       |
| `uv run ruff check .`                | Run ESLint code analysis        |
| `uv run ruff format .`               | Format code with Ruff           |
