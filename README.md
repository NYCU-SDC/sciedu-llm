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
| `uv run poe lint`                    | Run ESLint code analysis        |
| `uv run poe format`                  | Format code with Ruff           |
