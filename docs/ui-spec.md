# Admin panel

This folder holds the admin panel: a separately built frontend, served on its
own, that talks to the sciedu-llm FastAPI service over HTTP. Nothing here is
imported by the Python service — the two are deployed independently.

## What it has to do

The required features are specified in
[`../docs/admin-ui-spec.md`](../docs/admin-ui-spec.md): RAG configuration,
preset management, and evaluation runs. That document is the source of truth;
it deliberately leaves visual design, navigation, and framework choice open.

## Backend contract

Everything the panel needs is the `/admin/*` API on the FastAPI service. The
live schema is at `/docs` (OpenAPI at `/openapi.json`) on a running instance.
The route groups:

| Group | What it covers |
| --- | --- |
| `GET/PATCH /admin/rag/config`, `POST /admin/rag/rebuild`, `POST /admin/rag/reset` | Read, override, rebuild, and reset the live RAG pipeline configuration. |
| `GET/PUT/DELETE /admin/presets`, `/admin/presets/{name}`, `POST /admin/presets/refresh` | List, read, author, and delete presets; refresh the registry and read the per-item load errors. |
| `POST/GET /admin/evals/runs`, `GET /admin/evals/runs/{id}`, `POST /admin/evals/runs/{id}/cancel`, `GET /admin/evals/history` | Launch, poll, and cancel evaluation runs; browse past runs recorded in Langfuse. |
| `GET /admin/models`, `GET /admin/datasets`, `GET /admin/judge-prompts`, `GET /admin/tools` | Lookup listings that populate the pickers above — including the tool registry a preset's `tools` may name. |

Two things to design around:

* **No authentication.** The deployment is fronted by Cloudflare, so the API
  has no login, no tokens, and no per-user state. Do not build a sign-in flow.
* **No database.** All durable state lives in Langfuse; RAG config overrides
  live only in the service's memory and are lost on restart, which the panel is
  required to make visible.
