# sciedu-llm — service console

The admin panel for the sciedu-llm service: a Vite + React 19 + TypeScript app
that talks to the backend's `/admin` API over HTTP. There is no authentication
here — the deployment is fronted by Cloudflare — and no state of its own; every
screen reads and writes the live service.

It implements `docs/admin-ui-spec.md` and follows the approved mockup in
`ui-mockup/LLM Admin.dc.html`, built on the Organic design system vendored at
`src/styles/organic.css` (copied verbatim from
`ui-mockup/_ds/organic-.../styles.css`; `src/styles/app.css` layers this app's
own overrides on top of it, starting with the mockup's `<style>` block).

## Running it

```bash
pnpm install
pnpm dev            # http://localhost:5173
```

`pnpm dev` proxies every `/admin/...` request to `http://localhost:8080`, so
start the backend first (`uv run fastapi dev src/app/main.py --port 8080`, or
however the service is launched in your checkout). Point the proxy elsewhere
with `VITE_DEV_API_TARGET=http://host:port pnpm dev`.

Every screen degrades honestly with no backend running: each list shows the
service's own `{"detail": ...}` in an error panel rather than an empty table.

```bash
pnpm build          # tsc -b && vite build  →  dist/
pnpm preview        # serve dist/ (no proxy — set VITE_API_BASE_URL for this)
pnpm lint           # oxlint
```

## Environment variables

Both are read at **build time** (Vite inlines them), so a production build needs
them set before `pnpm build`.

| Variable | Default | What it does |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `""` (same origin) | Where the `/admin` API lives. Leave empty when the console is served by the backend itself. In dev the value is ignored in favour of the proxy unless you set it explicitly. |
| `VITE_LANGFUSE_URL` | unset | Base URL of your Langfuse instance, e.g. `https://langfuse.example.org`. When it is unset every "open in Langfuse" link is **hidden** rather than guessed at — session ids stay copyable either way. |
| `VITE_DEV_API_TARGET` | `http://localhost:8080` | Dev-server proxy target only; not compiled into the app. |

`example.env` documents all of these — copy it to `.env` for local development.

## Docker

```bash
docker build -t sciedu-llm-ui .
docker run -p 8081:80 -e BACKEND_URL=http://llm-provider:8080 sciedu-llm-ui
```

The image (see `Dockerfile`) builds the static bundle and serves it with nginx.
The backend URL is a **runtime** setting: the bundle makes same-origin
`/admin/...` calls and nginx proxies them to `$BACKEND_URL` (scheme + host +
port, no trailing slash — see `nginx/default.conf.template`), so one image
serves every environment without a rebuild. `VITE_LANGFUSE_URL` is the one
build-time knob: pass it as `--build-arg VITE_LANGFUSE_URL=https://...` to show
the "open in Langfuse" links. The proxy allows 30-minute reads to survive
synchronous RAG rebuilds.

## Screens, and how they map to the spec

| Route | Screen | Spec section | API |
| --- | --- | --- | --- |
| `/rag` (default) | Retrieval settings | 1. RAG configuration | `GET`/`PATCH /admin/rag/config`, `POST /admin/rag/rebuild`, `POST /admin/rag/reset`, `GET /admin/datasets`, `GET /admin/models` |
| `/presets` | Behaviour presets | 2. Preset management | `GET /admin/presets`, `GET /admin/presets/{name}`, `POST /admin/presets/refresh` |
| `/presets/:name`, `/presets/new` | Preset editor | 2. Preset management | `GET`/`PUT`/`DELETE /admin/presets/{name}`, `GET /admin/models` |
| `/evals` | Evaluations | 3. Evaluation runs | `POST`/`GET /admin/evals/runs`, `POST /admin/evals/runs/{id}/cancel`, `GET /admin/datasets`, `GET /admin/judge-prompts`, `GET /admin/models` |
| `/evals/runs/:runId` | Run detail | 3. Evaluation runs | `GET /admin/evals/runs/{id}`, `GET /admin/evals/history`, cancel |
| `/reference` | What's available | Supporting lookups | `GET /admin/models`, `GET /admin/datasets`, `GET /admin/judge-prompts` |

The sidebar has exactly those four top-level entries; the preset editor and the
run detail are sub-screens reached by opening a row.

## Things worth knowing before changing it

- **Nothing is invented.** Types in `src/api/types.ts` are derived one-for-one
  from `src/app/schema/admin/*.py` and `src/app/presets.py`. The mockup shows
  telemetry the backend does not report — chunk counts, build durations,
  embedding progress bars, score summaries — and none of it is faked here. A
  rebuild in flight gets a pulsing dot, not a percentage; the run detail points
  at Langfuse instead of showing scores.
- **Rebuild, reset and apply-with-rebuild are synchronous long HTTP calls.** The
  client gives them a 30-minute timeout (`LONG_TIMEOUT_MS`) and the screen
  disables its controls while one is in flight. The dev proxy is configured with
  the same generous timeout.
- **Applied RAG settings live in the service's memory.** A restart reverts them
  to the `RAG_*` environment defaults; the sidebar footer and the Apply note
  both say so.
- **Polling** only runs while something is non-terminal: the run list and one
  run's detail re-read every 3s until every run reaches
  `completed`/`failed`/`cancelled`.
- **No automatic retries.** react-query's retry parks a query in a "paused"
  state while the tab is unfocused, which would show a permanent "Loading…"
  instead of the failure. Failures surface at once and offer "Try again".
- **`GET /admin/tools` does not exist yet.** `useTools()` asks for it anyway and
  falls back to `FALLBACK_TOOLS` in `src/api/types.ts`, which mirrors
  `_REGISTRY` in `src/app/agents/tools.py` (`rag_search`, `summon_subagent`).
  When the endpoint lands, the list goes live with no frontend change.
- **The preset editor's "Validate" button is local only** — `JSON.parse` plus a
  required-shape check. The semantic rules (tool names, what forced retrieval
  forbids, summoned characters needing a prompt) belong to the server and run on
  save; its 422 is rendered field by field.
- **Backend content may be Traditional Chinese** (preset display names such as
  `助教`). The app chrome stays in English, as in the mockup.

## Layout

```
src/
  api/        client.ts (typed fetch + FastAPI error shapes), errors.ts, types.ts, hooks.ts
  components/ AppShell, Panel/Field, ErrorPanel, StatusTag, Choices, ConfirmDialog, …
  screens/    rag/, presets/, evals/, reference/  — one folder per screen
  styles/     organic.css (the design system, vendored) + app.css (this app's layer)
  lib/        format.ts
```

Build with the design system's classes (`.btn`, `.tag`, `.field`, `.input`,
`.seg`, `.radio`, `.card`, `.table`, `.dialog`) and the mockup's helpers
(`.panel`, `.sect`, `.note`, `.mono`) rather than inventing parallel ones, and
take colours, spacing and radii from the `var(--color-*)` / `var(--space-*)` /
`var(--radius-*)` tokens. Icons are lucide-react at `strokeWidth={2.75}`.
Caprasimo and Figtree are loaded by the `@import` at the top of `organic.css`.
