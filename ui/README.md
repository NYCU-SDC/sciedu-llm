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

`pnpm dev` proxies every `/admin/...` request — plus `/healthz`, which the top
bar polls, and `/agents`, which the playground streams — to
`http://localhost:8080`, so start the backend first (`uv run fastapi dev
src/app/main.py --port 8080`, or however the service is launched in your
checkout). Point the proxy elsewhere with
`VITE_DEV_API_TARGET=http://host:port pnpm dev`.

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

| Variable              | Default                 | What it does                                                                                                                                                                                                                       |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VITE_API_BASE_URL`   | `""` (same origin)      | Where the backend lives — it prefixes `/admin/...`, `/agents` and `/healthz` alike. Leave empty when the console is served by the backend itself. In dev the value is ignored in favour of the proxy unless you set it explicitly. |
| `VITE_LANGFUSE_URL`   | unset                   | Base URL of your Langfuse instance, e.g. `https://langfuse.example.org`. When it is unset every "open in Langfuse" link is **hidden** rather than guessed at — session ids stay copyable either way.                               |
| `VITE_DEV_API_TARGET` | `http://localhost:8080` | Dev-server proxy target only; not compiled into the app.                                                                                                                                                                           |

`example.env` documents all of these — copy it to `.env` for local development.

## Docker

```bash
docker build -t sciedu-llm-ui .
docker run -p 8081:80 -e BACKEND_URL=http://llm-provider:8080 sciedu-llm-ui
```

The image (see `Dockerfile`) builds the static bundle and serves it with nginx.
The backend URL is a **runtime** setting: the bundle makes same-origin
`/admin/...`, `/agents` and `/healthz` calls and nginx proxies them to
`$BACKEND_URL` (scheme + host + port, no trailing slash — see
`nginx/default.conf.template`), so one image serves every environment without a
rebuild. `VITE_LANGFUSE_URL` is the one build-time knob: pass it as
`--build-arg VITE_LANGFUSE_URL=https://...` to show the "open in Langfuse"
links. The `/agents` proxy turns buffering off so SSE frames reach the browser
as they are produced.

### Runtime variables

| Variable       | Default                 | What it does                                                                                                                                               |
| -------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BACKEND_URL`  | `http://localhost:8080` | Where `/admin/*`, `/agents` and `/healthz` are proxied. **Not** `localhost` in compose — inside this container that is nginx itself.                       |
| `DNS_RESOLVER` | `127.0.0.11`            | The DNS server `BACKEND_URL` is resolved against. The default is Docker's embedded DNS, which serves compose service names; change it only outside Docker. |

### In compose

Point `BACKEND_URL` at the backend **service name** and the containers can start
in any order:

```yaml
services:
    llm-provider:
        image: sciedu-llm
        # no ports needed: only the console talks to it
    ui:
        image: sciedu-llm-ui
        ports:
            - "8081:80"
        environment:
            BACKEND_URL: http://llm-provider:8080
```

nginx resolves that name per request rather than at startup, so the console
boots even when the backend is not up yet (API calls answer 502 until it is, and
the console reports that as an error rather than pretending), and a backend
container recreated on a new IP is followed within ten seconds — no `depends_on`
and no restart required.

If `BACKEND_URL` points anywhere that is _not_ the API — the console's own
hostname, or a gateway that routes back to it — `/admin/...` falls through to the
SPA and answers `index.html` with a 200. The client rejects a non-JSON 2xx with a
message naming exactly that, instead of failing deep inside a screen.

## Screens, and how they map to the spec

| Route                            | Screen             | Spec section                                   | API                                                                                                                                          |
| -------------------------------- | ------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `/rag` (default)                 | Retrieval settings | 1. RAG configuration                           | `GET`/`PATCH /admin/rag/config`, `POST /admin/rag/rebuild`, `POST /admin/rag/reset`, `GET /admin/datasets`, `GET /admin/models`              |
| `/presets`                       | Behaviour presets  | 2. Preset management                           | `GET /admin/presets`, `GET /admin/presets/{name}`, `POST /admin/presets/refresh`                                                             |
| `/presets/:name`, `/presets/new` | Preset editor      | 2. Preset management                           | `GET`/`PUT`/`DELETE /admin/presets/{name}`, `GET /admin/models`                                                                              |
| `/evals`                         | Evaluations        | 3. Evaluation runs                             | `POST`/`GET /admin/evals/runs`, `POST /admin/evals/runs/{id}/cancel`, `GET /admin/datasets`, `GET /admin/judge-prompts`, `GET /admin/models` |
| `/evals/runs/:runId`             | Run detail         | 3. Evaluation runs                             | `GET /admin/evals/runs/{id}`, `GET /admin/evals/history`, cancel                                                                             |
| `/playground`                    | Playground         | manual testing, not in `docs/admin-ui-spec.md` | `POST /agents` (SSE), `GET /admin/presets`                                                                                                   |
| `/reference`                     | What's available   | Supporting lookups                             | `GET /admin/models`, `GET /admin/datasets`, `GET /admin/judge-prompts`                                                                       |

The top bar carries exactly those top-level entries; the preset editor and the
run detail are sub-screens reached by opening a row. Beside them, on the right
of the bar, sits the backend indicator: `GET /healthz` every 5s, green when it
answers and red when it does not.

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
  to the `RAG_*` environment defaults. The screens no longer repeat that warning
  beside every Apply — it read as a caveat about the console rather than a fact
  about the service — but it is still what happens.
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
- **The preset editor is a form, and only a form.** Raw JSON authoring lives in
  one place: **Import presets** on the preset list, which takes a single document
  or an array of them, shape-checks each in the browser (`JSON.parse` plus
  `checkPresetShape`) and then writes them one `PUT /admin/presets/{name}` at a
  time so a rejection reports against the document it came from. The semantic
  rules (tool names, what forced retrieval forbids, summoned characters needing a
  prompt) belong to the server either way, and its 422 is rendered field by
  field.
- **Dataset pickers fold Langfuse's paths.** `FolderDatasetPicker` strips the
  group prefix, groups by the next segment and offers per-folder and
  select-everything checkboxes with a real indeterminate state — but every value
  it hands back is the full dataset name.
- **The playground is the one screen that talks past `/admin`.** It streams
  `POST /agents` itself — `EventSource` cannot POST, so `src/api/agentsStream.ts`
  reads the response body with a `fetch` reader, splits `data: …\n\n` frames
  across chunk boundaries and types every event in
  `docs/agents-spec.md`. Both proxies had to learn the route: `/agents` in
  `vite.config.ts`, and a `location /agents` block in the nginx template with
  `proxy_buffering off` — nginx would otherwise hold the whole answer back until
  the run finished, which is the opposite of what the screen is for.
- **Playground identity: one user, one session per page load.** Every request
  sends `user: "playground"`, so a Langfuse filter separates manual pokes at the
  service from anything real, and `session: <uuid>` minted once per page load
  (`crypto.randomUUID`, with a fallback for the plain-http lab origins where it
  is undefined) so the turns of one conversation group into a single trace
  session. The id is shown, copyable, in the screen header; **New session** mints
  a fresh one and clears the transcript. Nothing is persisted anywhere — no
  storage, no backend record — so a reload is a clean slate, and because /agents
  is stateless every turn re-sends the whole conversation, with each finished
  answer folded into one assistant message (its speakers' non-internal text, the
  reasoning and tool traffic dropped).
- **Backend content may be Traditional Chinese** (preset display names such as
  `助教`). The app chrome stays in English, as in the mockup.

## Layout

```
src/
  api/        client.ts (typed fetch + FastAPI error shapes), errors.ts, types.ts, hooks.ts,
              agentsStream.ts (the POST /agents SSE reader and its event union)
  components/ AppShell, Panel/Field, ErrorPanel, StatusTag, Choices, ConfirmDialog, …
  screens/    rag/, presets/, evals/, playground/, reference/  — one folder per screen
  styles/     organic.css (the design system, vendored) + app.css (this app's layer)
  lib/        format.ts
```

Build with the design system's classes (`.btn`, `.tag`, `.field`, `.input`,
`.seg`, `.radio`, `.card`, `.table`, `.dialog`) and the mockup's helpers
(`.panel`, `.sect`, `.note`, `.mono`) rather than inventing parallel ones, and
take colours, spacing and radii from the `var(--color-*)` / `var(--space-*)` /
`var(--radius-*)` tokens. Icons are lucide-react at `strokeWidth={2.75}`.
Caprasimo and Figtree are loaded by the `@import` at the top of `organic.css`.
