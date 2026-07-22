# Agentic Workflow Protocol — Design

> Status: Proposal · Target: Sprint 6+
> Scope: extend the LLM Module to support (1) tool-calling agents and (2) multi-character conversations, while preserving the streaming / resumption contract the product already relies on.

This is the **engineering design**. For the consumer-facing spec that backend/frontend engineers integrate against, see [`llm-interaction-protocol-agentic.md`](./llm-interaction-protocol-agentic.md).

---

## 1. Where we are today

This repository is the **LLM Module** — the `LLM` participant in the existing *LLM Interaction Protocol* sequence diagrams. It is **stateless**:

- `POST /chat` accepts a full OpenAI-style `messages[]` array plus `{stream, model, enable_rag, session, user}` and returns either a JSON body (`{content, finishReason}`) or an SSE stream of `data: {"delta": "...", "isFinished": bool}` chunks (with an optional `error` field on the terminal chunk).
- `POST /chat/title` summarizes a conversation into a title.
- A hybrid RAG pipeline (`src/rag`) augments the latest user turn with retrieved context.
- Observability is Langfuse: an outer `chat` span wraps a nested `generation` (and, for RAG, a `rag-retrieve` retriever span).

Persistence, message IDs, message `status`, and stream resumption (`GET /chat/stream/:messageID`) live in the **Backend** service, not here. The Backend buffers the module's token stream and re-exposes it to the Frontend with resumption semantics.

**Two invariants we must not break:**

1. **The LLM Module stays stateless.** No conversation DB in this repo. Everything needed to (re)produce a turn is passed in the request.
2. **The token stream stays resumable.** The Backend can reconnect mid-turn, replay what it missed, and continue. Any new event we add must be replayable from a buffer.

---

## 2. Goals & non-goals

**Goals**
- An assistant turn may internally take multiple steps: reason → call tools → observe results → repeat → answer.
- **All tools are server-executed** — they run inside this module (RAG search, retrieval-by-id, calculator, …). The whole agent loop completes within a single streaming call.
- A turn may be produced by **multiple named characters** collaborating. The primary pattern: a **teacher** character acts as the orchestrator and can **summon** a **student** character (sub-agent) via a tool call. The Frontend renders these as two distinct people having a conversation.

**Non-goals (for this iteration)**
- Making the LLM Module stateful / adding a database here.
- **Client-executed / human-in-the-loop tools.** No tool requires Frontend input, so the turn never pauses waiting on an external result. (If this is ever needed, it is a separate design — it would reintroduce a suspend/resume state.)
- A general plugin marketplace. Tools are declared per-request or registered in config; no dynamic code loading.
- Long-running (minutes/hours) background agents. A turn is bounded by `max_steps` / `max_turns` and a wall-clock timeout.

---

## 3. Core data-model change: a message is a list of typed *parts*

Today an assistant message is a flat string. An agentic turn is not linear text — it interleaves reasoning, tool calls, tool results, and prose, across several characters. We model an assistant message as an **ordered list of parts**:

```jsonc
{
  "role": "assistant",
  "parts": [
    { "type": "text",        "id": "p0", "agent": "teacher",
      "text": "Let's think this through. I'll ask a student to try first." },
    { "type": "tool_call",   "id": "p1", "agent": "teacher", "internal": true,
      "tool_call_id": "call_1", "name": "summon_student",
      "arguments": { "task": "Explain the light reactions" } },
    { "type": "text",        "id": "p2", "agent": "student",
      "text": "The light reactions happen in the thylakoid membrane…" },
    { "type": "tool_result", "id": "p3", "agent": "teacher", "internal": true,
      "tool_call_id": "call_1", "status": "ok", "content": "…student's answer…" },
    { "type": "text",        "id": "p4", "agent": "teacher",
      "text": "Good start — now let's correct one thing…" }
  ]
}
```

**Part types**

| type          | streamed field | meaning |
| ------------- | -------------- | ------- |
| `text`        | `text`         | Prose shown to the user, attributed to a character. |
| `reasoning`   | `text`         | Model thinking / scratchpad. Optional to render; may be collapsed or hidden. |
| `tool_call`   | `arguments`    | An agent calls a tool. `arguments` streams as partial JSON. Always server-executed. |
| `tool_result` | `content`      | Outcome of a tool call, keyed by `tool_call_id`. `status: "ok" \| "error"`. |

**Common fields on every part**

- `id` — stable, unique within the message.
- `agent` — **the character that authored this part** (e.g. `"teacher"`, `"student"`; `"assistant"` for a single-character turn). This is the field the Frontend keys on to attribute a part to a speaker.
- `internal` (optional, default `false`) — plumbing the Frontend hides by default. The `summon_student` tool call and the tool result that feeds the student's answer back to the teacher are `internal`; the student's own `text` parts are **not** — they are shown as a real speaker (see §6).

### 3.1 Relationship to the OpenAI wire format

Parts map cleanly onto the OpenAI Chat Completions message format the module already speaks, which is what keeps the module stateless:

- A `tool_call` part ⇄ an entry in an assistant message's `tool_calls[]`.
- A `tool_result` part ⇄ a `{ "role": "tool", "tool_call_id": ..., "content": ... }` message.
- `text` ⇄ assistant `content`.
- `reasoning` is a protocol-level part we persist and render but collapse away when reconstructing the OpenAI `messages[]` for the next upstream call (dropped, or mapped to the provider's reasoning field if supported).

So "history" is still an OpenAI `messages[]` array on the wire; parts are the richer view the Backend persists and the Frontend renders.

---

## 4. Message status

The status enum is **unchanged** from today — no new state is needed, because every turn runs to completion inside one streaming call (no external pause point):

| status      | meaning |
| ----------- | ------- |
| `created`   | Row exists, generation not started. |
| `streaming` | Module is actively producing parts. |
| `completed` | Turn finished (`finishReason: "stop"`). |
| `failed`    | Terminal error. |

---

## 5. Tool calling

### 5.1 Request shape

`POST /chat` gains optional fields:

```jsonc
{
  "messages": [ ... ],
  "stream": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "rag_search",
        "description": "Search the course corpus.",
        "parameters": { "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] }
      }
    }
  ],
  "tool_choice": "auto",            // "auto" | "none" | "required" | {name}
  "max_steps": 8                    // upper bound on the model⇄tool loop
}
```

Every tool must resolve to a handler registered in this module (the **server tool registry**, §5.3). Unknown tools → `400`.

### 5.2 Execution — one streaming call does the whole loop

There is exactly one execution path. Tools run in-process; the loop completes inside a single streaming call:

```
model → finish_reason=tool_calls
      → module runs the handler(s) in-process
      → append tool results to messages
      → model again → … → final text → done
```

The Frontend sees `tool_call` and `tool_result` parts stream by and the turn ends `completed`. No extra HTTP round-trip, no pausing. Independent tool calls within one model step may run concurrently (`asyncio.gather`, bounded). `max_steps` bounds the loop; exceeding it ends the turn with `finishReason: "max_steps"`.

### 5.3 Server tool registry

A registry in a new `src/agents/tools` package maps tool name → async handler + JSON schema. First inhabitants:

- **RAG as a tool** — `rag_search(query)` and `rag_fetch(chapter, start, end)` backed by the existing `RAGPipeline`. This subsumes today's `enable_rag`: instead of always augmenting the latest turn, the agent decides when to search. `enable_rag: true` becomes sugar for "register `rag_search` with `tool_choice: auto`."
- **`summon_student(task, …)`** and friends — the character-summon tools (§6).

Handlers are plain `async def handler(args: dict, ctx) -> ToolResult`. `ctx` exposes the OpenAI client, the RAG pipeline, the Langfuse span (so each execution nests a `tool` span), and — for summon tools — the orchestrator so a handler can run a sub-character.

---

## 6. Multi-character conversation

### 6.1 The model: a visible cast + summoning

A **character** is a named, user-visible persona backed by an agent config (system prompt, model, allowed tools). One character is the **orchestrator** (the **teacher**). The teacher can **summon** another character (the **student**) via a tool call.

The crucial product requirement: **the summoned character is a first-class speaker, not a collapsed tool result.** The Frontend must render teacher and student as two people talking. So we split the summon into two concerns:

- **The mechanism** (hidden): the teacher's `summon_student` `tool_call` part and the `tool_result` part that feeds the student's answer back to the teacher. Both carry `internal: true`.
- **The conversation** (shown): the student runs as its own agent and its `text`/`reasoning` parts stream with `agent: "student"`, bracketed by `agent_start`/`agent_end`. These are ordinary, non-internal parts — a real speaker segment.

So a tool call is how the teacher *invokes* the student, but the student's *output* surfaces as a peer speaker inline in the transcript, in the position where it was produced.

### 6.2 Request shape

```jsonc
{
  "messages": [ ... ],
  "stream": true,
  "workflow": {
    "orchestrator": "teacher",
    "characters": [
      { "id": "teacher", "displayName": "Teacher", "role": "teacher",
        "prompt_name": "agents/teacher", "model": "gpt-oss-120b",
        "tools": ["rag_search", "summon_student"] },
      { "id": "student", "displayName": "Student", "role": "student",
        "prompt_name": "agents/student", "tools": ["rag_search"] }
    ],
    "max_turns": 6
  }
}
```

- Character prompts are **Langfuse-managed** (`prompt_name`), consistent with RAG and title generation. Inline `prompt` strings allowed for dev.
- `displayName` / `role` are identity metadata the Frontend uses to label speakers (avatar, name, styling). The module streams them in the `cast` event (§7.2) so the Frontend knows the roster before content arrives.
- `summon_student` is a server tool whose handler runs the `student` character as a sub-agent and returns its answer as the tool result.
- `max_turns` bounds how many times control passes between characters (independent of each character's `max_steps`).

### 6.3 What streams (the teacher ⇄ student flow)

1. `cast` — the roster, once at the start.
2. `agent_start {agent: "teacher"}` → teacher `text` parts.
3. Teacher decides to summon: `tool_call(summon_student)` part with `internal: true`.
4. **Before** the tool result is produced, the module opens the student as a speaker: `agent_start {agent: "student", parent: "teacher", summonedBy: "call_1"}` → student `text` parts (**not** internal) → `agent_end {agent: "student"}`.
5. The student's answer is returned to the teacher as an `internal` `tool_result` part.
6. `agent_start {agent: "teacher"}` resumes → teacher `text` parts → `agent_end`.
7. `done {status: "completed"}`.

`parent` / `summonedBy` on `agent_start` let the Frontend optionally thread the student under the summoning turn, but the default render is a flat two-person dialogue keyed on `agent`.

### 6.4 Statelessness

The whole orchestration is deterministic given `{messages, workflow}`, so a run is fully reproducible for tracing/replay. Because no tool needs Frontend input, the turn never suspends — it either runs to `completed` or ends `failed`. Nothing about multi-character changes the stateless contract.

---

## 7. SSE stream changes

This is the heart of the change. Today the stream is a single flat event shape:

```
data: {"delta": "Hello", "isFinished": false}
data: {"delta": "", "isFinished": true}
```

That shape assumes **one linear run of text = one message**. Agentic turns break that assumption: a message is now an ordered list of typed parts, authored by different characters, interleaved with tool calls. So the stream carries **typed events with a `type` discriminator**, and each content event is anchored to a part by `index`.

### 7.1 Event types

`data:`-framed JSON, each with a `type`:

| `type`        | payload | meaning |
| ------------- | ------- | ------- |
| `cast`        | `{ characters: [{ id, displayName, role }] }` | The roster of characters in this turn. Emitted once, first. Single-character turns omit it. |
| `part_start`  | `{ index, part }` | A new part begins; `part` carries its metadata (`type`, `id`, `agent`, `internal`, `tool_call_id`, `name`) with an empty streamed field. |
| `delta`       | `{ index, delta }` | Append `delta` to part `index`'s streamed field (`text` / `arguments` / `content`). |
| `part_end`    | `{ index, part }` | Part `index` is finalized; `part` carries the full content. |
| `agent_start` | `{ agent, parent?, summonedBy? }` | A character begins speaking. `parent`/`summonedBy` set when it was summoned by another character. |
| `agent_end`   | `{ agent }` | A character's segment ends. |
| `done`        | `{ finishReason, status }` | Terminal success. `status` is always `completed` here. |
| `error`       | `{ error, code }` | Terminal failure. |

**How the pieces fit:**

- **`index`** is the part's position in the message's `parts[]`. It is how the Frontend routes a `delta` to the right part and how a reconnecting client re-anchors (§8). Deltas are strictly append-only per index.
- **`agent`** on every part (and the `agent_start`/`agent_end` frame) is how the Frontend knows *who is speaking right now* — the answer to "differentiate who's who." The Frontend flips the active speaker on `agent_start` and attributes every subsequent part to that character until `agent_end`.
- **`internal`** on a part tells the Frontend "this is plumbing" (the summon tool call / result), so it can hide it while still rendering the student's speaker segment as a person.

### 7.2 Worked example — teacher summons student

```
data: {"type":"cast","characters":[{"id":"teacher","displayName":"Teacher","role":"teacher"},{"id":"student","displayName":"Student","role":"student"}]}

data: {"type":"agent_start","agent":"teacher"}
data: {"type":"part_start","index":0,"part":{"type":"text","id":"p0","agent":"teacher"}}
data: {"type":"delta","index":0,"delta":"Let's have a student try first."}
data: {"type":"part_end","index":0,"part":{"type":"text","id":"p0","agent":"teacher","text":"Let's have a student try first."}}

data: {"type":"part_start","index":1,"part":{"type":"tool_call","id":"p1","agent":"teacher","internal":true,"tool_call_id":"call_1","name":"summon_student"}}
data: {"type":"delta","index":1,"delta":"{\"task\":\"Explain"}
data: {"type":"delta","index":1,"delta":" the light reactions\"}"}
data: {"type":"part_end","index":1,"part":{"type":"tool_call","id":"p1","agent":"teacher","internal":true,"tool_call_id":"call_1","name":"summon_student","arguments":{"task":"Explain the light reactions"}}}

data: {"type":"agent_start","agent":"student","parent":"teacher","summonedBy":"call_1"}
data: {"type":"part_start","index":2,"part":{"type":"text","id":"p2","agent":"student"}}
data: {"type":"delta","index":2,"delta":"The light reactions happen in the thylakoid membrane…"}
data: {"type":"part_end","index":2,"part":{"type":"text","id":"p2","agent":"student","text":"The light reactions happen in the thylakoid membrane…"}}
data: {"type":"agent_end","agent":"student"}

data: {"type":"part_start","index":3,"part":{"type":"tool_result","id":"p3","agent":"teacher","internal":true,"tool_call_id":"call_1"}}
data: {"type":"part_end","index":3,"part":{"type":"tool_result","id":"p3","agent":"teacher","internal":true,"tool_call_id":"call_1","status":"ok","content":"…"}}

data: {"type":"part_start","index":4,"part":{"type":"text","id":"p4","agent":"teacher"}}
data: {"type":"delta","index":4,"delta":"Good start — now let's correct one thing…"}
data: {"type":"part_end","index":4,"part":{"type":"text","id":"p4","agent":"teacher","text":"Good start — now let's correct one thing…"}}
data: {"type":"agent_end","agent":"teacher"}

data: {"type":"done","finishReason":"stop","status":"completed"}
```

The Frontend renders this as: Teacher speaks (p0) → [hidden summon] → Student speaks (p2) → [hidden result] → Teacher speaks (p4). Two people, clearly attributed, in real time.

### 7.3 Backward compatibility

A plain request (no `tools`, no `workflow`) must behave exactly like today. Two options, decide with the Backend:

- **(A) Legacy passthrough:** when the request has neither `tools` nor `workflow`, emit the old `{delta, isFinished}` shape. Zero client change. *Recommended for the first ship.*
- **(B) Versioned stream:** a `protocol: 2` request flag (or `Accept` header) selects typed events.

Prefer (A) initially, migrate the Backend to typed events, then make typed events the default and keep legacy behind a flag.

---

## 8. Resumption with a structured event log

The article's resumption rule ("replay what you missed, then continue live") generalizes: the Backend buffers the **ordered typed-event log** per streaming message, not a text string. On `GET /chat/stream/:messageID` (a Backend endpoint; the module itself stays stateless and simply produces the stream once):

1. Emit a **snapshot**: the `cast`, all completed parts (`part_start`/`part_end`, with `agent_start`/`agent_end` frames), plus the in-progress part replayed as one `part_start` + coalesced `delta` with everything accumulated so far.
2. Continue forwarding live events.

Because every content event carries its `index`, a client that reconnects discards duplicates and re-anchors deterministically. `completed` and `failed` messages have a fully-buffered, finite log — a reconnect returns the log and closes.

---

## 9. Observability

Extend the existing Langfuse nesting:

```
chat (span)
├─ agent:teacher (span)
│  ├─ generation (step 1 → text + summon_student tool_call)
│  ├─ tool:summon_student (span)
│  │  └─ agent:student (span)
│  │     ├─ generation (student step 1 → maybe rag_search)
│  │     ├─ tool:rag_search (span)   # wraps the existing rag-retrieve retriever span
│  │     └─ generation (student step 2 → answer)
│  └─ generation (step 2 → teacher's correction)
```

- One `generation` per model round-trip (a step), not per turn.
- One `tool:<name>` span per tool execution; the RAG tool nests the existing `rag-retrieve` span unchanged; the `summon_student` tool nests the whole `agent:student` sub-tree.
- `session`/`user` propagation is unchanged. Add `metadata`: `{ orchestrator, agent, step, finish_reason }`.

---

## 10. Proposed code layout

```
src/agents/
  __init__.py
  loop.py          # single-agent tool loop (model ⇄ tools, bounded by max_steps)
  orchestrator.py  # multi-character run: cast, summoning, turn budget
  events.py        # typed SSE event dataclasses + serialization
  parts.py         # Part models; parts ⇄ OpenAI messages[] conversion
  registry.py      # server tool registry
  tools/
    rag.py         # rag_search, rag_fetch (wrap RAGPipeline)
    summon.py      # summon_student etc. (run a sub-character via the orchestrator)
src/app/
  routers/chat.py  # thin: parse request → pick loop/orchestrator → stream events
  schema/chat.py   # ChatRequest gains tools/tool_choice/max_steps/workflow
```

`routers/chat.py` stays a thin adapter: build the trace context (as today), delegate to `loop.run(...)` or `orchestrator.run(...)`, serialize the yielded events to SSE. The RAG augmentation path is refactored into the `rag_search` tool; the legacy `enable_rag` flag becomes sugar over it.

---

## 11. Phasing

1. **Parts + typed events, single agent, server tools.** Ship legacy-passthrough (§7.3-A). Introduce the tool loop and RAG-as-a-tool. No API break.
2. **Multi-character (teacher + summon student).** Add the orchestrator, the `cast`/`agent_start`/`agent_end` events, `summon_*` tools, and per-character tracing.
3. **Typed events as default.** Flip the default once the Backend/Frontend consume typed events; keep legacy behind a flag.

---

## 12. Open questions

- **Reasoning exposure:** do we stream `reasoning` parts to the Frontend, hide them, or make it per-request? Depends on the upstream model and product/privacy stance.
- **Student → teacher visibility:** the student answers the teacher, but does the student also "hear" earlier conversation? Define exactly what history each character's sub-agent receives (full transcript vs. just the summon task).
- **Nested summons:** may a student summon another character? Bound the depth (recommend depth ≤ 1 initially) and enforce it in the orchestrator.
- **Tool-call parallelism:** run independent tool calls in one step concurrently? Straightforward with `asyncio.gather`; bound it.
- **Cost/loop safety:** `max_steps` + `max_turns` + wall-clock timeout + a per-turn token budget. Where do the defaults live — config or per-request?
