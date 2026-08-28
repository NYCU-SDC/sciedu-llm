# Admin Panel — Feature Specification

Scope note: this document lists **required features only**. Visual design, layout, navigation structure, and implementation technology are intentionally unspecified and will be decided separately. The panel is a separately built frontend (lives in `ui/`) talking to the single FastAPI backend over HTTP. There is no authentication (the deployment is fronted by Cloudflare) and no database (all persistent state lives in Langfuse).

The panel serves administrators of the sciedu-llm service. It has exactly three concern areas: RAG configuration, preset management, and evaluation runs.

## 1. RAG configuration

- View the currently active RAG configuration of the live service: embedding model, rerank model, generator prompt names, embedding batch size, max concurrency, chunk size, chunk overlap, BM25 top-n, dense top-n, RRF k, rerank pool size, final k, and the list of corpus datasets currently indexed.
- See whether the RAG index is currently built/available.
- Edit any of the above parameters and apply them to the running service. Applying changes that affect the index (corpus datasets, chunking, embedding model) triggers a rebuild; the user must be able to tell a rebuild is in progress and whether it succeeded or failed.
- Select which corpus datasets are used, chosen from the list of corpus datasets that exist in Langfuse (the backend exposes this list; datasets are grouped under a corpus folder prefix).
- Trigger a manual index rebuild without changing parameters.
- Reset the configuration back to the server's environment-variable defaults (and rebuild).
- The user must be informed that applied changes are **not persistent across service restarts** (they live in memory; restart reverts to env defaults).

## 2. Preset management

Presets are JSON documents stored in the Langfuse dataset `config/presets`. A preset defines a named behavior profile for the LLM service: which model to use, which Langfuse prompts each character uses, which tools each character may call, cast composition (single assistant, or orchestrator + summonable second character), step budgets, tool-choice policy, and RAG mode (off / forced / model-chosen via the `rag_search` tool).

- List all presets currently served, distinguishing built-in presets (the code-defined defaults `default-agents` / `default-chat` / `default-chat-plain`, always present even when Langfuse is down) from dataset-defined presets (which shadow built-ins of the same name). The defaults are seeded into the dataset at startup when it does not already hold them, so they normally appear as dataset entries shadowing their built-in.
- View a preset's full JSON definition.
- Create a new preset and edit an existing one. The backend validates the document (schema + semantic rules, e.g. tool names must exist, summoned characters need a prompt); validation errors must be surfaced to the user with enough detail to fix the document.
- Delete a dataset-defined preset (built-ins cannot be deleted; deleting a shadowing preset reverts to the built-in).
- Trigger a registry refresh so edits take effect immediately (the backend otherwise refreshes on a TTL), and view the result of the last load: which presets loaded, and per-item errors for any dataset entries that failed validation.
- Reference support for authoring: the user needs to see the list of available tool names (`GET /admin/tools`, which also says which tools need RAG) and available models (`GET /admin/models`) while editing a preset.

## 3. Evaluation (judge) runs

Evaluations are launched on demand and run inside the backend; scores and traces persist in Langfuse. The run list held by the backend is in-memory (lost on service restart), while full results always remain browsable in Langfuse.

- Launch an evaluation run by choosing:
  - eval model (the model being judged) and judge model — from the backend's model list;
  - one or more corpus datasets and one or more question datasets — from the backend's Langfuse dataset listings;
  - one or more judge prompts — from the backend's judge-prompt listing;
  - retrieval/indexing parameters: k, embedding model, rerank model, chunk size, chunk overlap (all optional, defaulting to the server's RAG configuration).
- See the list of runs known to the backend with their status (pending / building / judging / completed / failed / cancelled), parameters, timing, and error message when failed.
- Watch a run progress until completion (status updates while building the index and judging).
- Cancel an in-flight run.
- For a completed run, obtain the Langfuse session identifier so the user can jump to the full traces/scores in the Langfuse UI.
- Browse past experiment history per question dataset (names/timestamps read back from Langfuse), independent of the in-memory run list.

## Supporting lookups the backend provides

These exist to populate the features above, not as standalone screens:

- List of models advertised by the upstream provider, annotated with which are on the service's chat allowlist, plus the server's default models.
- List of Langfuse datasets, grouped into corpus and question sets.
- List of judge prompts available in Langfuse.

## Error-handling expectations (feature-level)

- Backend/Langfuse/upstream failures surface as actionable error messages (the backend returns structured errors; nothing fails silently).
- Long-running operations (index rebuild, eval runs) must not leave the user guessing: in-progress, succeeded, and failed states are always distinguishable.
