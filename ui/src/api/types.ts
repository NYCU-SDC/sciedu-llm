/* Types derived one-for-one from the FastAPI schemas in
 * `src/app/schema/admin/*.py` and `src/app/presets.py`. Nothing here is
 * invented: if a field is not on the Python model it is not in this file, and
 * the UI never renders a number the API did not send. */

// ── /healthz ──────────────────────────────────────────────────────────────
// app/schema/health.py

/** HealthzResponse. The body is barely the point — the status code is what the
 * top bar's indicator reports. */
export interface HealthzResponse {
  status: string
}

// ── /admin/rag ────────────────────────────────────────────────────────────
// app/schema/admin/rag.py

/** RAGConfigValues — the tunable knobs, all of them required on a response. */
export interface RagConfigValues {
  embedding_model: string
  rerank_model: string
  embedding_batch_size: number
  max_concurrency: number
  chunk_size: number
  chunk_overlap: number
  generator_system_prompt_name: string
  generator_user_prompt_name: string
  bm25_top_n: number
  dense_top_n: number
  rrf_k: number
  rerank_pool_size: number
  final_k: number
}

/** RAGConfigResponse — the effective config plus pipeline status. */
export interface RagConfigResponse extends RagConfigValues {
  is_built: boolean
  corpus_datasets: string[]
}

/** RAGConfigUpdate — a partial override. `rebuild` defaults to true server-side. */
export interface RagConfigUpdate extends Partial<RagConfigValues> {
  rebuild?: boolean
  corpus_datasets?: string[]
}

/** RAGConfigUpdateResponse */
export interface RagConfigUpdateResponse {
  config: RagConfigResponse
  rebuilt: boolean
}

/** Keys of the config whose value only takes effect once the index is rebuilt.
 * Mirrors the router: `corpus_datasets` always forces a rebuild, and chunking /
 * the embedding model are baked into the stored vectors. */
export const REBUILD_KEYS = [
  'corpus_datasets',
  'embedding_model',
  'chunk_size',
  'chunk_overlap',
] as const

// ── /admin/presets ────────────────────────────────────────────────────────
// app/presets.py + app/schema/admin/presets.py

/** PRESET_ID_PATTERN from app/presets.py. */
export const PRESET_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/
/** MAX_STEPS_CAP from app/agents/engine.py. */
export const MAX_STEPS_CAP = 16

export type ToolChoice = 'auto' | 'none' | 'required'
export type RagMode = 'off' | 'forced'

/** PresetCharacter. */
export interface PresetCharacter {
  id: string
  display_name: string
  role: string
  prompt_name: string | null
  tools: string[]
  max_steps: number
}

/** Preset — the exact document the JSON editor edits and PUT accepts. */
export interface Preset {
  name: string
  description: string
  model: string | null
  max_steps: number
  tool_choice: ToolChoice
  rag_mode: RagMode
  orchestrator: string
  characters: PresetCharacter[]
}

/** PresetSummary. A preset is dataset-defined — and so deletable — when
 * `!builtin || shadowed_builtin`. */
export interface PresetSummary {
  name: string
  description: string
  builtin: boolean
  shadowed_builtin: boolean
}

/** PresetDetail. */
export interface PresetDetail extends PresetSummary {
  definition: Preset
}

/** PresetLoadReportResponse. `errors` is keyed by Langfuse dataset *item id*;
 * `fetched_at` is null when the fetch itself failed and the previous map is
 * still in service. */
export interface PresetLoadReport {
  loaded: string[]
  errors: Record<string, string>
  fetched_at: number | null
}

// ── /admin/evals ──────────────────────────────────────────────────────────
// app/schema/admin/evals.py + judge/runner.py RunStatus

export type RunStatus =
  | 'pending'
  | 'building'
  | 'judging'
  | 'completed'
  | 'failed'
  | 'cancelled'

/** TERMINAL_STATUSES from judge/runner.py. */
export const TERMINAL_STATUSES: readonly RunStatus[] = [
  'completed',
  'failed',
  'cancelled',
]

export function isTerminal(status: string): boolean {
  return (TERMINAL_STATUSES as readonly string[]).includes(status)
}

/** EvalRunCreate. Unset model / chunking fields fall back to the server's
 * RAG config. */
export interface EvalRunCreate {
  eval_model: string
  judge_model: string
  corpus_datasets: string[]
  question_datasets: string[]
  judge_prompts: string[]
  k?: number
  embedding_model?: string | null
  rerank_model?: string | null
  chunk_size?: number | null
  chunk_overlap?: number | null
}

/** EvalRunResponse. `status` is typed loosely on the wire (`str`), so callers
 * should treat an unknown value as "in flight but unrecognised". */
export interface EvalRun {
  run_id: string
  status: RunStatus
  eval_model: string
  judge_model: string
  corpus_datasets: string[]
  question_datasets: string[]
  judge_prompts: string[]
  k: number
  embedding_model: string
  rerank_model: string
  chunk_size: number
  chunk_overlap: number
  max_concurrency: number
  started_at: string
  finished_at: string | null
  duration_seconds: number
  session_id: string | null
  error: string | null
}

/** EvalHistoryEntry — Langfuse's own durable record of a judge run. */
export interface EvalHistoryEntry {
  dataset_name: string
  run_name: string
  created_at: string
  description: string | null
}

// ── /admin metadata ───────────────────────────────────────────────────────
// app/schema/admin/meta.py

/** NamedResource — `name` is what to send back, `label` is the same name with
 * its folder prefix stripped. */
export interface NamedResource {
  name: string
  label: string
}

/** ModelDefaults. */
export interface ModelDefaults {
  eval_model: string
  judge_model: string
  embedding_model: string
  rerank_model: string
}

/** ModelsResponse. `models` is the unfiltered upstream listing;
 * `allowed_models` only governs what /chat may serve. */
export interface ModelsResponse {
  models: string[]
  allowed_models: string[]
  defaults: ModelDefaults
}

/** DatasetsResponse. */
export interface DatasetsResponse {
  corpus: NamedResource[]
  questions: NamedResource[]
}

// ── tools ─────────────────────────────────────────────────────────────────

export interface ToolInfo {
  name: string
  description: string
}

/** The tool registry has no HTTP endpoint yet (`app/agents/tools.py` owns it
 * in-process). `useTools()` tries `GET /admin/tools` first so the list becomes
 * live the day that endpoint lands, and falls back to this constant — which is
 * `_REGISTRY` in `src/app/agents/tools.py`, verbatim — on a 404. */
export const FALLBACK_TOOLS: ToolInfo[] = [
  { name: 'rag_search', description: '搜尋課程教材。' },
  {
    name: 'summon_subagent',
    description: '先呼叫另一位角色回答。',
  },
]
