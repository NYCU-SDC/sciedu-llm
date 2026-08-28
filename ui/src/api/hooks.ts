/* react-query bindings for the /admin surface.
 *
 * Queries are conservative: nothing refetches on window focus, because half the
 * screens here are forms the user is mid-way through filling in. The two
 * screens that need liveness (the run list and one run's detail) opt into
 * polling only while something is actually in flight. */

import {
    useMutation,
    useQuery,
    useQueryClient,
    type UseQueryResult,
} from "@tanstack/react-query";

import { api } from "./client";
import {
    FALLBACK_TOOLS,
    isTerminal,
    type DatasetsResponse,
    type EvalHistoryEntry,
    type EvalRun,
    type EvalRunCreate,
    type HealthzResponse,
    type ModelsResponse,
    type NamedResource,
    type Preset,
    type PresetDetail,
    type PresetLoadReport,
    type PresetSummary,
    type RagConfigResponse,
    type RagConfigUpdate,
    type RagConfigUpdateResponse,
    type ToolInfo,
} from "./types";

export const keys = {
    health: ["health"] as const,
    ragConfig: ["rag", "config"] as const,
    presets: ["presets", "list"] as const,
    preset: (name: string) => ["presets", "detail", name] as const,
    presetReport: ["presets", "report"] as const,
    runs: ["evals", "runs"] as const,
    run: (id: string) => ["evals", "run", id] as const,
    history: (dataset: string) => ["evals", "history", dataset] as const,
    models: ["meta", "models"] as const,
    datasets: ["meta", "datasets"] as const,
    judgePrompts: ["meta", "judge-prompts"] as const,
    tools: ["meta", "tools"] as const,
};

// ── health ────────────────────────────────────────────────────────────────

const HEALTH_POLL_MS = 5000;

/** Is the service answering right now? The only query in the console that polls
 * unconditionally — the top bar's indicator must mean one thing and one thing
 * only: `GET /healthz` answered 200 within the last few seconds. `retry` and
 * `networkMode` are restated here rather than inherited, because an indicator
 * that pauses when the browser thinks it is offline would be reporting the
 * browser's opinion instead of the service's. */
export function useHealth(): UseQueryResult<HealthzResponse> {
    return useQuery({
        queryKey: keys.health,
        queryFn: ({ signal }) => api.get<HealthzResponse>("/healthz", signal),
        refetchInterval: HEALTH_POLL_MS,
        // Keep polling even when the window is not the one in front: a console left
        // open beside something else would otherwise sit on a stale dot until it
        // was clicked back into, which is exactly when the answer matters least.
        refetchIntervalInBackground: true,
        refetchOnWindowFocus: true,
        staleTime: 0,
        retry: false,
        networkMode: "always",
    });
}

// ── metadata ──────────────────────────────────────────────────────────────

export function useModels(): UseQueryResult<ModelsResponse> {
    return useQuery({
        queryKey: keys.models,
        queryFn: ({ signal }) =>
            api.get<ModelsResponse>("/admin/models", signal),
    });
}

export function useDatasets(): UseQueryResult<DatasetsResponse> {
    return useQuery({
        queryKey: keys.datasets,
        queryFn: ({ signal }) =>
            api.get<DatasetsResponse>("/admin/datasets", signal),
    });
}

export function useJudgePrompts(): UseQueryResult<NamedResource[]> {
    return useQuery({
        queryKey: keys.judgePrompts,
        queryFn: ({ signal }) =>
            api.get<NamedResource[]>("/admin/judge-prompts", signal),
    });
}

/** The tool registry lives in-process on the backend and has no route yet. Try
 * for one anyway — the day `GET /admin/tools` lands this list goes live with no
 * frontend change — and otherwise serve the typed constant, which mirrors
 * `_REGISTRY` in `app/agents/tools.py`. The fallback covers every failure, not
 * just the 404: the preset editor needs the names to be authorable, and a
 * build-time copy of a code-defined registry is a better answer than a spinner
 * that never resolves. */
export function useTools(): UseQueryResult<ToolInfo[]> {
    return useQuery({
        queryKey: keys.tools,
        queryFn: async ({ signal }) => {
            try {
                const raw = await api.get<unknown>("/admin/tools", signal);
                const list = Array.isArray(raw) ? raw : [];
                const parsed = list.flatMap((entry): ToolInfo[] => {
                    if (typeof entry !== "object" || entry === null) return [];
                    const record = entry as Record<string, unknown>;
                    if (typeof record.name !== "string") return [];
                    return [
                        {
                            name: record.name,
                            description:
                                typeof record.description === "string"
                                    ? record.description
                                    : "",
                        },
                    ];
                });
                return parsed.length > 0 ? parsed : FALLBACK_TOOLS;
            } catch {
                return FALLBACK_TOOLS;
            }
        },
        staleTime: Infinity,
    });
}

// ── rag ───────────────────────────────────────────────────────────────────

const BUILD_POLL_MS = 2000;

/** The live config — and, in `build`, whatever the index build is doing.
 *
 * Re-read from the service every time the screen is opened, not served from the
 * 15s cache the other queries share: this screen's whole claim is "these are the
 * settings in use right now", and the service can have moved on without this tab
 * (a build finishing, a cancelled one rolling its settings back, another
 * operator applying a change). Polls while a build is running, which is the one
 * thing here that changes with nobody touching it. */
export function useRagConfig(): UseQueryResult<RagConfigResponse> {
    return useQuery({
        queryKey: keys.ragConfig,
        queryFn: ({ signal }) =>
            api.get<RagConfigResponse>("/admin/rag/config", signal),
        staleTime: 0,
        refetchOnMount: "always",
        // `build?.` and not `build.`: a service older than this console has no
        // `build` field at all, and an optional chain is the difference between
        // "never polls" and a TypeError thrown inside react-query.
        refetchInterval: (query) =>
            query.state.data?.build?.status === "building"
                ? BUILD_POLL_MS
                : false,
        refetchIntervalInBackground: true,
    });
}

/** PATCH, POST /rebuild, POST /rebuild/cancel and POST /reset all answer at
 * once: a rebuild is scheduled server-side, not awaited, so none of these needs
 * a long timeout. Each writes the returned config straight into the cache so the
 * screen never shows a stale snapshot, and `useRagConfig` takes over polling
 * from there. */
export function useRagMutations() {
    const client = useQueryClient();
    const settle = (config: RagConfigResponse) => {
        client.setQueryData(keys.ragConfig, config);
    };

    const apply = useMutation({
        mutationFn: (update: RagConfigUpdate) =>
            api.patch<RagConfigUpdateResponse>("/admin/rag/config", update),
        onSuccess: (result) => settle(result.config),
    });

    const rebuild = useMutation({
        mutationFn: () => api.post<RagConfigResponse>("/admin/rag/rebuild"),
        onSuccess: settle,
    });

    const cancelBuild = useMutation({
        mutationFn: () =>
            api.post<RagConfigResponse>("/admin/rag/rebuild/cancel"),
        onSuccess: settle,
    });

    const reset = useMutation({
        mutationFn: () => api.post<RagConfigUpdateResponse>("/admin/rag/reset"),
        onSuccess: (result) => settle(result.config),
    });

    return { apply, rebuild, cancelBuild, reset };
}

// ── presets ───────────────────────────────────────────────────────────────

export function usePresets(): UseQueryResult<PresetSummary[]> {
    return useQuery({
        queryKey: keys.presets,
        queryFn: ({ signal }) =>
            api.get<PresetSummary[]>("/admin/presets", signal),
    });
}

export function usePreset(
    name: string | undefined
): UseQueryResult<PresetDetail> {
    return useQuery({
        queryKey: keys.preset(name ?? ""),
        queryFn: ({ signal }) =>
            api.get<PresetDetail>(
                `/admin/presets/${encodeURIComponent(name ?? "")}`,
                signal
            ),
        enabled: Boolean(name),
    });
}

/** The load report. The backend deliberately has no `GET /load-report` — see
 * the module docstring on `app/routers/admin/presets.py`: `POST /refresh` is
 * the idempotent way to ask "what loaded, and what was rejected", so this is a
 * query that happens to use POST. */
export function usePresetReport(): UseQueryResult<PresetLoadReport> {
    return useQuery({
        queryKey: keys.presetReport,
        queryFn: () => api.post<PresetLoadReport>("/admin/presets/refresh"),
    });
}

export function usePresetMutations() {
    const client = useQueryClient();
    const invalidate = () => {
        void client.invalidateQueries({ queryKey: ["presets"] });
    };

    const save = useMutation({
        mutationFn: ({ name, preset }: { name: string; preset: Preset }) =>
            api.put<PresetDetail>(
                `/admin/presets/${encodeURIComponent(name)}`,
                preset
            ),
        onSuccess: invalidate,
    });

    const remove = useMutation({
        mutationFn: (name: string) =>
            api.delete<void>(`/admin/presets/${encodeURIComponent(name)}`),
        onSuccess: invalidate,
    });

    const refresh = useMutation({
        mutationFn: () => api.post<PresetLoadReport>("/admin/presets/refresh"),
        onSuccess: (report) => {
            client.setQueryData(keys.presetReport, report);
            void client.invalidateQueries({ queryKey: keys.presets });
        },
    });

    return { save, remove, refresh };
}

// ── evals ─────────────────────────────────────────────────────────────────

const POLL_MS = 3000;

export function useEvalRuns(): UseQueryResult<EvalRun[]> {
    return useQuery({
        queryKey: keys.runs,
        queryFn: ({ signal }) =>
            api.get<EvalRun[]>("/admin/evals/runs", signal),
        // Poll only while something is still moving.
        refetchInterval: (query) => {
            const runs = query.state.data;
            if (!runs) return false;
            return runs.some((run) => !isTerminal(run.status))
                ? POLL_MS
                : false;
        },
    });
}

export function useEvalRun(runId: string | undefined): UseQueryResult<EvalRun> {
    return useQuery({
        queryKey: keys.run(runId ?? ""),
        queryFn: ({ signal }) =>
            api.get<EvalRun>(
                `/admin/evals/runs/${encodeURIComponent(runId ?? "")}`,
                signal
            ),
        enabled: Boolean(runId),
        refetchInterval: (query) => {
            const run = query.state.data;
            if (!run) return false;
            return isTerminal(run.status) ? false : POLL_MS;
        },
    });
}

export function useEvalHistory(
    questionDataset: string | undefined
): UseQueryResult<EvalHistoryEntry[]> {
    return useQuery({
        queryKey: keys.history(questionDataset ?? ""),
        queryFn: ({ signal }) =>
            api.get<EvalHistoryEntry[]>(
                `/admin/evals/history?question_dataset=${encodeURIComponent(questionDataset ?? "")}`,
                signal
            ),
        enabled: Boolean(questionDataset),
    });
}

export function useEvalMutations() {
    const client = useQueryClient();

    const start = useMutation({
        mutationFn: (payload: EvalRunCreate) =>
            api.post<EvalRun>("/admin/evals/runs", payload),
        onSuccess: () => {
            void client.invalidateQueries({ queryKey: keys.runs });
        },
    });

    const cancel = useMutation({
        mutationFn: (runId: string) =>
            api.post<EvalRun>(
                `/admin/evals/runs/${encodeURIComponent(runId)}/cancel`
            ),
        onSuccess: (run) => {
            client.setQueryData(keys.run(run.run_id), run);
            void client.invalidateQueries({ queryKey: keys.runs });
        },
    });

    return { start, cancel };
}
