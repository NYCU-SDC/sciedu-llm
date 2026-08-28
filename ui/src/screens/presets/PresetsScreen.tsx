import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";

import { api } from "../../api/client";
import {
    keys,
    usePresetMutations,
    usePresetReport,
    usePresets,
} from "../../api/hooks";
import type { Preset, PresetDetail, PresetSummary } from "../../api/types";
import { ErrorPanel, QueryError } from "../../components/ErrorPanel";
import { Loading, PageHeader } from "../../components/States";
import { PresetSourceTag } from "../../components/StatusTag";
import { formatUnixSeconds, pluralise } from "../../lib/format";
import { ImportPresetsDialog } from "./ImportPresetsDialog";
import { describeRagMode } from "./presetShape";

export function PresetsScreen() {
    const navigate = useNavigate();
    const presets = usePresets();
    const report = usePresetReport();
    const { refresh } = usePresetMutations();
    const [importing, setImporting] = useState(false);

    // `GET /admin/presets` answers names and provenance only; the model, cast,
    // tool count and RAG mode live on the document, so the table fills its
    // remaining columns from one detail request per preset. Those reads are
    // served from the registry's in-memory map — no Langfuse round trip.
    const details = useQueries({
        queries: (presets.data ?? []).map((summary) => ({
            queryKey: keys.preset(summary.name),
            queryFn: ({ signal }: { signal: AbortSignal }) =>
                api.get<PresetDetail>(
                    `/admin/presets/${encodeURIComponent(summary.name)}`,
                    signal,
                ),
        })),
    });

    const documents = new Map<string, Preset>();
    (presets.data ?? []).forEach((summary, index) => {
        const document = details[index]?.data?.definition;
        if (document) documents.set(summary.name, document);
    });

    const errorEntries = Object.entries(report.data?.errors ?? {});

    return (
        <>
            <PageHeader
                kicker="config/presets"
                title="行為預設"
                lede="預設值是助理的一組命名行為：使用哪個模型、各角色採用哪個提示詞、可使用哪些工具，以及是否搜尋課程教材。"
                actions={
                    <>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            disabled={refresh.isPending}
                            onClick={() => refresh.mutate()}
                        >
                            {refresh.isPending
                                ? "重新載入中…"
                                : "從 Langfuse 重新載入"}
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => setImporting(true)}
                        >
                            匯入預設值
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => void navigate("/presets/new")}
                        >
                            新增預設值
                        </button>
                    </>
                }
            />

            <LoadBanner
                loading={!report.data && !report.isError}
                loaded={report.data?.loaded.length ?? null}
                fetchedAt={report.data?.fetched_at ?? null}
                rejected={errorEntries.length}
            />

            {report.isError && (
                <div style={{ marginTop: 14 }}>
                    <QueryError
                        what="無法從 Langfuse 重新載入預設值"
                        error={report.error}
                    />
                </div>
            )}

            <div className="panel table-wrap" style={{ marginTop: 14 }}>
                {presets.isError ? (
                    <QueryError what="無法列出預設值" error={presets.error} />
                ) : !presets.data ? (
                    <Loading what="預設值" />
                ) : presets.data.length === 0 ? (
                    <p className="quiet">
                        No presets are being served. That should not happen —
                        the built-ins are code-defined and always available.
                    </p>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>預設值</th>
                                <th>模型</th>
                                <th>角色群</th>
                                <th>課程教材</th>
                                <th>工具</th>
                                <th>來源</th>
                                <th />
                            </tr>
                        </thead>
                        <tbody>
                            {presets.data?.map((summary) => (
                                <PresetRow
                                    key={summary.name}
                                    summary={summary}
                                    document={documents.get(summary.name)}
                                />
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
            <p className="note" style={{ marginTop: 10 }}>
                Built-in presets ship with the service and can always be
                restored. A preset stored in Langfuse with the same name takes
                over from the built-in one; delete it and the built-in comes
                back.
            </p>

            {errorEntries.length > 0 && (
                <div style={{ marginTop: 20 }}>
                    <ErrorPanel
                        title={`${pluralise(errorEntries.length, "entry", "entries")} in config/presets could not be loaded`}
                        copyText={errorEntries
                            .map(([id, message]) => `item ${id}\n${message}`)
                            .join("\n\n")}
                    >
                        <div className="alarm-body mono">
                            {errorEntries.map(([itemId, message]) => (
                                <div key={itemId} style={{ marginBottom: 8 }}>
                                    <strong>item {itemId}</strong>
                                    <br />
                                    {message}
                                </div>
                            ))}
                        </div>
                        <p className="alarm-body" style={{ marginTop: 4 }}>
                            Those items were skipped; everything else stayed in
                            service. Fix them in the Langfuse dataset, or open
                            the preset here and re-save it.
                        </p>
                    </ErrorPanel>
                </div>
            )}

            {importing && (
                <ImportPresetsDialog onClose={() => setImporting(false)} />
            )}
        </>
    );
}

function PresetRow({
    summary,
    document,
}: {
    summary: PresetSummary;
    document: Preset | undefined;
}) {
    const cast = document
        ? document.characters.length > 1
            ? `orchestrator + ${document.characters.length - 1}`
            : "single"
        : "…";
    const toolCount = document
        ? document.characters.reduce(
              (total, character) => total + character.tools.length,
              0,
          )
        : null;

    return (
        <tr>
            <td>
                <Link
                    to={`/presets/${encodeURIComponent(summary.name)}`}
                    className="mono"
                    style={{ fontSize: 13, fontWeight: 600 }}
                >
                    {summary.name}
                </Link>
                {summary.description && (
                    <span className="cell-sub">{summary.description}</span>
                )}
            </td>
            <td className="mono" style={{ fontSize: 12.5 }}>
                {document ? (document.model ?? "伺服器預設值") : "…"}
            </td>
            <td style={{ fontSize: 13 }}>{cast}</td>
            <td style={{ fontSize: 13 }}>
                {document ? describeRagMode(document) : "…"}
            </td>
            <td className="mono" style={{ fontSize: 12.5 }}>
                {toolCount === null ? "…" : toolCount === 0 ? "—" : toolCount}
            </td>
            <td>
                <PresetSourceTag
                    builtin={summary.builtin}
                    shadowed={summary.shadowed_builtin}
                />
            </td>
            <td className="right">
                <Link
                    className="btn btn-ghost"
                    to={`/presets/${encodeURIComponent(summary.name)}`}
                >
                    {summary.builtin && !summary.shadowed_builtin
                        ? "檢視"
                        : "編輯"}
                </Link>
            </td>
        </tr>
    );
}

/** What the last registry load produced. The backend has no `GET /load-report`
 * on purpose — `POST /refresh` is the idempotent way to ask, so this line is
 * the result of that call. */
function LoadBanner({
    loading,
    loaded,
    fetchedAt,
    rejected,
}: {
    loading: boolean;
    loaded: number | null;
    fetchedAt: number | null;
    rejected: number;
}) {
    if (loading) {
        return (
            <div className="banner banner-idle" style={{ marginTop: 20 }}>
                <span className="banner-led" />
                <div className="banner-body">
                    <div className="banner-title">正在讀取預設值資料集…</div>
                </div>
            </div>
        );
    }
    if (loaded === null) return null;

    const failedFetch = fetchedAt === null;
    return (
        <div
            className={`banner ${failedFetch ? "banner-idle" : "banner-good"}`}
            style={{ marginTop: 20 }}
        >
            <span className="banner-led" />
            <div className="banner-body">
                <div className="banner-title">
                    {failedFetch
                        ? "無法讀取 Langfuse — 服務仍使用已載入的預設值"
                        : `上次載入時間：${formatUnixSeconds(fetchedAt)}`}
                </div>
                <div className="banner-line">
                    <strong>載入了 {loaded} 份範本</strong>
                    {rejected > 0 &&
                        `, ${pluralise(rejected, "entry", "entries")} rejected`}
                </div>
            </div>
            {rejected > 0 && <span className="tag tag-outline">請見下方</span>}
        </div>
    );
}
