import { Link } from "react-router-dom";
import { useRagConfig, usePresetRagMutations } from "../../api/hooks";
import type { RagBuildState } from "../../api/types";
import { errorMessage } from "../../api/errors";
import { QueryError } from "../../components/ErrorPanel";
import { Loading } from "../../components/States";
import { RagBuildStatus } from "../../components/RagBuildStatus";

const IDLE: RagBuildState = {
    status: "idle",
    corpus_datasets: [],
    started_at: null,
    finished_at: null,
    duration_seconds: null,
    error: null,
    cancel_requested: false,
};

export function PresetRagIndexPanel({
    name,
    unsavedCorpus,
    savedRagEnabled,
}: {
    name?: string;
    unsavedCorpus: boolean;
    savedRagEnabled: boolean;
}) {
    const config = useRagConfig(name ?? "", Boolean(name && savedRagEnabled));
    const { rebuild, cancelBuild } = usePresetRagMutations(name ?? "");
    if (!name)
        return <p className="note">儲存預設值後會建立所選教材的索引。</p>;
    if (!savedRagEnabled)
        return (
            <p className="note">
                儲存並啟用 RAG 後，可在此查看教材索引與建置狀態。
            </p>
        );
    if (config.isError)
        return (
            <QueryError
                what="無法讀取此預設值的索引狀態"
                error={config.error}
            />
        );
    if (!config.data) return <Loading what="此預設值的索引" />;
    const data = config.data;
    const build = data.build ?? IDLE;
    const building = build.status === "building";
    const stopping = cancelBuild.isPending || build.cancel_requested;
    const failure = rebuild.error ?? cancelBuild.error;
    return (
        <div style={{ marginTop: 18 }}>
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    flexWrap: "wrap",
                }}
            >
                {building ? (
                    <button
                        className="btn btn-secondary"
                        type="button"
                        disabled={stopping}
                        onClick={() => cancelBuild.mutate()}
                    >
                        {stopping ? "正在停止…" : "停止重建"}
                    </button>
                ) : (
                    <button
                        className="btn btn-secondary"
                        type="button"
                        disabled={
                            unsavedCorpus ||
                            rebuild.isPending ||
                            data.corpus_datasets.length === 0
                        }
                        onClick={() => rebuild.mutate()}
                    >
                        {rebuild.isPending
                            ? "正在開始重建…"
                            : "重建此預設值的索引"}
                    </button>
                )}
                <Link className="link-btn" to="/server">
                    共用 RAG 設定
                </Link>
            </div>
            {unsavedCorpus && (
                <p className="note" style={{ marginTop: 10 }}>
                    請先儲存教材變更。下方狀態對應已儲存的教材。
                </p>
            )}
            <RagBuildStatus
                isBuilt={data.is_built}
                datasetCount={data.corpus_datasets.length}
                build={build}
                stopping={Boolean(stopping)}
                pending={rebuild.isPending ? "正在開始重建索引" : null}
                failure={failure ? errorMessage(failure) : null}
                retryDisabled={unsavedCorpus || rebuild.isPending}
                onRetry={() => rebuild.mutate()}
            />
        </div>
    );
}
