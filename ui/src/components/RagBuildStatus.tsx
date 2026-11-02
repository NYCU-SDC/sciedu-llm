import type { RagBuildState } from "../api/types";
import { ErrorPanel } from "./ErrorPanel";
import { formatDuration, formatTime } from "../lib/format";

export function RagBuildStatus({
    isBuilt,
    datasetCount,
    build,
    stopping,
    pending,
    failure,
    onRetry,
    retryDisabled = false,
}: {
    isBuilt: boolean;
    datasetCount: number;
    build: RagBuildState;
    stopping: boolean;
    pending: string | null;
    failure: string | null;
    onRetry: () => void;
    retryDisabled?: boolean;
}) {
    if (build.status === "building") {
        return (
            <div
                className="banner banner-busy"
                style={{ marginTop: 20 }}
                aria-live="polite"
            >
                <span className="banner-led" />
                <div className="banner-body">
                    <div className="banner-title">
                        {stopping
                            ? "正在停止重建"
                            : "正在建立此預設值的共用索引"}
                        {isBuilt
                            ? " — 先前的索引仍可使用"
                            : " — 完成前無法提供此預設值的檢索回答"}
                    </div>
                    <div className="banner-line">
                        {`已耗時 ${formatDuration(build.duration_seconds)}；重建通常會需要數分鐘，在此期間你可以離開這個頁面。`}
                        {!stopping &&
                            "停止重建會影響共用此索引的預設值；其他索引不受影響。"}
                    </div>
                </div>
                <span className="tag tag-accent">建立中</span>
            </div>
        );
    }

    if (pending) {
        return (
            <div
                className="banner banner-busy"
                style={{ marginTop: 20 }}
                aria-live="polite"
            >
                <span className="banner-led" />
                <div className="banner-body">
                    <div className="banner-title">{pending}</div>
                    <div className="banner-line">
                        重建會在服務端進行，這個請求只負責啟動它。
                    </div>
                </div>
            </div>
        );
    }

    if (failure) {
        return (
            <div style={{ marginTop: 20 }}>
                <ErrorPanel
                    title="上次操作失敗"
                    detail={failure}
                    copyText={failure}
                    actions={
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={onRetry}
                            disabled={retryDisabled}
                        >
                            再次重建
                        </button>
                    }
                />
            </div>
        );
    }

    if (build.status === "failed") {
        const detail = build.error ?? "服務未提供原因。";
        return (
            <div style={{ marginTop: 20 }}>
                <ErrorPanel
                    title={
                        isBuilt
                            ? "重建失敗，先前的索引仍可使用"
                            : "索引建置失敗，尚無可用索引"
                    }
                    detail={detail}
                    copyText={detail}
                    actions={
                        <button
                            type="button"
                            className="btn btn-primary"
                            onClick={onRetry}
                            disabled={retryDisabled}
                        >
                            再次重建
                        </button>
                    }
                />
            </div>
        );
    }

    if (build.status === "cancelled") {
        return (
            <div className="banner banner-idle" style={{ marginTop: 20 }}>
                <span className="banner-led" />
                <div className="banner-body">
                    <div className="banner-title">已停止上次的重建</div>
                    <div className="banner-line">
                        {isBuilt
                            ? "老師仍可使用同一教材與建置設定的先前索引；全域設定保留不變。"
                            : "目前沒有可用的索引。請確認下方設定後重新建立。"}
                    </div>
                </div>
                <span className="tag tag-neutral">已停止</span>
            </div>
        );
    }

    if (!isBuilt) {
        return (
            <div className="banner banner-idle" style={{ marginTop: 20 }}>
                <span className="banner-led" />
                <div className="banner-body">
                    <div className="banner-title">尚未建立索引</div>
                    <div className="banner-line">
                        請先儲存此預設值的教材，再按 <strong>重建索引</strong>。
                    </div>
                </div>
                <span className="tag tag-neutral">未建立</span>
            </div>
        );
    }

    return (
        <div className="banner banner-good" style={{ marginTop: 20 }}>
            <span className="banner-led" />
            <div className="banner-body">
                <div className="banner-title">索引已建立並可供回答</div>
                <div className="banner-line mono">
                    已建立 {datasetCount} 個語料庫資料集的索引
                    {build.status === "completed" &&
                        ` · 上次重建 ${formatTime(build.finished_at)} 完成，耗時 ${formatDuration(build.duration_seconds)}`}
                </div>
            </div>
            <span className="tag tag-accent-2">正常</span>
        </div>
    );
}
