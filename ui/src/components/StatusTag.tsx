import type { RunStatus } from "../api/types";

/** Run-status tag colours, following the mockup: anything in flight wears the
 * terracotta accent, a completed run the sage second accent, a failure the
 * outline, a cancellation plain neutral. */
const RUN_TAG: Record<RunStatus, string> = {
    pending: "tag-neutral",
    building: "tag-accent",
    judging: "tag-accent",
    completed: "tag-accent-2",
    failed: "tag-outline",
    cancelled: "tag-neutral",
};

export function RunStatusTag({ status }: { status: string }) {
    const className = RUN_TAG[status as RunStatus] ?? "tag-neutral";
    const label: Record<string, string> = {
        pending: "等待中",
        building: "建立中",
        judging: "評分中",
        completed: "已完成",
        failed: "失敗",
        cancelled: "已取消",
    };
    return (
        <span className={`tag ${className}`}>{label[status] ?? status}</span>
    );
}

/** Where a preset's document came from — the `builtin` / `shadowed_builtin`
 * pair from `PresetSummary`. */
export function PresetSourceTag({
    builtin,
    shadowed,
}: {
    builtin: boolean;
    shadowed: boolean;
}) {
    if (!builtin) return <span className="tag tag-accent">來自 Langfuse</span>;
    if (shadowed) return <span className="tag tag-outline">覆寫內建預設</span>;
    return <span className="tag tag-neutral">內建</span>;
}

/** The tag the mockup puts next to any control whose value only lands after a
 * re-index. */
export function RebuildTag({
    children = "變更此項會重建索引",
}: {
    children?: string;
}) {
    return <span className="tag tag-accent">{children}</span>;
}
