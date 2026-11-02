import { useMemo, useState } from "react";

import { errorMessage } from "../../api/errors";
import {
    useModels,
    usePrompts,
    useRagSettings,
    useRagSettingsMutations,
} from "../../api/hooks";
import type { NamedResource } from "../../api/types";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { QueryError } from "../../components/ErrorPanel";
import { Field, Panel } from "../../components/Panel";
import { Loading } from "../../components/States";
import { RebuildTag } from "../../components/StatusTag";
import {
    buildUpdate,
    diffDraft,
    toDraft,
    type NumericKey,
    type RagDraft,
} from "./draft";

/** The plain-language notes the mockup puts beside each retrieval knob. */
const KNOBS: { key: NumericKey; label: string; note: string }[] = [
    {
        key: "bm25_top_n",
        label: "關鍵字前 n 筆",
        note: "從精確措辭搜尋（BM25）保留的段落。",
    },
    {
        key: "dense_top_n",
        label: "語意前 n 筆",
        note: "從語意相近搜尋保留的段落。",
    },
    {
        key: "rrf_k",
        label: "合併常數（RRF k）",
        note: "兩份結果清單的混合強度；數值越高越平緩。",
    },
    {
        key: "rerank_pool_size",
        label: "重排序候選集",
        note: "合併後會再次閱讀與重新排序的段落數量。",
    },
    {
        key: "final_k",
        label: "最終 k",
        note: "老師回答前實際閱讀的段落數量。",
    },
];

export function RagSettingsPanel() {
    const config = useRagSettings();
    const models = useModels();
    const prompts = usePrompts();
    const { apply, reset } = useRagSettingsMutations();

    const [draft, setDraft] = useState<RagDraft | null>(null);
    const [confirmReset, setConfirmReset] = useState(false);

    const server = useMemo(
        () => (config.data ? toDraft(config.data) : null),
        [config.data]
    );

    // The draft is seeded from the server snapshot and only exists once the user
    // has touched something; that way a background refetch cannot clobber an edit
    // in progress, and "no draft" genuinely means "no unsaved changes".
    const current = draft ?? server;
    const changes = useMemo(
        () => (server && draft ? diffDraft(server, draft) : []),
        [server, draft]
    );
    const built = useMemo(
        () => (draft ? buildUpdate(changes, draft) : null),
        [changes, draft]
    );

    const busy = apply.isPending || reset.isPending;
    const rebuildCount = changes.filter((change) => change.rebuilds).length;
    const rebuildsOnApply = rebuildCount > 0;

    const set = <K extends keyof RagDraft>(key: K, value: RagDraft[K]) => {
        setDraft((previous) => {
            const base = previous ?? server;
            if (!base) return previous;
            return { ...base, [key]: value };
        });
    };

    const problemFor = (key: keyof RagDraft) =>
        built?.problems.find((problem) => problem.key === key)?.message;

    const onApply = () => {
        if (!built || built.problems.length > 0) return;
        apply.mutate(
            { ...built.update, rebuild: rebuildsOnApply },
            { onSuccess: () => setDraft(null) }
        );
    };

    if (config.isError) {
        return (
            <QueryError what="無法讀取伺服器 RAG 設定" error={config.error} />
        );
    }
    if (!config.data || !current) return <Loading what="伺服器 RAG 設定" />;

    const modelOptions = models.data?.models.map((model) => model.id) ?? [];

    return (
        <>
            <Header disabled={busy} onReset={() => setConfirmReset(true)} />
            {(apply.error || reset.error) && (
                <QueryError
                    what="無法套用伺服器 RAG 設定"
                    error={apply.error ?? reset.error}
                />
            )}
            <div className="split split-wide" style={{ marginTop: 22 }}>
                <div className="col">
                    <Panel title="模型">
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: 14,
                            }}
                        >
                            <div
                                className="grid-2"
                                style={{ alignItems: "start" }}
                            >
                                <div>
                                    <Field label="嵌入模型">
                                        {(id) => (
                                            <ModelPicker
                                                id={id}
                                                value={current.embedding_model}
                                                options={modelOptions}
                                                disabled={busy}
                                                onChange={(value) =>
                                                    set(
                                                        "embedding_model",
                                                        value
                                                    )
                                                }
                                            />
                                        )}
                                    </Field>
                                    <div style={{ marginTop: 7 }}>
                                        <RebuildTag>會重建索引</RebuildTag>
                                    </div>
                                </div>
                                <p className="note" style={{ paddingTop: 19 }}>
                                    將文件轉為數字表示，以找出語意相近的段落。變更後必須重新讀取所有文件。
                                </p>
                            </div>
                            <div
                                className="grid-2"
                                style={{ alignItems: "start" }}
                            >
                                <Field label="重排序模型">
                                    {(id) => (
                                        <ModelPicker
                                            id={id}
                                            value={current.rerank_model}
                                            options={modelOptions}
                                            disabled={busy}
                                            onChange={(value) =>
                                                set("rerank_model", value)
                                            }
                                        />
                                    )}
                                </Field>
                                <p className="note" style={{ paddingTop: 19 }}>
                                    第二道處理會依段落回答問題的程度重新排序候選段落，可隨時變更。
                                </p>
                            </div>
                        </div>
                        {models.isError && (
                            <p className="note" style={{ marginTop: 12 }}>
                                模型清單無法取得（{errorMessage(models.error)}
                                ），目前改為自由輸入欄位。
                            </p>
                        )}
                    </Panel>

                    <Panel title="文件切分方式">
                        <p className="note" style={{ marginBottom: 14 }}>
                            長文件會切成彼此重疊的片段。較小的片段更精確；較大的片段會保留更多上下文。
                        </p>
                        <div className="grid-2">
                            <NumberField
                                label="片段大小（字元）"
                                value={current.chunk_size}
                                problem={problemFor("chunk_size")}
                                disabled={busy}
                                onChange={(value) => set("chunk_size", value)}
                            />
                            <NumberField
                                label="片段重疊（字元）"
                                value={current.chunk_overlap}
                                problem={problemFor("chunk_overlap")}
                                disabled={busy}
                                onChange={(value) =>
                                    set("chunk_overlap", value)
                                }
                            />
                        </div>
                        <div style={{ marginTop: 12 }}>
                            <RebuildTag>變更這些設定會重建索引</RebuildTag>
                        </div>
                    </Panel>

                    <Panel title="段落搜尋方式">
                        <p className="note" style={{ marginBottom: 14 }}>
                            關鍵字與語意搜尋會並行執行，再將結果合併、篩選與重新排序後交給老師。
                        </p>
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: 8,
                            }}
                        >
                            {KNOBS.map((knob) => (
                                <div className="knob-row" key={knob.key}>
                                    <label htmlFor={`knob-${knob.key}`}>
                                        {knob.label}
                                    </label>
                                    <input
                                        id={`knob-${knob.key}`}
                                        className={`input mono input-sunken${problemFor(knob.key) ? " input-invalid" : ""}`}
                                        inputMode="numeric"
                                        value={current[knob.key]}
                                        disabled={busy}
                                        onChange={(event) =>
                                            set(knob.key, event.target.value)
                                        }
                                    />
                                    <span className="note">
                                        {knob.note}
                                        {problemFor(knob.key) && (
                                            <span
                                                style={{
                                                    display: "block",
                                                    color: "var(--color-alarm-ink)",
                                                }}
                                            >
                                                {problemFor(knob.key)}
                                            </span>
                                        )}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </Panel>

                    <div className="grid-2" style={{ gap: 14 }}>
                        <Panel title="回答提示詞">
                            <div
                                style={{
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 12,
                                }}
                            >
                                <PromptField
                                    label="系統提示詞（Langfuse）"
                                    value={current.generator_system_prompt_name}
                                    options={prompts.data ?? []}
                                    problem={problemFor(
                                        "generator_system_prompt_name"
                                    )}
                                    disabled={busy}
                                    onChange={(value) =>
                                        set(
                                            "generator_system_prompt_name",
                                            value
                                        )
                                    }
                                />
                                <PromptField
                                    label="使用者提示詞（Langfuse）"
                                    value={current.generator_user_prompt_name}
                                    options={prompts.data ?? []}
                                    problem={problemFor(
                                        "generator_user_prompt_name"
                                    )}
                                    disabled={busy}
                                    onChange={(value) =>
                                        set("generator_user_prompt_name", value)
                                    }
                                />
                            </div>
                            <p className="note" style={{ marginTop: 12 }}>
                                提示詞內容儲存在
                                Langfuse；這裡只指定要取得哪個提示詞。
                            </p>
                            {prompts.isError && (
                                <p
                                    className="note"
                                    style={{ color: "var(--color-alarm-ink)" }}
                                >
                                    無法載入提示詞清單 —{" "}
                                    {errorMessage(prompts.error)}
                                </p>
                            )}
                        </Panel>
                        <Panel title="處理量">
                            <div
                                style={{
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 12,
                                }}
                            >
                                <NumberField
                                    label="嵌入批次大小"
                                    value={current.embedding_batch_size}
                                    problem={problemFor("embedding_batch_size")}
                                    disabled={busy}
                                    onChange={(value) =>
                                        set("embedding_batch_size", value)
                                    }
                                />
                                <NumberField
                                    label="最大並行數"
                                    value={current.max_concurrency}
                                    problem={problemFor("max_concurrency")}
                                    disabled={busy}
                                    onChange={(value) =>
                                        set("max_concurrency", value)
                                    }
                                />
                            </div>
                            <p className="note" style={{ marginTop: 12 }}>
                                此設定決定重建時對模型伺服器的負載。若因速率限制而失敗，請調低數值。
                            </p>
                        </Panel>
                    </div>

                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 18,
                            flexWrap: "wrap",
                            padding: "4px 2px",
                        }}
                    >
                        <button
                            type="button"
                            className="btn btn-primary"
                            style={{ fontSize: 14.5, padding: "11px 22px" }}
                            disabled={
                                busy ||
                                changes.length === 0 ||
                                (built?.problems.length ?? 0) > 0
                            }
                            onClick={onApply}
                        >
                            {apply.isPending ? "套用中…" : "套用至執行中的服務"}
                        </button>
                        {changes.length > 0 && (
                            <p className="note" style={{ maxWidth: "46ch" }}>
                                {rebuildsOnApply
                                    ? `${rebuildCount === 1 ? "有一項變更需要" : `有 ${rebuildCount} 項變更需要`}重建索引。缺少的索引會在背景建立；請至各預設值頁面查看或停止建置。`
                                    : "目前的變更不需要重建索引。"}
                            </p>
                        )}
                    </div>
                </div>

                <aside className="panel sticky-aside">
                    <h5 className="sect" style={{ margin: 0 }}>
                        未儲存的變更
                    </h5>
                    <div
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 10,
                            marginTop: 12,
                        }}
                    >
                        {changes.length === 0 ? (
                            <p className="note">
                                尚未編輯任何設定。畫面顯示的就是服務目前使用的設定。
                            </p>
                        ) : (
                            <>
                                {changes.map((change) => (
                                    <div className="tile" key={change.key}>
                                        <div
                                            style={{
                                                fontSize: 13,
                                                fontWeight: 600,
                                            }}
                                        >
                                            {change.label}
                                        </div>
                                        <div
                                            className="mono"
                                            style={{
                                                fontSize: 12,
                                                marginTop: 1,
                                            }}
                                        >
                                            <span
                                                style={{
                                                    textDecoration:
                                                        "line-through",
                                                    color: "var(--color-neutral-500)",
                                                }}
                                            >
                                                {change.from}
                                            </span>{" "}
                                            →{" "}
                                            <span
                                                style={{
                                                    color: "var(--color-accent-700)",
                                                    fontWeight: 600,
                                                }}
                                            >
                                                {change.to}
                                            </span>
                                        </div>
                                        {change.rebuilds && (
                                            <div style={{ marginTop: 6 }}>
                                                <RebuildTag>
                                                    會重建索引
                                                </RebuildTag>
                                            </div>
                                        )}
                                    </div>
                                ))}
                                <p className="note">
                                    按下「套用」前，不會將任何內容傳送至服務。
                                </p>
                                <button
                                    type="button"
                                    className="btn btn-ghost"
                                    style={{ alignSelf: "flex-start" }}
                                    disabled={busy}
                                    onClick={() => setDraft(null)}
                                >
                                    捨棄變更
                                </button>
                            </>
                        )}
                    </div>
                </aside>
            </div>

            {confirmReset && (
                <ConfirmDialog
                    title="要還原為伺服器預設值嗎？"
                    body={
                        <>
                            此程序持有的所有覆寫設定都會移除，並還原服務
                            <span className="mono">RAG_*</span>{" "}
                            環境變數的值。符合設定的索引快取會直接使用，其餘預設值會在背景建置索引。
                        </>
                    }
                    confirmLabel="還原設定"
                    busy={reset.isPending}
                    onCancel={() => setConfirmReset(false)}
                    onConfirm={() => {
                        setConfirmReset(false);
                        reset.mutate(undefined, {
                            onSuccess: () => setDraft(null),
                        });
                    }}
                />
            )}
        </>
    );
}

function Header({
    disabled,
    onReset,
}: {
    disabled: boolean;
    onReset: () => void;
}) {
    return (
        <div
            style={{
                display: "flex",
                alignItems: "start",
                justifyContent: "space-between",
                gap: 18,
                flexWrap: "wrap",
            }}
        >
            <div>
                <h3 style={{ margin: 0 }}>RAG 設定</h3>
                <p className="note" style={{ marginTop: 8 }}>
                    所有預設值共用這些檢索與建置設定。教材選擇與索引建置狀態請至各預設值頁面查看。
                </p>
            </div>
            <button
                type="button"
                className="btn btn-secondary"
                disabled={disabled}
                onClick={onReset}
            >
                還原伺服器預設值
            </button>
        </div>
    );
}

function ModelPicker({
    id,
    value,
    options,
    disabled,
    onChange,
}: {
    id: string;
    value: string;
    options: string[];
    disabled?: boolean;
    onChange: (value: string) => void;
}) {
    // The upstream listing does not always contain a model the service is
    // already configured with, so the current value is always offered.
    const all = options.includes(value) ? options : [value, ...options];
    if (options.length === 0) {
        return (
            <input
                id={id}
                className="input mono"
                value={value}
                disabled={disabled}
                onChange={(event) => onChange(event.target.value)}
            />
        );
    }
    return (
        <select
            id={id}
            className="input mono"
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
        >
            {all.map((option) => (
                <option key={option} value={option}>
                    {option}
                </option>
            ))}
        </select>
    );
}

function NumberField({
    label,
    value,
    problem,
    disabled,
    onChange,
}: {
    label: string;
    value: string;
    problem?: string;
    disabled?: boolean;
    onChange: (value: string) => void;
}) {
    return (
        <Field label={label} hint={problem ? <Bad>{problem}</Bad> : undefined}>
            {(id) => (
                <input
                    id={id}
                    className={`input mono${problem ? " input-invalid" : ""}`}
                    inputMode="numeric"
                    value={value}
                    disabled={disabled}
                    onChange={(event) => onChange(event.target.value)}
                />
            )}
        </Field>
    );
}

function PromptField({
    label,
    value,
    options,
    problem,
    disabled,
    onChange,
}: {
    label: string;
    value: string;
    options: NamedResource[];
    problem?: string;
    disabled?: boolean;
    onChange: (value: string) => void;
}) {
    const names = options.map((option) => option.name);
    const all = names.includes(value) ? names : [value, ...names];
    return (
        <Field label={label} hint={problem ? <Bad>{problem}</Bad> : undefined}>
            {(id) => (
                <select
                    id={id}
                    className={`input mono${problem ? " input-invalid" : ""}`}
                    value={value}
                    disabled={disabled}
                    onChange={(event) => onChange(event.target.value)}
                >
                    {all.map((option) => (
                        <option key={option} value={option}>
                            {option}
                        </option>
                    ))}
                </select>
            )}
        </Field>
    );
}

function Bad({ children }: { children: React.ReactNode }) {
    return <span style={{ color: "var(--color-alarm-ink)" }}>{children}</span>;
}
