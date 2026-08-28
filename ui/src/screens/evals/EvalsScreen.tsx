import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { errorMessage } from "../../api/errors";
import {
    useDatasets,
    useEvalMutations,
    useEvalRuns,
    useJudgePrompts,
    useModels,
} from "../../api/hooks";
import type { EvalRun, EvalRunCreate, NamedResource } from "../../api/types";
import { isTerminal } from "../../api/types";
import { CheckList, type Choice } from "../../components/Choices";
import { QueryError } from "../../components/ErrorPanel";
import { FolderDatasetPicker } from "../../components/FolderDatasetPicker";
import { Field, Panel } from "../../components/Panel";
import { Loading, PageHeader } from "../../components/States";
import { RunStatusTag } from "../../components/StatusTag";
import { formatDuration, formatTime, joinNames } from "../../lib/format";

/** What "Run again with these settings" hands over from the run detail. */
export interface EvalPrefill {
    eval_model: string;
    judge_model: string;
    corpus_datasets: string[];
    question_datasets: string[];
    judge_prompts: string[];
    k: number;
    embedding_model: string;
    rerank_model: string;
    chunk_size: number;
    chunk_overlap: number;
}

interface FormState {
    evalModel: string;
    judgeModel: string;
    k: string;
    corpus: string[];
    questions: string[];
    prompts: string[];
    embeddingModel: string;
    rerankModel: string;
    chunkSize: string;
    chunkOverlap: string;
}

const EMPTY: FormState = {
    evalModel: "",
    judgeModel: "",
    k: "5",
    corpus: [],
    questions: [],
    prompts: [],
    embeddingModel: "",
    rerankModel: "",
    chunkSize: "",
    chunkOverlap: "",
};

export function EvalsScreen() {
    const navigate = useNavigate();
    const location = useLocation();
    const prefill = (location.state as { prefill?: EvalPrefill } | null)
        ?.prefill;

    const models = useModels();
    const datasets = useDatasets();
    const prompts = useJudgePrompts();
    const runs = useEvalRuns();
    const { start, cancel } = useEvalMutations();

    const [form, setForm] = useState<FormState>(() =>
        prefill
            ? {
                  evalModel: prefill.eval_model,
                  judgeModel: prefill.judge_model,
                  k: String(prefill.k),
                  corpus: prefill.corpus_datasets,
                  questions: prefill.question_datasets,
                  prompts: prefill.judge_prompts,
                  embeddingModel: prefill.embedding_model,
                  rerankModel: prefill.rerank_model,
                  chunkSize: String(prefill.chunk_size),
                  chunkOverlap: String(prefill.chunk_overlap),
              }
            : EMPTY,
    );
    const [localError, setLocalError] = useState<string | null>(null);

    // The two model pickers fall back to the server's own defaults until the user
    // picks something — derived during render rather than written into state, so
    // a slow /admin/models cannot overwrite a choice already made.
    const defaults = models.data?.defaults;
    const evalModel = form.evalModel || defaults?.eval_model || "";
    const judgeModel = form.judgeModel || defaults?.judge_model || "";

    // A prefill arrives through history state; clear it so a refresh does not
    // silently re-apply settings the user has since edited.
    useEffect(() => {
        if (prefill) void navigate(".", { replace: true, state: null });
    }, [prefill, navigate]);

    const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
        setForm((previous) => ({ ...previous, [key]: value }));

    const toggle = (key: "prompts") => (value: string, next: boolean) => {
        setForm((previous) => {
            const chosen = new Set(previous[key]);
            if (next) chosen.add(value);
            else chosen.delete(value);
            return { ...previous, [key]: [...chosen] };
        });
    };

    const submit = () => {
        setLocalError(null);
        if (!evalModel || !judgeModel) {
            setLocalError("請選擇受測模型與評分模型。");
            return;
        }
        if (form.corpus.length === 0) {
            setLocalError("請至少選擇一個要建立索引的語料庫資料集。");
            return;
        }
        if (form.questions.length === 0) {
            setLocalError("請至少選擇一個題目集。");
            return;
        }
        if (form.prompts.length === 0) {
            setLocalError("請至少選擇一個評分提示詞。");
            return;
        }
        const k = Number(form.k);
        if (!/^\d+$/.test(form.k.trim()) || k < 1 || k > 20) {
            setLocalError("k 必須是介於 1 到 20 的整數。");
            return;
        }

        const payload: EvalRunCreate = {
            eval_model: evalModel,
            judge_model: judgeModel,
            corpus_datasets: form.corpus,
            question_datasets: form.questions,
            judge_prompts: form.prompts,
            k,
        };
        // An override left blank means "use the server's RAG configuration", which
        // is exactly what omitting the field does.
        if (form.embeddingModel.trim())
            payload.embedding_model = form.embeddingModel.trim();
        if (form.rerankModel.trim())
            payload.rerank_model = form.rerankModel.trim();
        if (form.chunkSize.trim()) payload.chunk_size = Number(form.chunkSize);
        if (form.chunkOverlap.trim())
            payload.chunk_overlap = Number(form.chunkOverlap);

        start.mutate(payload, {
            onSuccess: (run) =>
                void navigate(`/evals/runs/${encodeURIComponent(run.run_id)}`),
        });
    };

    const modelIds = models.data?.models ?? [];

    return (
        <>
            <PageHeader
                kicker="評分執行"
                title="評估"
                lede="以一組題目測試課程教材，再由第二個模型為答案評分。分數與完整逐字稿會保留在 Langfuse。"
            />

            <Panel title="開始執行" style={{ marginTop: 20 }}>
                {/* Three rows of two, read left to right: who answers and who scores,
            what they read and what they are asked, then how much of it each
            question gets and what the scoring is judged against. */}
                <div className="pair" style={{ rowGap: 20 }}>
                    <Field label="受測模型">
                        {(id) => (
                            <ModelSelect
                                id={id}
                                value={evalModel}
                                options={modelIds}
                                onChange={(value) => set("evalModel", value)}
                            />
                        )}
                    </Field>
                    <Field label="評分模型">
                        {(id) => (
                            <ModelSelect
                                id={id}
                                value={judgeModel}
                                options={modelIds}
                                onChange={(value) => set("judgeModel", value)}
                            />
                        )}
                    </Field>

                    <DatasetField
                        label="課程教材"
                        resources={datasets.data?.corpus}
                        loading={!datasets.data}
                        error={datasets.isError ? datasets.error : null}
                        empty="corpus 資料夾下沒有資料集。"
                        selected={form.corpus}
                        onChange={(next) => set("corpus", next)}
                    />
                    <DatasetField
                        label="題目集"
                        resources={datasets.data?.questions}
                        loading={!datasets.data}
                        error={datasets.isError ? datasets.error : null}
                        empty="questions 資料夾下沒有資料集。"
                        selected={form.questions}
                        onChange={(next) => set("questions", next)}
                    />

                    <Field
                        label="每題段落數（k）"
                        hint="每個問題回答時使用的段落數量，介於 1 到 20。"
                    >
                        {(id) => (
                            <input
                                id={id}
                                className="input mono"
                                inputMode="numeric"
                                value={form.k}
                                onChange={(event) =>
                                    set("k", event.target.value)
                                }
                            />
                        )}
                    </Field>
                    <PickerField
                        label="評分提示詞"
                        resources={prompts.data}
                        loading={!prompts.data}
                        error={prompts.isError ? prompts.error : null}
                        errorWhat="無法列出評分提示詞"
                        empty="judge 資料夾下沒有提示詞。"
                        selected={form.prompts}
                        onToggle={toggle("prompts")}
                    />
                </div>

                {models.isError && (
                    <div style={{ marginTop: 14 }}>
                        <QueryError what="無法列出模型" error={models.error} />
                    </div>
                )}

                <details style={{ marginTop: 16 }}>
                    <summary>
                        本次執行的檢索設定（未設定時使用即時設定）
                    </summary>
                    <div className="grid-4" style={{ marginTop: 12 }}>
                        <Field label="嵌入模型">
                            {(id) => (
                                <ModelSelect
                                    id={id}
                                    value={form.embeddingModel}
                                    options={modelIds}
                                    placeholder="— 即時設定 —"
                                    onChange={(value) =>
                                        set("embeddingModel", value)
                                    }
                                />
                            )}
                        </Field>
                        <Field label="重排序模型">
                            {(id) => (
                                <ModelSelect
                                    id={id}
                                    value={form.rerankModel}
                                    options={modelIds}
                                    placeholder="— 即時設定 —"
                                    onChange={(value) =>
                                        set("rerankModel", value)
                                    }
                                />
                            )}
                        </Field>
                        <Field label="片段大小">
                            {(id) => (
                                <input
                                    id={id}
                                    className="input mono input-sunken"
                                    inputMode="numeric"
                                    placeholder="即時設定"
                                    value={form.chunkSize}
                                    onChange={(event) =>
                                        set("chunkSize", event.target.value)
                                    }
                                />
                            )}
                        </Field>
                        <Field label="片段重疊">
                            {(id) => (
                                <input
                                    id={id}
                                    className="input mono input-sunken"
                                    inputMode="numeric"
                                    placeholder="即時設定"
                                    value={form.chunkOverlap}
                                    onChange={(event) =>
                                        set("chunkOverlap", event.target.value)
                                    }
                                />
                            )}
                        </Field>
                    </div>
                </details>
            </Panel>

            {localError && (
                <p
                    className="note"
                    style={{ marginTop: 14, color: "var(--color-alarm-ink)" }}
                >
                    {localError}
                </p>
            )}
            {start.error && (
                <div style={{ marginTop: 14 }}>
                    <QueryError
                        what="服務無法開始此次執行"
                        error={start.error}
                    />
                </div>
            )}

            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 18,
                    marginTop: 18,
                    flexWrap: "wrap",
                }}
            >
                <button
                    type="button"
                    className="btn btn-primary"
                    style={{ fontSize: 14.5, padding: "11px 22px" }}
                    onClick={submit}
                    disabled={start.isPending}
                >
                    {start.isPending ? "正在開始..." : "開始評估"}
                </button>
            </div>

            <h5 className="sect" style={{ margin: "26px 0 10px" }}>
                執行紀錄
            </h5>
            <div className="panel table-wrap">
                {runs.isError ? (
                    <QueryError
                        what="無法列出執行紀錄"
                        error={runs.error}
                    />
                ) : !runs.data ? (
                    <Loading what="評估狀態" />
                ) : runs.data.length === 0 ? (
                    <p className="quiet">
                        這個程序中沒有任何執行紀錄，開始一個評估後結果就會出現在列表中。
                    </p>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>名稱</th>
                                <th>狀態</th>
                                <th>評分</th>
                                <th>問題集</th>
                                <th>段落數</th>
                                <th>開始時間</th>
                                <th>執行時長</th>
                                <th />
                            </tr>
                        </thead>
                        <tbody>
                            {runs.data?.map((run) => (
                                <RunRow
                                    key={run.run_id}
                                    run={run}
                                    onCancel={() => cancel.mutate(run.run_id)}
                                    cancelling={
                                        cancel.isPending &&
                                        cancel.variables === run.run_id
                                    }
                                />
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
            {cancel.error && (
                <div style={{ marginTop: 14 }}>
                    <QueryError
                        what="無法取消該次執行"
                        error={cancel.error}
                    />
                </div>
            )}
        </>
    );
}

function RunRow({
    run,
    onCancel,
    cancelling,
}: {
    run: EvalRun;
    onCancel: () => void;
    cancelling: boolean;
}) {
    const live = !isTerminal(run.status);
    return (
        <tr>
            <td>
                <Link
                    to={`/evals/runs/${encodeURIComponent(run.run_id)}`}
                    className="mono"
                    style={{ fontSize: 12.5, fontWeight: 600 }}
                >
                    {run.run_id}
                </Link>
            </td>
            <td>
                <RunStatusTag status={run.status} />
                {run.error && (
                    <span className="cell-sub mono" style={{ marginTop: 3 }}>
                        {run.error}
                    </span>
                )}
            </td>
            <td className="mono" style={{ fontSize: 12 }}>
                {run.eval_model} / {run.judge_model}
            </td>
            <td className="mono" style={{ fontSize: 12 }}>
                {joinNames(run.question_datasets)}
            </td>
            <td className="mono" style={{ fontSize: 12 }}>
                {run.k}
            </td>
            <td className="mono" style={{ fontSize: 12 }}>
                {formatTime(run.started_at)}
            </td>
            <td className="mono" style={{ fontSize: 12 }}>
                {formatDuration(run.duration_seconds)}
            </td>
            <td className="right">
                {live && (
                    <button
                        type="button"
                        className="btn btn-ghost btn-danger"
                        onClick={onCancel}
                        disabled={cancelling}
                    >
                        {cancelling ? "取消中…" : "取消"}
                    </button>
                )}
                <Link
                    className="btn btn-ghost"
                    to={`/evals/runs/${encodeURIComponent(run.run_id)}`}
                >
                    開啟
                </Link>
            </td>
        </tr>
    );
}

/** A dataset multiselect: the folder picker, plus the mono summary line the
 * mockup puts under every choice list. Full Langfuse names go back to the form;
 * the picker only folds how they are shown. */
function DatasetField({
    label,
    resources,
    loading,
    error,
    empty,
    selected,
    onChange,
}: {
    label: string;
    resources: NamedResource[] | undefined;
    loading: boolean;
    error: unknown;
    empty: string;
    selected: string[];
    onChange: (next: string[]) => void;
}) {
    return (
        <div className="field">
            <label>{label}</label>
            {error ? (
                <p className="note" style={{ color: "var(--color-alarm-ink)" }}>
                    無法列出 Langfuse 資料集 — {errorMessage(error)}
                </p>
            ) : loading ? (
                <p className="note">載入中…</p>
            ) : (
                <FolderDatasetPicker
                    items={(resources ?? []).map((resource) => ({
                        name: resource.name,
                        label: resource.label,
                    }))}
                    selected={selected}
                    onChange={onChange}
                    boxed
                    empty={empty}
                />
            )}
        </div>
    );
}

/** One multiselect column: a scrolling checkbox list plus the mono summary line
 * the mockup puts under it. */
function PickerField({
    label,
    resources,
    loading,
    error,
    errorWhat,
    empty,
    selected,
    onToggle,
}: {
    label: string;
    resources: NamedResource[] | undefined;
    /** True while nothing has arrived and nothing has failed. */
    loading: boolean;
    error: unknown;
    errorWhat: string;
    empty: string;
    selected: string[];
    onToggle: (value: string, next: boolean) => void;
}) {
    const choices: Choice[] = (resources ?? []).map((resource) => ({
        value: resource.name,
        label: (
            <span className="mono" style={{ fontSize: 12.5 }}>
                {resource.label}
            </span>
        ),
    }));

    return (
        <div className="field">
            <label>{label}</label>
            {error ? (
                <p className="note" style={{ color: "var(--color-alarm-ink)" }}>
                    {errorWhat} — {errorMessage(error)}
                </p>
            ) : loading ? (
                <p className="note">載入中…</p>
            ) : (
                <CheckList
                    choices={choices}
                    selected={selected}
                    onToggle={onToggle}
                    boxed
                    empty={empty}
                />
            )}
            <span className="choice-summary mono">
                {selected.length === 0
                    ? "尚未選取"
                    : joinNames(selected)}
            </span>
        </div>
    );
}

function ModelSelect({
    id,
    value,
    options,
    placeholder,
    onChange,
}: {
    id: string;
    value: string;
    options: string[];
    placeholder?: string;
    onChange: (value: string) => void;
}) {
    if (options.length === 0) {
        return (
            <input
                id={id}
                className="input mono"
                value={value}
                placeholder={placeholder}
                onChange={(event) => onChange(event.target.value)}
            />
        );
    }
    const all =
        value && !options.includes(value) ? [value, ...options] : options;
    return (
        <select
            id={id}
            className="input mono"
            value={value}
            onChange={(event) => onChange(event.target.value)}
        >
            <option value="">{placeholder ?? "— pick a model —"}</option>
            {all.map((option) => (
                <option key={option} value={option}>
                    {option}
                </option>
            ))}
        </select>
    );
}
