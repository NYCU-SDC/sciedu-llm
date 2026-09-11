import { useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plus, Trash2 } from "lucide-react";

import { ApiError } from "../../api/client";
import { errorMessage, locPath } from "../../api/errors";
import {
    useModels,
    usePreset,
    usePresetMutations,
    usePrompts,
} from "../../api/hooks";
import type {
    NamedResource,
    Preset,
    PresetDetail,
    SubagentPersona,
    SubagentToolConfig,
    ToolChoice,
} from "../../api/types";
import { MAX_STEPS_CAP } from "../../api/types";
import { Checkbox } from "../../components/Checkbox";
import { RadioList } from "../../components/Choices";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { ErrorPanel, QueryError } from "../../components/ErrorPanel";
import { Field, Panel } from "../../components/Panel";
import { Loading, PageHeader } from "../../components/States";
import {
    blankPreset,
    checkPresetShape,
    configuredSubagentPersonas,
    labelForLoc,
    MAX_SUBAGENT_PERSONAS,
    ragSelection,
    setRagSelection,
    type RagSelection,
    type ShapeProblem,
} from "./presetShape";

const TOOL_CHOICES: { value: ToolChoice; label: string }[] = [
    { value: "auto", label: "auto — 由模型決定" },
    { value: "required", label: "required — 必須先使用工具" },
    { value: "none", label: "none — 不使用工具" },
];

const RAG_CHOICES: { value: RagSelection; label: ReactNode }[] = [
    { value: "disabled", label: "停用 — 不提供教材搜尋" },
    { value: "enabled", label: "啟用 — 由老師決定何時搜尋" },
    { value: "forced", label: "強制 — 每次回答前先搜尋" },
];

export function PresetEditorScreen() {
    const { name } = useParams<{ name: string }>();
    const isNew = name === undefined;
    const navigate = useNavigate();
    const loaded = usePreset(name);
    const models = useModels();
    const prompts = usePrompts();
    const { save, remove } = usePresetMutations();
    const [localProblems, setLocalProblems] = useState<ShapeProblem[] | null>(
        null
    );
    const [confirmDelete, setConfirmDelete] = useState(false);
    const detail: PresetDetail | undefined = loaded.data;
    const [draft, setDraft] = useState<Preset | null>(null);
    const fresh = useMemo(() => blankPreset(), []);
    const base = isNew ? fresh : (detail?.definition ?? null);
    const preset = draft ?? base;

    const editPreset = (fn: (previous: Preset) => Preset) =>
        setDraft((previous) => {
            const from = previous ?? base;
            return from ? fn(from) : previous;
        });

    if (!isNew && loaded.isError) {
        return (
            <>
                <PageHeader title={name ?? "預設值"} back={<BackLink />} mono />
                <div style={{ marginTop: 20 }}>
                    <QueryError
                        what={`無法開啟預設值「${name}」`}
                        error={loaded.error}
                    />
                </div>
            </>
        );
    }
    if (!preset) {
        return (
            <>
                <PageHeader title={name ?? "預設值"} back={<BackLink />} mono />
                <Loading what="預設值" />
            </>
        );
    }

    const update = (patch: Partial<Preset>) =>
        editPreset((previous) => ({ ...previous, ...patch }));
    const updateSubagents = (patch: Partial<SubagentToolConfig>) =>
        editPreset((previous) => ({
            ...previous,
            tools: {
                ...previous.tools,
                subagents: { ...previous.tools.subagents, ...patch },
            },
        }));
    const onSave = () => {
        const problems = checkPresetShape(preset);
        setLocalProblems(problems);
        if (problems.length > 0) return;
        save.mutate(
            { name: preset.name, preset },
            {
                onSuccess: (saved) => {
                    setLocalProblems(null);
                    if (isNew || saved.name !== name) {
                        void navigate(
                            `/presets/${encodeURIComponent(saved.name)}`,
                            { replace: true }
                        );
                    }
                },
            }
        );
    };

    const availableModels = models.data?.models ?? [];
    const allowed = models.data?.allowed_models ?? [];
    const promptOptions = prompts.data ?? [];
    const renamed = !isNew && detail && preset.name !== detail.name;
    const deletable = detail
        ? !detail.builtin || detail.shadowed_builtin
        : false;
    const ragMode = ragSelection(preset);
    const subagents = preset.tools.subagents;
    const personas = configuredSubagentPersonas(subagents);
    const setPersonas = (next: SubagentPersona[]) =>
        updateSubagents({ personas: next, prompt_name: null });
    const updatePersona = (index: number, patch: Partial<SubagentPersona>) =>
        setPersonas(
            personas.map((persona, at) =>
                at === index ? { ...persona, ...patch } : persona
            )
        );

    return (
        <>
            <PageHeader
                back={<BackLink />}
                title={preset.name || (isNew ? "新增預設值" : (name ?? ""))}
                mono
                lede={<StoredIn isNew={isNew} detail={detail} />}
                actions={
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={onSave}
                        disabled={save.isPending}
                    >
                        {save.isPending ? "儲存中…" : "儲存並重新載入登錄表"}
                    </button>
                }
            />

            {localProblems && localProblems.length > 0 && (
                <div style={{ marginTop: 20 }}>
                    <ErrorPanel
                        title={`文件格式不正確 — ${localProblems.length} 個問題`}
                    >
                        <div className="alarm-list">
                            {localProblems.map((problem) => (
                                <div
                                    className="mono"
                                    style={{ fontSize: 12.5 }}
                                    key={problem.path + problem.message}
                                >
                                    {problem.path} — {problem.message}
                                </div>
                            ))}
                        </div>
                    </ErrorPanel>
                </div>
            )}
            {save.error && (
                <div style={{ marginTop: 14 }}>
                    <SaveError error={save.error} />
                </div>
            )}
            {remove.error && (
                <div style={{ marginTop: 14 }}>
                    <QueryError what="無法刪除此預設值" error={remove.error} />
                </div>
            )}

            <div className="split split-narrow" style={{ marginTop: 14 }}>
                <div className="col">
                    <Panel title="基本資料">
                        <div className="grid-2">
                            <Field
                                label="預設值名稱"
                                hint={
                                    renamed
                                        ? `以新的名稱另存新檔；'${detail?.name}' 將不被更動`
                                        : '小寫字母、數字、"_"與"-"'
                                }
                            >
                                {(id) => (
                                    <input
                                        id={id}
                                        className="input mono"
                                        value={preset.name}
                                        onChange={(event) =>
                                            update({ name: event.target.value })
                                        }
                                    />
                                )}
                            </Field>
                            <Field
                                label="模型"
                                hint="未設定時使用服務本身的預設值。"
                            >
                                {(id) => (
                                    <select
                                        id={id}
                                        className="input mono"
                                        value={preset.model ?? ""}
                                        onChange={(event) =>
                                            update({
                                                model:
                                                    event.target.value || null,
                                            })
                                        }
                                    >
                                        <option value="">
                                            — 伺服器預設模型 —
                                        </option>
                                        {modelOptions(
                                            availableModels,
                                            preset.model
                                        ).map((option) => (
                                            <option key={option} value={option}>
                                                {option}
                                            </option>
                                        ))}
                                    </select>
                                )}
                            </Field>
                        </div>
                        <Field label="此預設值的用途" style={{ marginTop: 14 }}>
                            {(id) => (
                                <input
                                    id={id}
                                    className="input"
                                    value={preset.description}
                                    onChange={(event) =>
                                        update({
                                            description: event.target.value,
                                        })
                                    }
                                />
                            )}
                        </Field>
                    </Panel>

                    <Panel title="主要代理人">
                        <div className="cast-card">
                            <div className="row" style={{ gap: 10 }}>
                                <span className="tag tag-neutral">
                                    固定角色
                                </span>
                                <strong>老師</strong>
                                <span className="mono note">teacher</span>
                            </div>
                            <Field
                                label="老師提示詞（Langfuse）"
                                hint={
                                    ragMode === "forced"
                                        ? "強制 RAG 會提供系統提示詞，因此此欄停用。"
                                        : "選填；不選時不注入額外角色提示。"
                                }
                                style={{ marginTop: 14 }}
                            >
                                {(id) => (
                                    <PromptSelect
                                        id={id}
                                        value={preset.teacher_prompt_name}
                                        options={promptOptions}
                                        disabled={ragMode === "forced"}
                                        onChange={(teacher_prompt_name) =>
                                            update({ teacher_prompt_name })
                                        }
                                    />
                                )}
                            </Field>
                        </div>
                    </Panel>

                    <ToolPanel
                        title="RAG 教材搜尋"
                        code="rag_search"
                        enabled={preset.tools.rag.enable_tool}
                        onEnabledChange={(enable_tool) =>
                            setDraft(
                                setRagSelection(
                                    preset,
                                    enable_tool ? "enabled" : "disabled"
                                )
                            )
                        }
                    >
                        <RadioList
                            name="rag-mode"
                            options={RAG_CHOICES}
                            value={ragMode}
                            onChange={(selection) =>
                                setDraft(setRagSelection(preset, selection))
                            }
                        />
                    </ToolPanel>

                    <ToolPanel
                        title="子代理人"
                        code="summon_subagent"
                        enabled={subagents.enable_tool}
                        onEnabledChange={(enable_tool) =>
                            updateSubagents({
                                enable_tool,
                                character_forcing: enable_tool
                                    ? subagents.character_forcing
                                    : false,
                            })
                        }
                    >
                        <ToggleRow
                            label="角色強制"
                            description="讓老師從設定好的角色中選擇一位子代理人，並套用該角色的提示詞。關閉時，子代理人會以一般代理人的方式直接處理任務。"
                            checked={subagents.character_forcing}
                            onChange={(character_forcing) => {
                                if (
                                    character_forcing &&
                                    personas.length === 0
                                ) {
                                    updateSubagents({
                                        character_forcing,
                                        personas: [blankPersona([])],
                                    });
                                    return;
                                }
                                updateSubagents({ character_forcing });
                            }}
                        />
                        {subagents.character_forcing && (
                            <div style={{ marginTop: 14 }}>
                                <div
                                    style={{
                                        display: "flex",
                                        flexDirection: "column",
                                        gap: 10,
                                    }}
                                >
                                    {personas.map((persona, index) => (
                                        <PersonaCard
                                            key={index}
                                            persona={persona}
                                            index={index}
                                            prompts={promptOptions}
                                            onChange={(patch) =>
                                                updatePersona(index, patch)
                                            }
                                            onRemove={
                                                personas.length > 1
                                                    ? () =>
                                                          setPersonas(
                                                              personas.filter(
                                                                  (_, at) =>
                                                                      at !==
                                                                      index
                                                              )
                                                          )
                                                    : undefined
                                            }
                                        />
                                    ))}
                                </div>
                                {personas.length < MAX_SUBAGENT_PERSONAS && (
                                    <button
                                        type="button"
                                        className="btn btn-ghost"
                                        style={{ marginTop: 10 }}
                                        onClick={() =>
                                            setPersonas([
                                                ...personas,
                                                blankPersona(personas),
                                            ])
                                        }
                                    >
                                        <Plus size={15} aria-hidden />
                                        新增角色
                                    </button>
                                )}
                            </div>
                        )}
                        {!subagents.character_forcing && (
                            <Field
                                label="子代理人的最大步數"
                                hint={`最多 ${MAX_STEPS_CAP} 步。`}
                                style={{ marginTop: 14 }}
                            >
                                {(id) => (
                                    <input
                                        id={id}
                                        className="input mono"
                                        inputMode="numeric"
                                        value={subagents.max_steps}
                                        onChange={(event) =>
                                            updateSubagents({
                                                max_steps: toInt(
                                                    event.target.value,
                                                    subagents.max_steps
                                                ),
                                            })
                                        }
                                    />
                                )}
                            </Field>
                        )}
                    </ToolPanel>

                    {prompts.isError && (
                        <p
                            className="note"
                            style={{ color: "var(--color-alarm-ink)" }}
                        >
                            無法載入 Langfuse 提示詞清單 —{" "}
                            {errorMessage(prompts.error)}
                        </p>
                    )}

                    <Panel title="限制">
                        <div className="grid-2">
                            <Field
                                label="每次回覆的最大步數"
                                hint={`在回答前最多能使用 ${MAX_STEPS_CAP} 次工具。`}
                            >
                                {(id) => (
                                    <input
                                        id={id}
                                        className="input mono"
                                        inputMode="numeric"
                                        value={preset.max_steps}
                                        onChange={(event) =>
                                            update({
                                                max_steps: toInt(
                                                    event.target.value,
                                                    preset.max_steps
                                                ),
                                            })
                                        }
                                    />
                                )}
                            </Field>
                            <Field label="工具選擇">
                                {(id) => (
                                    <select
                                        id={id}
                                        className="input"
                                        value={preset.tool_choice}
                                        onChange={(event) =>
                                            update({
                                                tool_choice: event.target
                                                    .value as ToolChoice,
                                            })
                                        }
                                    >
                                        {TOOL_CHOICES.map((choice) => (
                                            <option
                                                key={choice.value}
                                                value={choice.value}
                                            >
                                                {choice.label}
                                            </option>
                                        ))}
                                    </select>
                                )}
                            </Field>
                        </div>
                    </Panel>

                    <div className="row" style={{ gap: 16, flexWrap: "wrap" }}>
                        <button
                            type="button"
                            className="btn btn-primary"
                            style={{ fontSize: 14.5, padding: "11px 22px" }}
                            onClick={onSave}
                            disabled={save.isPending}
                        >
                            {save.isPending
                                ? "儲存中…"
                                : "儲存並重新載入登錄表"}
                        </button>
                        {deletable && (
                            <button
                                type="button"
                                className="btn btn-ghost btn-danger"
                                onClick={() => setConfirmDelete(true)}
                                disabled={remove.isPending}
                            >
                                <Trash2
                                    size={15}
                                    strokeWidth={2.75}
                                    aria-hidden
                                />
                                刪除此預設值
                            </button>
                        )}
                    </div>
                </div>

                <aside className="sticky-side">
                    <Panel title="允許清單中的模型">
                        {models.isError ? (
                            <p className="note">
                                無法使用 — {errorMessage(models.error)}
                            </p>
                        ) : !models.data ? (
                            <Loading what="模型" />
                        ) : allowed.length === 0 ? (
                            <p className="note">
                                尚未設定允許清單，因此可使用上游伺服器提供的任何模型。
                            </p>
                        ) : (
                            allowed.map((model) => (
                                <div
                                    className="mono"
                                    style={{ fontSize: 12.5, padding: "3px 0" }}
                                    key={model}
                                >
                                    {model}
                                </div>
                            ))
                        )}
                    </Panel>
                    <div
                        style={{
                            background: "var(--color-accent-2-100)",
                            border: "1px solid var(--color-accent-2-300)",
                            borderRadius: 12,
                            padding: "14px 16px",
                            fontSize: 12.5,
                            lineHeight: 1.55,
                            color: "var(--color-accent-2-800)",
                        }}
                    >
                        已有寫好的文件嗎？在預設值清單中選擇
                        <strong>匯入預設值</strong>，即可匯入一份 JSON
                        文件或整個陣列。
                    </div>
                </aside>
            </div>

            {confirmDelete && detail && (
                <ConfirmDialog
                    title={`要刪除「${detail.name}」嗎？`}
                    danger
                    body={
                        detail.shadowed_builtin
                            ? "Langfuse 項目將被移除，內建版本將恢復使用。"
                            : "Langfuse 項目將被移除，且此預設值將停止提供服務。"
                    }
                    confirmLabel="刪除"
                    busy={remove.isPending}
                    onCancel={() => setConfirmDelete(false)}
                    onConfirm={() => {
                        setConfirmDelete(false);
                        remove.mutate(detail.name, {
                            onSuccess: () => void navigate("/presets"),
                        });
                    }}
                />
            )}
        </>
    );
}

function blankPersona(existing: SubagentPersona[]): SubagentPersona {
    const ids = new Set(existing.map((persona) => persona.id));
    if (!ids.has("student")) {
        return {
            id: "student",
            display_name: "學生",
            prompt_name: "",
            max_steps: 3,
        };
    }
    if (!ids.has("ta")) {
        return {
            id: "ta",
            display_name: "TA",
            prompt_name: "",
            max_steps: 3,
        };
    }
    let suffix = existing.length + 1;
    while (ids.has(`persona-${suffix}`)) suffix += 1;
    return {
        id: `persona-${suffix}`,
        display_name: "新角色",
        prompt_name: "",
        max_steps: 3,
    };
}

function PersonaCard({
    persona,
    index,
    prompts,
    onChange,
    onRemove,
}: {
    persona: SubagentPersona;
    index: number;
    prompts: NamedResource[];
    onChange: (patch: Partial<SubagentPersona>) => void;
    onRemove?: () => void;
}) {
    return (
        <div className="cast-card">
            <div className="row" style={{ gap: 10 }}>
                <span className="tag tag-accent">角色 {index + 1}</span>
                <strong>{persona.display_name || "未命名角色"}</strong>
                <span className="mono note">{persona.id || "—"}</span>
                {onRemove && (
                    <button
                        type="button"
                        className="btn btn-ghost btn-danger"
                        style={{ marginLeft: "auto" }}
                        onClick={onRemove}
                    >
                        <Trash2 size={14} aria-hidden />
                        移除
                    </button>
                )}
            </div>
            <div className="grid-2" style={{ marginTop: 14 }}>
                <Field
                    label="角色 ID"
                    hint="穩定的英文識別碼；老師召喚時會優先使用顯示名稱。"
                >
                    {(id) => (
                        <input
                            id={id}
                            className="input mono"
                            value={persona.id}
                            onChange={(event) =>
                                onChange({ id: event.target.value })
                            }
                        />
                    )}
                </Field>
                <Field label="顯示名稱">
                    {(id) => (
                        <input
                            id={id}
                            className="input"
                            value={persona.display_name}
                            onChange={(event) =>
                                onChange({ display_name: event.target.value })
                            }
                        />
                    )}
                </Field>
            </div>
            <Field
                label="角色提示詞（Langfuse）"
                hint="每個角色都必須選擇自己的提示詞。"
                style={{ marginTop: 14 }}
            >
                {(id) => (
                    <PromptSelect
                        id={id}
                        value={persona.prompt_name}
                        options={prompts}
                        required
                        onChange={(prompt_name) =>
                            onChange({ prompt_name: prompt_name ?? "" })
                        }
                    />
                )}
            </Field>
            <Field
                label="角色的最大步數"
                hint={`最多 ${MAX_STEPS_CAP} 步。`}
                style={{ marginTop: 14 }}
            >
                {(id) => (
                    <input
                        id={id}
                        className="input mono"
                        inputMode="numeric"
                        value={persona.max_steps}
                        onChange={(event) =>
                            onChange({
                                max_steps: toInt(
                                    event.target.value,
                                    persona.max_steps
                                ),
                            })
                        }
                    />
                )}
            </Field>
        </div>
    );
}

function ToolPanel({
    title,
    code,
    enabled,
    onEnabledChange,
    children,
}: {
    title: string;
    code: string;
    enabled: boolean;
    onEnabledChange: (enabled: boolean) => void;
    children: ReactNode;
}) {
    return (
        <Panel
            title={title}
            actions={
                <div className="row" style={{ gap: 10 }}>
                    <span className="mono note">{code}</span>
                    <Checkbox
                        checked={enabled}
                        onChange={onEnabledChange}
                        style={{ gap: 6 }}
                    >
                        <span className="mono" style={{ fontSize: 12 }}>
                            enable_tool
                        </span>
                    </Checkbox>
                </div>
            }
        >
            {enabled && (
                <div
                    style={{
                        borderTop: "1px solid var(--color-neutral-200)",
                        marginTop: 10,
                        paddingTop: 14,
                    }}
                >
                    {children}
                </div>
            )}
        </Panel>
    );
}

function ToggleRow({
    label,
    description,
    checked,
    onChange,
}: {
    label: string;
    description: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
}) {
    return (
        <Checkbox
            checked={checked}
            onChange={onChange}
            style={{ alignItems: "flex-start", gap: 10 }}
        >
            <span>
                <strong>{label}</strong>
                <span
                    className="note"
                    style={{ display: "block", marginTop: 3 }}
                >
                    {description}
                </span>
            </span>
        </Checkbox>
    );
}

function PromptSelect({
    id,
    value,
    options,
    required = false,
    disabled = false,
    onChange,
}: {
    id: string;
    value: string | null;
    options: NamedResource[];
    required?: boolean;
    disabled?: boolean;
    onChange: (value: string | null) => void;
}) {
    const names = options.map((option) => option.name);
    const choices =
        value && !names.includes(value)
            ? [{ name: value, label: value }, ...options]
            : options;
    return (
        <select
            id={id}
            className="input mono"
            value={value ?? ""}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value || null)}
        >
            <option value="">
                {required ? "— 選擇提示詞 —" : "— 不使用提示詞 —"}
            </option>
            {choices.map((option) => (
                <option key={option.name} value={option.name}>
                    {option.label}
                </option>
            ))}
        </select>
    );
}

function BackLink() {
    return (
        <Link to="/presets" style={{ fontSize: 12.5 }}>
            ← 所有預設值
        </Link>
    );
}

function StoredIn({
    isNew,
    detail,
}: {
    isNew: boolean;
    detail: PresetDetail | undefined;
}) {
    if (isNew)
        return (
            <span style={{ fontSize: 13 }}>
                尚未儲存 · 儲存時會寫入 Langfuse 資料集{" "}
                <span className="mono">config/presets</span>
            </span>
        );
    if (!detail) return null;
    if (detail.builtin && !detail.shadowed_builtin)
        return (
            <span style={{ fontSize: 13 }}>
                內建於服務中 · 儲存後會建立 Langfuse 覆寫
            </span>
        );
    if (detail.builtin)
        return (
            <span style={{ fontSize: 13 }}>
                儲存於 Langfuse · 覆寫內建預設值
            </span>
        );
    return <span style={{ fontSize: 13 }}>儲存於 Langfuse</span>;
}

function SaveError({ error }: { error: unknown }) {
    const problems = error instanceof ApiError ? error.problems : [];
    const status = error instanceof ApiError ? error.status : null;
    if (problems.length === 0)
        return (
            <ErrorPanel
                title={status ? `服務拒絕此文件 — ${status}` : "無法連線至服務"}
                detail={errorMessage(error)}
                copyText={errorMessage(error)}
            />
        );
    return (
        <ErrorPanel
            title={`服務拒絕此文件 — ${problems.length} 個問題`}
            copyText={problems
                .map((problem) => `${locPath(problem.loc)}: ${problem.msg}`)
                .join("\n")}
        >
            <div className="alarm-list">
                {problems.map((problem, index) => (
                    <div
                        className="mono"
                        style={{ fontSize: 12.5 }}
                        key={index}
                    >
                        <strong>{labelForLoc(problem.loc)}</strong> —{" "}
                        {problem.msg}
                    </div>
                ))}
            </div>
        </ErrorPanel>
    );
}

function modelOptions(available: string[], current: string | null): string[] {
    if (current && !available.includes(current)) return [current, ...available];
    return available;
}

function toInt(text: string, fallback: number): number {
    const value = Number(text);
    return /^\d+$/.test(text.trim()) && Number.isFinite(value)
        ? value
        : fallback;
}
