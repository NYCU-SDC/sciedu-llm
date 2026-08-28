import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plus, Trash2 } from "lucide-react";

import { ApiError } from "../../api/client";
import { errorMessage, locPath } from "../../api/errors";
import {
    useModels,
    usePreset,
    usePresetMutations,
    useTools,
} from "../../api/hooks";
import type {
    Preset,
    PresetCharacter,
    PresetDetail,
    ToolChoice,
} from "../../api/types";
import { MAX_STEPS_CAP } from "../../api/types";
import { RadioList } from "../../components/Choices";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { ErrorPanel, QueryError } from "../../components/ErrorPanel";
import { Field, Panel } from "../../components/Panel";
import { Loading, PageHeader } from "../../components/States";
import {
    RAG_SEARCH,
    SUMMON_SUBAGENT,
    blankCharacter,
    blankPreset,
    courseMaterialMode,
    forcedRagObjections,
    formatTools,
    labelForLoc,
    parseTools,
    setCourseMaterialMode,
    type CourseMaterialMode,
    type ShapeProblem,
} from "./presetShape";

const TOOL_CHOICES: { value: ToolChoice; label: string }[] = [
    { value: "auto", label: "auto — 由模型決定" },
    { value: "required", label: "required — 必須先使用工具" },
    { value: "none", label: "none — 不使用工具" },
];

const COURSE_MATERIAL: { value: CourseMaterialMode; label: React.ReactNode }[] =
    [
        { value: "never", label: "不搜尋 — 模型依自身知識回答" },
        { value: "always", label: "回答前一律搜尋" },
        {
            value: "decide",
            label: (
                <>
                    由模型決定，使用 <span className="mono">{RAG_SEARCH}</span>{" "}
                    工具
                </>
            ),
        },
    ];

export function PresetEditorScreen() {
    const { name } = useParams<{ name: string }>();
    const isNew = name === undefined;
    const navigate = useNavigate();

    const loaded = usePreset(name);
    const models = useModels();
    const tools = useTools();
    const { save, remove } = usePresetMutations();

    const [localProblems, setLocalProblems] = useState<ShapeProblem[] | null>(
        null
    );
    const [confirmDelete, setConfirmDelete] = useState(false);

    const detail: PresetDetail | undefined = loaded.data;

    // The buffer only exists once the user edits something: until then the screen
    // renders the served document directly. Deriving it this way rather than
    // copying it into state on arrival means a background refetch can never
    // clobber an edit in progress, and needs no effect.
    const [draft, setDraft] = useState<Preset | null>(null);
    const fresh = useMemo(() => blankPreset(), []);
    const base = isNew ? fresh : (detail?.definition ?? null);
    const preset = draft ?? base;

    const setPreset = (next: Preset) => setDraft(next);
    const editPreset = (fn: (previous: Preset) => Preset) =>
        setDraft((previous) => {
            const from = previous ?? base;
            return from ? fn(from) : previous;
        });

    const deletable = detail
        ? !detail.builtin || detail.shadowed_builtin
        : false;

    const onSave = () => {
        const document = preset;
        if (!document) return;
        if (!document.name) {
            setLocalProblems([
                { path: "name", message: "預設值必須有名稱才能儲存。" },
            ]);
            return;
        }
        save.mutate(
            { name: document.name, preset: document },
            {
                onSuccess: (saved) => {
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

    const updateCharacter = (index: number, patch: Partial<PresetCharacter>) =>
        editPreset((previous) => {
            const characters = previous.characters.map((character, at) =>
                at === index ? { ...character, ...patch } : character
            );
            // Keep `orchestrator` pointing at the same character when its id changes.
            const wasOrchestrator =
                previous.characters[index]?.id === previous.orchestrator;
            const orchestrator =
                wasOrchestrator && patch.id !== undefined
                    ? patch.id
                    : previous.orchestrator;
            return { ...previous, characters, orchestrator };
        });

    const addCharacter = () =>
        editPreset((previous) => {
            if (previous.characters.length >= 2) return previous;
            const extra = blankCharacter();
            const characters = [...previous.characters, extra];
            // A second character is only reachable if the orchestrator may summon it.
            const withSummon = characters.map((character) =>
                character.id === previous.orchestrator &&
                !character.tools.includes(SUMMON_SUBAGENT)
                    ? {
                          ...character,
                          tools: [...character.tools, SUMMON_SUBAGENT],
                      }
                    : character
            );
            return { ...previous, characters: withSummon };
        });

    const removeCharacter = (index: number) =>
        editPreset((previous) => {
            const characters = previous.characters.filter(
                (_, at) => at !== index
            );
            // With nobody left to summon, the tool is invalid — the backend rejects it.
            const cleaned = characters.map((character) => ({
                ...character,
                tools: character.tools.filter(
                    (tool) => tool !== SUMMON_SUBAGENT
                ),
            }));
            return { ...previous, characters: cleaned };
        });

    const objections = forcedRagObjections(preset);
    const allowed = models.data?.allowed_models ?? [];
    const renamed = !isNew && detail && preset.name !== detail.name;

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

            {localProblems !== null && localProblems.length > 0 && (
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
                        <p className="alarm-body">
                            已在瀏覽器中檢查；儲存時服務會執行完整驗證。
                        </p>
                    </ErrorPanel>
                </div>
            )}

            {save.error && (
                <div style={{ marginTop: 14 }}>
                    <SaveError error={save.error} preset={preset} />
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
                                                    event.target.value === ""
                                                        ? null
                                                        : event.target.value,
                                            })
                                        }
                                    >
                                        <option value="">
                                            — 伺服器預設模型 —
                                        </option>
                                        {modelOptions(
                                            allowed,
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

                    <Panel title="角色群">
                        <p className="note" style={{ marginBottom: 14 }}>
                            預設由一位助理回答。需要不同觀點時，可新增第二位供第一位角色召喚。
                        </p>
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: 10,
                            }}
                        >
                            {preset.characters.map((character, index) => (
                                <CharacterCard
                                    key={index}
                                    character={character}
                                    isOrchestrator={
                                        character.id === preset.orchestrator
                                    }
                                    onChange={(patch) =>
                                        updateCharacter(index, patch)
                                    }
                                    onRemove={
                                        preset.characters.length > 1 &&
                                        character.id !== preset.orchestrator
                                            ? () => removeCharacter(index)
                                            : undefined
                                    }
                                />
                            ))}
                            {preset.characters.length < 2 && (
                                <button
                                    type="button"
                                    className="btn btn-ghost"
                                    style={{ alignSelf: "flex-start" }}
                                    onClick={addCharacter}
                                >
                                    <Plus
                                        size={15}
                                        strokeWidth={2.75}
                                        aria-hidden
                                    />
                                    新增可召喚角色
                                </button>
                            )}
                        </div>
                    </Panel>

                    <Panel title="課程教材">
                        <RadioList
                            name="course-material"
                            options={COURSE_MATERIAL}
                            value={courseMaterialMode(preset)}
                            onChange={(mode) =>
                                setPreset(setCourseMaterialMode(preset, mode))
                            }
                        />
                        {objections.length > 0 && (
                            <div style={{ marginTop: 12 }}>
                                <ErrorPanel title="服務會拒絕此組合">
                                    <div className="alarm-list">
                                        {objections.map((objection) => (
                                            <div
                                                style={{ fontSize: 12.5 }}
                                                key={objection}
                                            >
                                                {objection}
                                            </div>
                                        ))}
                                    </div>
                                </ErrorPanel>
                            </div>
                        )}
                    </Panel>

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

                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 16,
                            flexWrap: "wrap",
                            padding: 2,
                        }}
                    >
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
                            <>
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
                                {detail?.shadowed_builtin && (
                                    <p
                                        className="note"
                                        style={{ maxWidth: "38ch" }}
                                    >
                                        刪除後，內建的{" "}
                                        <span className="mono">
                                            {detail.name}
                                        </span>{" "}
                                        將恢復使用。
                                    </p>
                                )}
                            </>
                        )}
                    </div>
                </div>

                <aside className="sticky-side">
                    <Panel title="可指定的工具">
                        {!tools.data ? (
                            <Loading what="工具" />
                        ) : (
                            <div className="kv-list">
                                {(tools.data ?? []).map((tool) => (
                                    <div key={tool.name}>
                                        <span
                                            className="mono"
                                            style={{
                                                fontSize: 12.5,
                                                flex: "none",
                                                fontWeight: 600,
                                            }}
                                        >
                                            {tool.name}
                                        </span>
                                        <span
                                            style={{
                                                fontSize: 11.5,
                                                color: "var(--color-neutral-600)",
                                                textAlign: "right",
                                                flex: 1,
                                            }}
                                        >
                                            {tool.description}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}
                        <p className="note" style={{ marginTop: 10 }}>
                            <span className="mono">{SUMMON_SUBAGENT}</span>{" "}
                            只能由協調角色使用，且必須存在可召喚的第二位角色。
                        </p>
                    </Panel>

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
                        文件或整個陣列；兩種方式都會執行相同的驗證。
                    </div>
                </aside>
            </div>

            {confirmDelete && detail && (
                <ConfirmDialog
                    title={`要刪除「${detail.name}」嗎？`}
                    danger
                    body={
                        detail.shadowed_builtin ? (
                            <>
                                Langfuse 項目將被移除，內建的{" "}
                                <span className="mono">{detail.name}</span> goes
                                將恢復使用。
                            </>
                        ) : (
                            <>
                                Langfuse
                                項目將被移除，且此預設值將停止提供服務。任何以名稱呼叫它的地方都會收到錯誤。
                            </>
                        )
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
    if (isNew) {
        return (
            <span style={{ fontSize: 13 }}>
                尚未儲存 · 儲存時會寫入 Langfuse 資料集{" "}
                <span className="mono">config/presets</span>
            </span>
        );
    }
    if (!detail) return null;
    if (detail.builtin && !detail.shadowed_builtin) {
        return (
            <span style={{ fontSize: 13 }}>
                內建於服務中 · 儲存後會建立覆寫它的 Langfuse
                副本，內建版本仍會保留
            </span>
        );
    }
    if (detail.builtin) {
        return (
            <span style={{ fontSize: 13 }}>
                儲存於 Langfuse · 覆寫內建預設值{" "}
                <span className="mono">{detail.name}</span>
            </span>
        );
    }
    return <span style={{ fontSize: 13 }}>儲存於 Langfuse</span>;
}

function CharacterCard({
    character,
    isOrchestrator,
    onChange,
    onRemove,
}: {
    character: PresetCharacter;
    isOrchestrator: boolean;
    onChange: (patch: Partial<PresetCharacter>) => void;
    onRemove?: () => void;
}) {
    return (
        <div className="cast-card">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                    className={
                        isOrchestrator ? "tag tag-neutral" : "tag tag-accent"
                    }
                >
                    {isOrchestrator ? "協調角色" : "可召喚"}
                </span>
                <span
                    className="mono"
                    style={{ fontSize: 13, fontWeight: 600 }}
                >
                    {character.id || "（沒有 ID）"}
                </span>
                {onRemove && (
                    <button
                        type="button"
                        className="btn btn-ghost btn-danger"
                        style={{ marginLeft: "auto" }}
                        onClick={onRemove}
                    >
                        移除
                    </button>
                )}
            </div>
            <div className="grid-2" style={{ marginTop: 12 }}>
                <Field label="ID（文件中使用）">
                    {(id) => (
                        <input
                            id={id}
                            className="input mono"
                            value={character.id}
                            onChange={(event) =>
                                onChange({ id: event.target.value })
                            }
                        />
                    )}
                </Field>
                <Field label="顯示名稱（學生可見）">
                    {(id) => (
                        <input
                            id={id}
                            className="input"
                            value={character.display_name}
                            onChange={(event) =>
                                onChange({ display_name: event.target.value })
                            }
                        />
                    )}
                </Field>
            </div>
            <div className="grid-2" style={{ marginTop: 12 }}>
                <Field label="角色">
                    {(id) => (
                        <input
                            id={id}
                            className="input mono"
                            value={character.role}
                            onChange={(event) =>
                                onChange({ role: event.target.value })
                            }
                        />
                    )}
                </Field>
                <Field
                    label="提示詞（Langfuse）"
                    hint={
                        isOrchestrator
                            ? undefined
                            : "被召喚的角色必須有自己的提示詞。"
                    }
                >
                    {(id) => (
                        <input
                            id={id}
                            className="input mono"
                            placeholder={isOrchestrator ? "選填" : "必填"}
                            value={character.prompt_name ?? ""}
                            onChange={(event) =>
                                onChange({
                                    prompt_name:
                                        event.target.value === ""
                                            ? null
                                            : event.target.value,
                                })
                            }
                        />
                    )}
                </Field>
            </div>
            <div className="grid-2" style={{ marginTop: 12 }}>
                <Field
                    label="可呼叫的工具"
                    hint="以半形逗號分隔，名稱取自右側清單。"
                >
                    {(id) => (
                        <input
                            id={id}
                            className="input mono"
                            value={formatTools(character.tools)}
                            onChange={(event) =>
                                onChange({
                                    tools: parseTools(event.target.value),
                                })
                            }
                        />
                    )}
                </Field>
                {!isOrchestrator && (
                    <Field
                        label="被召喚時的步數"
                        hint={`最多 ${MAX_STEPS_CAP} 步。`}
                    >
                        {(id) => (
                            <input
                                id={id}
                                className="input mono"
                                inputMode="numeric"
                                value={character.max_steps}
                                onChange={(event) =>
                                    onChange({
                                        max_steps: toInt(
                                            event.target.value,
                                            character.max_steps
                                        ),
                                    })
                                }
                            />
                        )}
                    </Field>
                )}
            </div>
        </div>
    );
}

/** A 422 from `PUT /admin/presets/{name}` arrives either as a flat string (the
 * name-mismatch check) or as pydantic's list of problems. The list gets mapped
 * back onto the form's own wording where the `loc` allows it. */
function SaveError({ error, preset }: { error: unknown; preset: Preset }) {
    const problems = error instanceof ApiError ? error.problems : [];
    const status = error instanceof ApiError ? error.status : null;

    if (problems.length === 0) {
        return (
            <ErrorPanel
                title={
                    status === 502
                        ? "Langfuse 拒絕寫入"
                        : status
                          ? `服務拒絕此文件 — ${status}`
                          : "無法連線至服務"
                }
                detail={errorMessage(error)}
                copyText={errorMessage(error)}
            />
        );
    }

    return (
        <ErrorPanel
            title={`服務拒絕此文件 — ${problems.length} 個問題`}
            copyText={problems
                .map((p) => `${locPath(p.loc)}: ${p.msg}`)
                .join("\n")}
        >
            <div className="alarm-list">
                {problems.map((problem, index) => (
                    <div
                        className="mono"
                        style={{ fontSize: 12.5 }}
                        key={index}
                    >
                        <strong>{labelForLoc(problem.loc, preset)}</strong> —{" "}
                        {problem.msg}
                    </div>
                ))}
            </div>
        </ErrorPanel>
    );
}

function modelOptions(allowed: string[], current: string | null): string[] {
    if (current && !allowed.includes(current)) return [current, ...allowed];
    return allowed;
}

function toInt(text: string, fallback: number): number {
    const value = Number(text);
    return /^\d+$/.test(text.trim()) && Number.isFinite(value)
        ? value
        : fallback;
}
