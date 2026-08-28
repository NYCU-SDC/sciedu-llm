/* The playground: one conversation with `POST /agents`, rendered the way the
 * protocol means it to be read.
 *
 * Nothing here persists. The transcript lives in component state, the session id
 * is minted once per page load, and neither is written to storage of any kind —
 * a reload is a clean slate on purpose, because this screen exists to try things
 * out, not to keep them. The backend is stateless too: every turn re-sends the
 * whole conversation (see `historyMessages`). */

import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";
import {
    Brain,
    FlaskConical,
    RotateCcw,
    Search,
    Send,
    Square,
    Wrench,
} from "lucide-react";

import {
    streamAgents,
    type AgentEvent,
    type CastCharacter,
} from "../../api/agentsStream";
import { langfuseSessionUrl } from "../../api/client";
import { errorMessage } from "../../api/errors";
import { usePresets } from "../../api/hooks";
import { CopyButton } from "../../components/CopyButton";
import { ErrorPanel, QueryError } from "../../components/ErrorPanel";
import { PageHeader } from "../../components/States";
import {
    blockIsVisible,
    buildTurn,
    hasInternal,
    historyMessages,
    type SpeakerBlock,
    type StreamedPart,
    type Turn,
} from "./transcript";

/** Every run from this screen is traced under one user id, so a Langfuse filter
 * separates manual pokes at the service from anything real. */
const PLAYGROUND_USER = "playground";

/** The server's own fallback (`AGENTS_DEFAULT_PRESET`). Naming it in the picker
 * costs no extra request — choosing it simply sends no `preset` at all. */
const SERVER_DEFAULT_LABEL = "server default (default-agents)";

/** `crypto.randomUUID` only exists in a secure context, and this console is
 * often reached over plain http on a lab network — the same reason CopyButton
 * keeps a fallback. The id only has to be unique enough to group traces. */
function newSessionId(): string {
    if (
        typeof crypto !== "undefined" &&
        typeof crypto.randomUUID === "function"
    ) {
        return crypto.randomUUID();
    }
    const bytes = new Uint8Array(16);
    if (
        typeof crypto !== "undefined" &&
        typeof crypto.getRandomValues === "function"
    ) {
        crypto.getRandomValues(bytes);
    } else {
        for (let at = 0; at < bytes.length; at++)
            bytes[at] = Math.floor(Math.random() * 256);
    }
    const hex = [...bytes]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function PlaygroundScreen() {
    const presets = usePresets();

    const [sessionId, setSessionId] = useState(newSessionId);
    const [preset, setPreset] = useState("");
    const [turns, setTurns] = useState<Turn[]>([]);
    const [draft, setDraft] = useState("");
    const [runningId, setRunningId] = useState<string | null>(null);

    const abort = useRef<AbortController | null>(null);
    const nextId = useRef(0);
    const foot = useRef<HTMLDivElement | null>(null);

    const busy = runningId !== null;

    // Abort whatever is in flight when the screen goes away, so a half-read
    // stream does not go on calling setState.
    useEffect(() => () => abort.current?.abort(), []);

    const patch = useCallback((id: string, change: (turn: Turn) => Turn) => {
        setTurns((previous) =>
            previous.map((turn) => (turn.id === id ? change(turn) : turn))
        );
    }, []);

    const send = () => {
        const content = draft.trim();
        if (!content || busy) return;

        const userId = `t${nextId.current++}`;
        const assistantId = `t${nextId.current++}`;
        const chosen = preset === "" ? undefined : preset;
        // The whole conversation so far, plus this message: /agents keeps nothing.
        const messages = [
            ...historyMessages(turns),
            { role: "user" as const, content },
        ];

        setTurns((previous) => [
            ...previous,
            { kind: "user", id: userId, content },
            {
                kind: "assistant",
                id: assistantId,
                preset: chosen ?? null,
                events: [],
                aborted: false,
                failure: null,
            },
        ]);
        setDraft("");
        setRunningId(assistantId);

        const controller = new AbortController();
        abort.current = controller;

        void streamAgents(
            {
                messages,
                preset: chosen,
                session: sessionId,
                user: PLAYGROUND_USER,
            },
            {
                signal: controller.signal,
                onEvent: (event: AgentEvent) => {
                    patch(assistantId, (turn) =>
                        turn.kind === "assistant"
                            ? { ...turn, events: [...turn.events, event] }
                            : turn
                    );
                },
            }
        )
            .catch((error: unknown) => {
                if (controller.signal.aborted) {
                    patch(assistantId, (turn) =>
                        turn.kind === "assistant"
                            ? { ...turn, aborted: true }
                            : turn
                    );
                    return;
                }
                patch(assistantId, (turn) =>
                    turn.kind === "assistant"
                        ? {
                              ...turn,
                              failure: {
                                  title: "服務未執行此輪對話",
                                  detail: errorMessage(error),
                              },
                          }
                        : turn
                );
            })
            .finally(() => {
                if (abort.current === controller) abort.current = null;
                setRunningId((current) =>
                    current === assistantId ? null : current
                );
            });
    };

    const stop = () => abort.current?.abort();

    const newSession = () => {
        abort.current?.abort();
        setSessionId(newSessionId());
        setTurns([]);
        setDraft("");
    };

    // Follow the conversation as turns are added — but not on every token, which
    // would fight anyone scrolling back through a long run.
    useEffect(() => {
        foot.current?.scrollIntoView({ block: "nearest" });
    }, [turns.length]);

    const sessionUrl = langfuseSessionUrl(sessionId);

    return (
        <>
            <PageHeader
                kicker={
                    <span
                        style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 6,
                        }}
                    >
                        <FlaskConical
                            size={12}
                            strokeWidth={2.75}
                            aria-hidden
                        />
                        手動測試
                    </span>
                }
                title="測試區"
                lede="如同用戶端般呼叫 POST /agents，並查看帶型別的事件：誰正在說話、查了什麼，以及底層機制做了什麼。"
                actions={
                    <>
                        <PresetControl
                            value={preset}
                            names={
                                presets.data?.map((entry) => entry.name) ?? []
                            }
                            disabled={busy}
                            onChange={setPreset}
                        />
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={newSession}
                        >
                            <RotateCcw
                                size={14}
                                strokeWidth={2.75}
                                aria-hidden
                            />
                            新工作階段
                        </button>
                    </>
                }
            />

            <div className="pg-page">
                <div className="pg-identity">
                    <span className="note">工作階段</span>
                    <span className="mono pg-session">{sessionId}</span>
                    <CopyButton
                        text={sessionId}
                        label="複製 ID"
                        className="btn btn-ghost pg-mini"
                    />
                    {sessionUrl && (
                        <a
                            href={sessionUrl}
                            target="_blank"
                            rel="noreferrer"
                            style={{ fontSize: 12.5 }}
                        >
                            在 Langfuse 開啟此工作階段
                        </a>
                    )}
                </div>

                {presets.isError && (
                    <div style={{ marginTop: 14 }}>
                        <QueryError
                            what="無法列出預設值"
                            error={presets.error}
                            actions={
                                <button
                                    type="button"
                                    className="btn btn-secondary"
                                    onClick={() => void presets.refetch()}
                                >
                                    再試一次
                                </button>
                            }
                        />
                    </div>
                )}

                <div className="pg-transcript">
                    {turns.length === 0 && (
                        <p className="quiet">
                            尚未傳送任何訊息。請在上方選擇預設值並在下方提問；多角色預設值會在發言前介紹角色群。
                        </p>
                    )}
                    {turns.map((turn) =>
                        turn.kind === "user" ? (
                            <UserBubble key={turn.id} content={turn.content} />
                        ) : (
                            <AssistantTurnView
                                key={turn.id}
                                events={turn.events}
                                aborted={turn.aborted}
                                failure={turn.failure}
                            />
                        )
                    )}
                    <div ref={foot} />
                </div>
            </div>

            {/* Docked to the bottom of the viewport rather than to the end of the
          transcript: this screen is a conversation, and its input belongs where
          a conversation's input is, however long the transcript has grown. */}
            <div className="pg-dock">
                <Composer
                    value={draft}
                    busy={busy}
                    onChange={setDraft}
                    onSend={send}
                    onStop={stop}
                />
            </div>
        </>
    );
}

// ── the composer ──────────────────────────────────────────────────────────

function Composer({
    value,
    busy,
    onChange,
    onSend,
    onStop,
}: {
    value: string;
    busy: boolean;
    onChange: (value: string) => void;
    onSend: () => void;
    onStop: () => void;
}) {
    return (
        <div className="pg-composer">
            <textarea
                className="input pg-input"
                placeholder={
                    busy
                        ? "正在等待執行完成…"
                        : "請提問。Enter 傳送，Shift+Enter 換行。"
                }
                value={value}
                disabled={busy}
                rows={3}
                onChange={(event) => onChange(event.target.value)}
                onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        onSend();
                    }
                }}
            />
            <div className="pg-composer-actions">
                <span className="note">
                    {busy
                        ? "串流中 — 此輪結束前無法輸入。"
                        : "Enter 傳送 · Shift+Enter 換行"}
                </span>
                {busy ? (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onStop}
                    >
                        <Square size={14} strokeWidth={2.75} aria-hidden />
                        停止
                    </button>
                ) : (
                    <button
                        type="button"
                        className="btn btn-primary"
                        disabled={value.trim() === ""}
                        onClick={onSend}
                    >
                        <Send size={14} strokeWidth={2.75} aria-hidden />
                        傳送
                    </button>
                )}
            </div>
        </div>
    );
}

/** The preset picker as a header control, beside "new session": which behaviour
 * to run is the one setting this screen has, and it applies to every turn of the
 * session, so it belongs with the session itself rather than in a side panel. */
function PresetControl({
    value,
    names,
    disabled,
    onChange,
}: {
    value: string;
    names: string[];
    disabled: boolean;
    onChange: (value: string) => void;
}) {
    const hint =
        value === ""
            ? "不會傳送 preset，因此由服務的 AGENTS_DEFAULT_PRESET 決定。"
            : `每一輪都會以 preset=${value} 傳送。`;

    // With no listing to choose from — the service is down, or /admin/presets
    // failed — the name becomes a free-text field rather than an empty dropdown,
    // the same fallback the retrieval screen's model picker uses. An unknown name
    // is a 400 from /agents, which the transcript renders like any other refusal.
    return (
        <label className="pg-preset" title={hint}>
            <span className="note">預設值</span>
            {names.length === 0 ? (
                <input
                    className="input mono pg-preset-field"
                    placeholder="預設值名稱（留白＝伺服器預設值）"
                    value={value}
                    disabled={disabled}
                    onChange={(event) => onChange(event.target.value)}
                />
            ) : (
                <select
                    className="input mono pg-preset-field"
                    value={value}
                    disabled={disabled}
                    onChange={(event) => onChange(event.target.value)}
                >
                    <option value="">{SERVER_DEFAULT_LABEL}</option>
                    {names.map((name) => (
                        <option key={name} value={name}>
                            {name}
                        </option>
                    ))}
                </select>
            )}
        </label>
    );
}

// ── the transcript ────────────────────────────────────────────────────────

function UserBubble({ content }: { content: string }) {
    return (
        <div className="pg-turn pg-turn-user">
            <div className="pg-speaker-head">
                <span className="pg-avatar pg-avatar-user">Y</span>
                <span className="pg-name">您</span>
            </div>
            <p className="pg-text">{content}</p>
        </div>
    );
}

// The preset a turn ran under is no longer shown per turn — the cast row says
// who is speaking and the page header says which preset is selected — but
// `Turn.preset` still records it, so a turn from before the picker was changed
// stays attributable.
function AssistantTurnView({
    events,
    aborted,
    failure,
}: {
    events: AgentEvent[];
    aborted: boolean;
    failure: { title: string; detail: string } | null;
}) {
    const [showInternal, setShowInternal] = useState(false);
    const turn = useMemo(
        () => buildTurn(events, { aborted, failed: failure !== null }),
        [events, aborted, failure]
    );

    const cast = new Map(
        turn.cast.map((character) => [character.id, character])
    );
    // The cast is ordered by the server with the orchestrator first, which is the
    // only signal for who is running the show.
    const orchestrator = turn.cast[0]?.id ?? null;
    const internalPresent = turn.blocks.some(hasInternal);

    return (
        <div className="pg-turn">
            {turn.cast.length > 0 && (
                <div className="pg-cast">
                    <span className="sect" style={{ marginRight: "auto" }}>
                        啟用角色
                    </span>
                    <div className="pg-characters">
                        {turn.cast.map((character) => (
                            <span className="pg-cast-chip" key={character.id}>
                                <Avatar
                                    character={character}
                                    orchestrator={orchestrator}
                                />
                                <span className="pg-name">
                                    {character.displayName}
                                </span>
                                <span className="tag tag-neutral">
                                    {character.role}
                                </span>
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {turn.blocks.map((block) =>
                blockIsVisible(block, showInternal) ? (
                    <SpeakerBlockView
                        key={block.key}
                        block={block}
                        cast={cast}
                        orchestrator={orchestrator}
                        activeKey={turn.activeBlockKey}
                        showInternal={showInternal}
                    />
                ) : null
            )}

            {turn.error && (
                <div style={{ marginTop: 10 }}>
                    <ErrorPanel
                        title="執行失敗"
                        detail={turn.error.message}
                        copyText={`${turn.error.code} — ${turn.error.message}`}
                    >
                        <p className="alarm-body" style={{ marginTop: 6 }}>
                            錯誤代碼{" "}
                            <span className="mono">{turn.error.code}</span>
                        </p>
                    </ErrorPanel>
                </div>
            )}

            {failure && (
                <div style={{ marginTop: 10 }}>
                    <ErrorPanel
                        title={failure.title}
                        detail={failure.detail}
                        copyText={failure.detail}
                    />
                </div>
            )}

            <div className="pg-turn-foot">
                {turn.status === "streaming" && (
                    <span className="note">串流中…</span>
                )}
                {turn.status === "completed" && (
                    <span className="note">
                        已完成 ·{" "}
                        <span className="mono">
                            {turn.finishReason ?? "未指定"}
                        </span>
                    </span>
                )}
                {turn.status === "aborted" && (
                    <span className="tag tag-outline">已停止</span>
                )}
                {internalPresent && (
                    <label className="radio pg-toggle">
                        <input
                            type="checkbox"
                            checked={showInternal}
                            onChange={(event) =>
                                setShowInternal(event.target.checked)
                            }
                        />
                        <span className="dot" />
                        <span>顯示內部處理</span>
                    </label>
                )}
            </div>
        </div>
    );
}

function Avatar({
    character,
    orchestrator,
}: {
    character: CastCharacter;
    orchestrator: string | null;
}) {
    const lead = orchestrator === null || character.id === orchestrator;
    return (
        <span
            className={lead ? "pg-avatar" : "pg-avatar pg-avatar-2"}
            aria-hidden
            title={character.displayName}
        >
            {[...character.displayName][0] ?? "?"}
        </span>
    );
}

function SpeakerBlockView({
    block,
    cast,
    orchestrator,
    activeKey,
    showInternal,
}: {
    block: SpeakerBlock;
    cast: Map<string, CastCharacter>;
    orchestrator: string | null;
    activeKey: string | null;
    showInternal: boolean;
}) {
    // A single-character run sends no cast at all, so the agent id ("assistant")
    // is the only name there is.
    const character: CastCharacter = cast.get(block.agent) ?? {
        id: block.agent,
        displayName: block.agent,
        role: "",
    };
    const summoner = block.parent
        ? (cast.get(block.parent)?.displayName ?? block.parent)
        : null;
    const speaking = block.key === activeKey;

    return (
        <div
            className={
                block.depth > 0 ? "pg-speaker pg-speaker-nested" : "pg-speaker"
            }
        >
            <div className="pg-speaker-head">
                <Avatar character={character} orchestrator={orchestrator} />
                <span className="pg-name">{character.displayName}</span>
                {character.role && (
                    <span className="tag tag-neutral">{character.role}</span>
                )}
                {summoner && (
                    <span className="note">
                        由 {summoner} 召喚
                        {block.summonedBy && (
                            <>
                                {" · "}
                                <span className="mono">{block.summonedBy}</span>
                            </>
                        )}
                    </span>
                )}
                {speaking && <span className="pg-live" aria-label="正在發言" />}
            </div>

            <div className="pg-speaker-body">
                {block.items.map((item) => {
                    if (item.kind === "part") {
                        if (!showInternal && item.part.internal === true)
                            return null;
                        return (
                            <PartView
                                key={item.part.id}
                                part={item.part}
                                showInternal={showInternal}
                            />
                        );
                    }
                    if (!blockIsVisible(item.block, showInternal)) return null;
                    return (
                        <SpeakerBlockView
                            key={item.block.key}
                            block={item.block}
                            cast={cast}
                            orchestrator={orchestrator}
                            activeKey={activeKey}
                            showInternal={showInternal}
                        />
                    );
                })}
            </div>
        </div>
    );
}

/** What a part shows depends on whether it has finished: until `part_end`
 * lands, the only content there is is what the `delta`s have accumulated. */
function partText(part: StreamedPart): string {
    if (part.done) return part.text ?? "";
    return part.streamed;
}

function partArguments(part: StreamedPart): string {
    if (part.done && part.arguments !== undefined) {
        return JSON.stringify(part.arguments, null, 2);
    }
    // Mid-stream a tool call's arguments are raw JSON fragments from the model,
    // which are shown as they arrive rather than held back.
    return part.streamed;
}

function PartView({
    part,
    showInternal,
}: {
    part: StreamedPart;
    showInternal: boolean;
}) {
    const internal = part.internal === true;
    const wrap = (children: ReactNode) => (
        <div className={internal ? "pg-part pg-part-internal" : "pg-part"}>
            {internal && showInternal && (
                <span className="tag tag-outline pg-internal-tag">內部</span>
            )}
            {children}
        </div>
    );

    if (part.type === "text") {
        return wrap(
            <p className="pg-text">
                {partText(part)}
                {!part.done && <span className="pg-caret" />}
            </p>
        );
    }

    if (part.type === "reasoning") {
        return wrap(
            <details className="pg-aside">
                <summary>
                    <Brain size={13} strokeWidth={2.75} aria-hidden /> 推理過程
                    {!part.done && " — 思考中"}
                </summary>
                <p className="pg-aside-body">{partText(part)}</p>
            </details>
        );
    }

    if (part.type === "tool_call") {
        const rag = part.name === "rag_search";
        return wrap(
            <details className="pg-chip">
                <summary>
                    {rag ? (
                        <Search size={13} strokeWidth={2.75} aria-hidden />
                    ) : (
                        <Wrench size={13} strokeWidth={2.75} aria-hidden />
                    )}
                    <span className="mono">{part.name ?? "工具"}</span>
                    <span
                        className={
                            part.done ? "tag tag-neutral" : "tag tag-accent"
                        }
                    >
                        {part.done ? "已呼叫" : "呼叫中…"}
                    </span>
                </summary>
                <pre className="pg-code">
                    {partArguments(part) || "（無參數）"}
                </pre>
            </details>
        );
    }

    return wrap(
        <details className="pg-chip">
            <summary>
                <Wrench size={13} strokeWidth={2.75} aria-hidden />
                <span className="mono">結果</span>
                <span
                    className={
                        part.status === "error"
                            ? "tag tag-outline"
                            : part.done
                              ? "tag tag-accent-2"
                              : "tag tag-accent"
                    }
                >
                    {part.done ? (part.status ?? "已返回") : "執行中…"}
                </span>
                {part.code && <span className="mono note">{part.code}</span>}
            </summary>
            <pre className="pg-code">
                {part.content ?? (part.done ? "（無內容）" : "…")}
            </pre>
        </details>
    );
}
