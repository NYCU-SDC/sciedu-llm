/* Fetch-based SSE for `POST /agents`.
 *
 * `EventSource` cannot POST, and this endpoint needs a request body (the
 * conversation, the preset, the trace identity), so the stream is read off a
 * `fetch` response body by hand.
 *
 * The wire format is `data: {json}\n\n`, UTF-8, *not* ascii-escaped — unlike
 * /chat, `src/app/routers/agents.py` dumps with `ensure_ascii=False` because the
 * corpus is Traditional Chinese. That is why frames are decoded with a real
 * streaming `TextDecoder`: a multi-byte character can be split across two
 * network chunks, and a byte-per-character shortcut would mangle it.
 *
 * The event union below is the table in `docs/agents-spec.md`; the part shape is
 * `Part.end_payload()` in `src/app/agents/events.py`. Nothing is cast: every
 * frame is checked field by field, and a frame this app does not recognise — an
 * event type added to the protocol later, or one missing a field the UI relies
 * on — is dropped rather than half-rendered. */

import { ApiError, BASE, NetworkError } from "./client";

// ── the protocol ──────────────────────────────────────────────────────────

/** One conversation message. The playground only ever sends these two roles;
 * the backend accepts the full OpenAI message union. */
export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
}

export type PartType = "text" | "reasoning" | "tool_call" | "tool_result";
export type ToolResultStatus = "ok" | "error";

/** A typed step inside one assistant message. `part_start` carries only the
 * identifying fields; the streaming ones arrive as `delta`s and the complete
 * part comes back on `part_end`. */
export interface AgentPart {
    type: PartType;
    id: string;
    agent: string;
    /** Present (and always `true`) for the summon machinery the frontend hides. */
    internal?: boolean;
    /** text / reasoning. */
    text?: string;
    /** tool_call / tool_result. */
    tool_call_id?: string;
    /** tool_call. */
    name?: string;
    arguments?: unknown;
    /** tool_result. */
    status?: ToolResultStatus;
    code?: string;
    content?: string;
}

/** A speaker the UI needs a nameplate for. Sent once, at the head of a
 * multi-character run; omitted entirely for a single-character preset. */
export interface CastCharacter {
    id: string;
    displayName: string;
    role: string;
}

export type AgentEvent =
    | { type: "cast"; characters: CastCharacter[] }
    | {
          type: "agent_start";
          agent: string;
          parent?: string;
          summonedBy?: string;
      }
    | { type: "agent_end"; agent: string }
    | { type: "part_start"; index: number; part: AgentPart }
    | { type: "delta"; index: number; delta: string }
    | { type: "part_end"; index: number; part: AgentPart }
    | { type: "done"; finishReason: string | null; status: string }
    | { type: "error"; error: string; code: string };

// ── parsing one frame ─────────────────────────────────────────────────────

function asRecord(value: unknown): Record<string, unknown> | null {
    return typeof value === "object" && value !== null && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function asString(value: unknown): string | undefined {
    return typeof value === "string" ? value : undefined;
}

function asIndex(value: unknown): number | undefined {
    return typeof value === "number" && Number.isInteger(value) && value >= 0
        ? value
        : undefined;
}

const PART_TYPES: readonly PartType[] = [
    "text",
    "reasoning",
    "tool_call",
    "tool_result",
];

function parsePart(value: unknown): AgentPart | null {
    const raw = asRecord(value);
    if (!raw) return null;
    const type = asString(raw.type);
    const id = asString(raw.id);
    const agent = asString(raw.agent);
    if (
        !type ||
        !(PART_TYPES as readonly string[]).includes(type) ||
        !id ||
        !agent
    ) {
        return null;
    }
    const part: AgentPart = { type: type as PartType, id, agent };
    if (raw.internal === true) part.internal = true;
    if (typeof raw.text === "string") part.text = raw.text;
    if (typeof raw.tool_call_id === "string")
        part.tool_call_id = raw.tool_call_id;
    if (typeof raw.name === "string") part.name = raw.name;
    if (raw.arguments !== undefined) part.arguments = raw.arguments;
    if (raw.status === "ok" || raw.status === "error") part.status = raw.status;
    if (typeof raw.code === "string") part.code = raw.code;
    if (typeof raw.content === "string") part.content = raw.content;
    return part;
}

function parseCharacters(value: unknown): CastCharacter[] | null {
    if (!Array.isArray(value)) return null;
    const characters: CastCharacter[] = [];
    for (const entry of value) {
        const raw = asRecord(entry);
        const id = asString(raw?.id);
        if (!raw || !id) return null;
        characters.push({
            id,
            // A cast entry without a display name is still a speaker; fall back to
            // its id rather than dropping the whole event.
            displayName: asString(raw.displayName) ?? id,
            role: asString(raw.role) ?? "",
        });
    }
    return characters;
}

/** One decoded `data:` payload → a typed event, or null when it is not one this
 * app knows how to render. */
export function parseAgentEvent(value: unknown): AgentEvent | null {
    const raw = asRecord(value);
    if (!raw) return null;
    switch (asString(raw.type)) {
        case "cast": {
            const characters = parseCharacters(raw.characters);
            return characters ? { type: "cast", characters } : null;
        }
        case "agent_start": {
            const agent = asString(raw.agent);
            if (!agent) return null;
            const event: AgentEvent = { type: "agent_start", agent };
            const parent = asString(raw.parent);
            const summonedBy = asString(raw.summonedBy);
            if (parent !== undefined) event.parent = parent;
            if (summonedBy !== undefined) event.summonedBy = summonedBy;
            return event;
        }
        case "agent_end": {
            const agent = asString(raw.agent);
            return agent ? { type: "agent_end", agent } : null;
        }
        case "part_start":
        case "part_end": {
            const index = asIndex(raw.index);
            const part = parsePart(raw.part);
            if (index === undefined || !part) return null;
            return raw.type === "part_start"
                ? { type: "part_start", index, part }
                : { type: "part_end", index, part };
        }
        case "delta": {
            const index = asIndex(raw.index);
            const delta = asString(raw.delta);
            if (index === undefined || delta === undefined) return null;
            return { type: "delta", index, delta };
        }
        case "done":
            return {
                type: "done",
                // `finishReason` is nullable on the wire (DoneEvent defaults it to None).
                finishReason: asString(raw.finishReason) ?? null,
                status: asString(raw.status) ?? "completed",
            };
        case "error":
            return {
                type: "error",
                error: asString(raw.error) ?? "執行失敗。",
                code: asString(raw.code) ?? "unknown",
            };
        default:
            return null;
    }
}

// ── framing ───────────────────────────────────────────────────────────────

/** SSE separates events with a blank line; both line endings are legal. */
const FRAME_BOUNDARY = /\r?\n\r?\n/;

/** Split whatever complete frames the buffer holds, keeping the tail — a frame
 * cut in half by a chunk boundary — for the next read. */
function drainFrames(buffer: string): { frames: string[]; rest: string } {
    const frames: string[] = [];
    let rest = buffer;
    for (;;) {
        const match = FRAME_BOUNDARY.exec(rest);
        if (!match) break;
        frames.push(rest.slice(0, match.index));
        rest = rest.slice(match.index + match[0].length);
    }
    return { frames, rest };
}

/** The `data:` payload of one frame. Comment lines (`: keep-alive`) and fields
 * this endpoint never sends (`event:`, `id:`, `retry:`) are ignored; multiple
 * `data:` lines join with a newline, as the SSE spec prescribes. */
function frameData(frame: string): string | null {
    const parts: string[] = [];
    for (const line of frame.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        parts.push(line.slice(5).replace(/^ /, ""));
    }
    return parts.length > 0 ? parts.join("\n") : null;
}

function decodeFrame(frame: string): AgentEvent | null {
    const data = frameData(frame);
    if (data === null || data === "") return null;
    try {
        return parseAgentEvent(JSON.parse(data));
    } catch {
        // A frame that is not JSON is not something this screen can render. It is
        // dropped rather than surfaced: the stream itself is still well-formed.
        return null;
    }
}

// ── the request ───────────────────────────────────────────────────────────

export interface AgentsRun {
    messages: ChatMessage[];
    /** Omitted from the body entirely when undefined, so the server's own
     * `AGENTS_DEFAULT_PRESET` decides. */
    preset?: string;
    session: string;
    user: string;
}

/** POST /agents and hand every typed event to `onEvent` as it lands.
 *
 * Resolves when the server closes the stream — which is *not* the same as the
 * run having succeeded: a failed run ends with an `error` event and a normal
 * close. Rejects with `ApiError` (the endpoint answered a status), or
 * `NetworkError` (it did not answer at all, or the connection died mid-stream),
 * or the abort reason when `signal` fires. */
export async function streamAgents(
    run: AgentsRun,
    {
        signal,
        onEvent,
    }: { signal: AbortSignal; onEvent: (event: AgentEvent) => void }
): Promise<void> {
    const body = {
        messages: run.messages,
        stream: true,
        ...(run.preset === undefined ? {} : { preset: run.preset }),
        session: run.session,
        user: run.user,
    };

    let response: Response;
    try {
        response = await fetch(`${BASE}/agents`, {
            method: "POST",
            signal,
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
            },
            body: JSON.stringify(body),
        });
    } catch (error) {
        if (signal.aborted) throw error;
        throw new NetworkError(
            `無法連線至 ${BASE || window.location.origin}/agents 的服務。` +
                `${error instanceof Error ? error.message : String(error)}`
        );
    }

    if (!response.ok) {
        // A refusal is a normal FastAPI body — an unknown preset is a 400 listing
        // what is available, a broken preset a 502/503 — so it reads like every
        // other failure in this console.
        const text = await response.text();
        let payload: unknown = text;
        try {
            payload = JSON.parse(text);
        } catch {
            /* not JSON; the raw text is the best detail there is */
        }
        const detail =
            payload && typeof payload === "object" && "detail" in payload
                ? (payload as { detail: unknown }).detail
                : payload;
        throw new ApiError(
            response.status,
            detail,
            `${response.status} ${response.statusText}`
        );
    }

    if (!response.body) {
        throw new NetworkError("此瀏覽器未提供可讀取的 /agents 串流內容。");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            // `stream: true` keeps a split multi-byte character in the decoder until
            // the rest of it arrives.
            buffer += decoder.decode(value, { stream: true });
            const drained = drainFrames(buffer);
            buffer = drained.rest;
            for (const frame of drained.frames) {
                const event = decodeFrame(frame);
                if (event) onEvent(event);
            }
        }
        // Flush the decoder, then anything left in the buffer: a well-behaved
        // stream ends with a blank line, but a truncated one may not.
        buffer += decoder.decode();
        const tail = decodeFrame(buffer);
        if (tail) onEvent(tail);
    } catch (error) {
        if (signal.aborted) throw error;
        throw new NetworkError(
            `/agents 串流提前結束。${error instanceof Error ? error.message : String(error)}`
        );
    } finally {
        // Aborting the fetch already cancels the body; this covers the ordinary
        // path and makes an early `return` safe if one is ever added.
        reader.releaseLock();
    }
}
