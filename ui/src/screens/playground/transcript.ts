/* Turning a stream of typed events into something that reads like a
 * conversation.
 *
 * The screen keeps the raw `AgentEvent[]` per assistant turn and rebuilds this
 * view of it on every render. That is deliberate: the protocol's alignment rules
 * (`index` binds a `delta` to its part, `agent_start`/`agent_end` say who holds
 * the floor) are stated once, in one pure function, rather than smeared across
 * a reducer and a component. Runs are tens of parts long, so rebuilding costs
 * nothing worth optimising.
 *
 * The shape it produces is a tree, because the spec's point is that a summoned
 * character is a *speaker*, not a folded-up tool result: an `agent_start`
 * carrying `parent`/`summonedBy` opens a block nested inside the block of the
 * character that called it in, exactly where the summon happened. */

import type {
    AgentEvent,
    AgentPart,
    CastCharacter,
    ChatMessage,
} from "../../api/agentsStream";

/** A part plus what streaming added to it. `streamed` is the concatenation of
 * every `delta` for this index — the answer text as it arrives, or, for a
 * `tool_call`, the raw JSON fragments of its arguments. */
export interface StreamedPart extends AgentPart {
    index: number;
    streamed: string;
    /** `part_end` has been seen, so the fields above are the complete part. */
    done: boolean;
}

export type BlockItem =
    | { kind: "part"; part: StreamedPart }
    | { kind: "block"; block: SpeakerBlock };

/** One stretch of a turn during which a single character was speaking. */
export interface SpeakerBlock {
    key: string;
    agent: string;
    /** Set when this character was summoned: who called it in, and on which tool
     * call. Both come straight off `agent_start`. */
    parent: string | null;
    summonedBy: string | null;
    depth: number;
    items: BlockItem[];
    closed: boolean;
}

export type TurnStatus = "streaming" | "completed" | "failed" | "aborted";

export interface AssistantTurn {
    /** Empty for a single-character preset — the server omits `cast` then. */
    cast: CastCharacter[];
    blocks: SpeakerBlock[];
    parts: StreamedPart[];
    /** The block still holding the floor, for the pulsing "speaking now" mark. */
    activeBlockKey: string | null;
    status: TurnStatus;
    finishReason: string | null;
    error: { message: string; code: string } | null;
}

// ── building it ───────────────────────────────────────────────────────────

/** How a turn stopped, when the stream itself did not say. A run that ended
 * with `done` or `error` needs neither; a stopped one and a connection that
 * died mid-frame both do, or the speaker who had the floor would keep pulsing
 * forever. */
export interface TurnEnd {
    aborted: boolean;
    failed: boolean;
}

export function buildTurn(events: AgentEvent[], end: TurnEnd): AssistantTurn {
    let cast: CastCharacter[] = [];
    const parts: StreamedPart[] = [];
    const byIndex = new Map<number, StreamedPart>();
    const blocks: SpeakerBlock[] = [];
    // Only used while building — a block's parent is not part of the rendered
    // shape, and putting it there would make the tree cyclic.
    const parentOf = new Map<string, SpeakerBlock | null>();
    let current: SpeakerBlock | null = null;
    let finishReason: string | null = null;
    let error: { message: string; code: string } | null = null;
    let done = false;
    let counter = 0;

    const open = (
        agent: string,
        parent: string | null,
        summonedBy: string | null
    ) => {
        const nested =
            (parent !== null || summonedBy !== null) && current !== null;
        const block: SpeakerBlock = {
            key: `b${counter++}`,
            agent,
            parent,
            summonedBy,
            depth: nested && current ? current.depth + 1 : 0,
            items: [],
            closed: false,
        };
        if (nested && current) {
            current.items.push({ kind: "block", block });
            parentOf.set(block.key, current);
        } else {
            blocks.push(block);
            parentOf.set(block.key, null);
        }
        current = block;
        return block;
    };

    /** Where a part belongs. Normally the speaker holding the floor — but the
     * `tool_result` that hands a summoned character's answer back to its caller
     * arrives *after* that character's `agent_end` and before the caller's next
     * `agent_start`, so a part is matched by its own `agent` field first. */
    const blockFor = (agent: string): SpeakerBlock => {
        if (current && current.agent === agent && !current.closed)
            return current;
        for (
            let walk: SpeakerBlock | null = current;
            walk;
            walk = parentOf.get(walk.key) ?? null
        ) {
            if (walk.agent === agent && !walk.closed) {
                current = walk;
                return walk;
            }
        }
        return open(agent, null, null);
    };

    for (const event of events) {
        switch (event.type) {
            case "cast":
                cast = event.characters;
                break;

            case "agent_start": {
                const summoned =
                    event.parent !== undefined ||
                    event.summonedBy !== undefined;
                // The orchestrator is announced again every time it takes the floor
                // back (starts are teacher, student, teacher; ends are student,
                // teacher). Re-announcing whoever is already speaking must not open a
                // second, empty nameplate.
                if (
                    !summoned &&
                    current &&
                    current.agent === event.agent &&
                    !current.closed
                ) {
                    break;
                }
                open(
                    event.agent,
                    event.parent ?? null,
                    event.summonedBy ?? null
                );
                break;
            }

            case "agent_end": {
                for (
                    let walk: SpeakerBlock | null = current;
                    walk;
                    walk = parentOf.get(walk.key) ?? null
                ) {
                    if (walk.agent === event.agent) {
                        walk.closed = true;
                        current = parentOf.get(walk.key) ?? null;
                        break;
                    }
                }
                break;
            }

            case "part_start": {
                const part: StreamedPart = {
                    ...event.part,
                    index: event.index,
                    streamed: "",
                    done: false,
                };
                parts.push(part);
                byIndex.set(event.index, part);
                blockFor(part.agent).items.push({ kind: "part", part });
                break;
            }

            case "delta": {
                const part = byIndex.get(event.index);
                if (part) part.streamed += event.delta;
                break;
            }

            case "part_end": {
                const part = byIndex.get(event.index);
                if (!part) break;
                // The complete part replaces what streaming accumulated, field by
                // field; `streamed` is kept so a re-render mid-stream and after it look
                // the same.
                Object.assign(part, event.part, { done: true });
                break;
            }

            case "done":
                finishReason = event.finishReason;
                done = true;
                break;

            case "error":
                error = { message: event.error, code: event.code };
                break;
        }
    }

    const status: TurnStatus = error
        ? "failed"
        : done
          ? "completed"
          : end.failed
            ? "failed"
            : end.aborted
              ? "aborted"
              : "streaming";

    return {
        cast,
        blocks,
        parts,
        activeBlockKey:
            status === "streaming" && current && !current.closed
                ? current.key
                : null,
        status,
        finishReason,
        error,
    };
}

// ── reading it back ───────────────────────────────────────────────────────

/** Whether anything in this block — or in a block nested inside it — would be
 * hidden by the internal-machinery toggle. */
export function hasInternal(block: SpeakerBlock): boolean {
    return block.items.some((item) =>
        item.kind === "part"
            ? item.part.internal === true
            : hasInternal(item.block)
    );
}

/** A block with nothing showing (a stretch that held only the summon
 * machinery) is not drawn at all — an empty nameplate would be a lie about who
 * spoke. */
export function blockIsVisible(
    block: SpeakerBlock,
    showInternal: boolean
): boolean {
    if (showInternal) return block.items.length > 0;
    return block.items.some((item) =>
        item.kind === "part"
            ? item.part.internal !== true
            : blockIsVisible(item.block, false)
    );
}

// ── the conversation the next request carries ─────────────────────────────

export interface UserTurn {
    kind: "user";
    id: string;
    content: string;
}

export interface AssistantTurnState {
    kind: "assistant";
    id: string;
    /** What was actually named on the request; null means none was sent, so the
     * server's default preset ran. */
    preset: string | null;
    events: AgentEvent[];
    aborted: boolean;
    /** A transport or HTTP failure — the run never reached the point of emitting
     * an `error` event. */
    failure: { title: string; detail: string } | null;
}

export type Turn = UserTurn | AssistantTurnState;

/** Fold the transcript back into the `messages` array for the next turn.
 *
 * /agents is stateless: it is handed the whole conversation every time, and it
 * has no notion of a previous run. A finished assistant turn — which on the
 * wire was a cast, several speakers and a pile of typed parts — therefore has
 * to collapse into ONE `{role: "assistant", content}` message. The content is
 * every non-internal `text` part in index order, joined by blank lines: that is
 * what was actually said out loud, teacher and student alike. Reasoning, tool
 * calls and tool results are dropped — they are this run's private working, and
 * the model that reads them back would treat them as its own prior speech.
 *
 * A turn that produced no text (aborted before the first token, or refused
 * outright) contributes no message rather than an empty one, which the upstream
 * API would reject. */
export function historyMessages(turns: Turn[]): ChatMessage[] {
    const messages: ChatMessage[] = [];
    for (const turn of turns) {
        if (turn.kind === "user") {
            messages.push({ role: "user", content: turn.content });
            continue;
        }
        const content = spokenText(
            buildTurn(turn.events, {
                aborted: turn.aborted,
                failed: turn.failure !== null,
            })
        );
        if (content) messages.push({ role: "assistant", content });
    }
    return messages;
}

/** The non-internal text of a finished turn — what the reader saw. */
export function spokenText(turn: AssistantTurn): string {
    return turn.parts
        .filter((part) => part.type === "text" && part.internal !== true)
        .map((part) => (part.done ? (part.text ?? "") : part.streamed))
        .filter((text) => text.trim() !== "")
        .join("\n\n");
}
