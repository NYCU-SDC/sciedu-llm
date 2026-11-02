/* The edit buffer for the retrieval screen.
 *
 * Numbers live in the draft as strings so a half-typed value ("" on the way
 * from 500 to 400) does not become NaN mid-keystroke. They are parsed and
 * checked once, at Apply. */

import {
    REBUILD_KEYS,
    type RagConfigValues,
    type RagConfigUpdate,
} from "../../api/types";

export interface RagDraft {
    embedding_model: string;
    rerank_model: string;
    embedding_batch_size: string;
    max_concurrency: string;
    chunk_size: string;
    chunk_overlap: string;
    generator_system_prompt_name: string;
    generator_user_prompt_name: string;
    bm25_top_n: string;
    dense_top_n: string;
    rrf_k: string;
    rerank_pool_size: string;
    final_k: string;
}

export type NumericKey =
    | "embedding_batch_size"
    | "max_concurrency"
    | "chunk_size"
    | "chunk_overlap"
    | "bm25_top_n"
    | "dense_top_n"
    | "rrf_k"
    | "rerank_pool_size"
    | "final_k";

export type TextKey =
    | "embedding_model"
    | "rerank_model"
    | "generator_system_prompt_name"
    | "generator_user_prompt_name";

const NUMERIC_KEYS: NumericKey[] = [
    "embedding_batch_size",
    "max_concurrency",
    "chunk_size",
    "chunk_overlap",
    "bm25_top_n",
    "dense_top_n",
    "rrf_k",
    "rerank_pool_size",
    "final_k",
];

const TEXT_KEYS: TextKey[] = [
    "embedding_model",
    "rerank_model",
    "generator_system_prompt_name",
    "generator_user_prompt_name",
];

/** Plain-language names, matching the mockup's wording. */
export const FIELD_LABELS: Record<keyof RagDraft, string> = {
    embedding_model: "Embedding model",
    rerank_model: "Rerank model",
    embedding_batch_size: "Embedding batch size",
    max_concurrency: "Max concurrency",
    chunk_size: "Chunk size",
    chunk_overlap: "Chunk overlap",
    generator_system_prompt_name: "System prompt",
    generator_user_prompt_name: "User prompt",
    bm25_top_n: "Keyword top-n",
    dense_top_n: "Meaning top-n",
    rrf_k: "Merge constant (RRF k)",
    rerank_pool_size: "Rerank pool",
    final_k: "Final k",
};

/** `chunk_overlap` is the one field the schema allows at 0; everything else the
 * server declares `gt=0`, so a 0 would come back as a 422. */
const ALLOWS_ZERO: ReadonlySet<string> = new Set(["chunk_overlap"]);

export function toDraft(config: RagConfigValues): RagDraft {
    return {
        embedding_model: config.embedding_model,
        rerank_model: config.rerank_model,
        embedding_batch_size: String(config.embedding_batch_size),
        max_concurrency: String(config.max_concurrency),
        chunk_size: String(config.chunk_size),
        chunk_overlap: String(config.chunk_overlap),
        generator_system_prompt_name: config.generator_system_prompt_name,
        generator_user_prompt_name: config.generator_user_prompt_name,
        bm25_top_n: String(config.bm25_top_n),
        dense_top_n: String(config.dense_top_n),
        rrf_k: String(config.rrf_k),
        rerank_pool_size: String(config.rerank_pool_size),
        final_k: String(config.final_k),
    };
}

export interface DraftChange {
    key: keyof RagDraft;
    label: string;
    from: string;
    to: string;
    rebuilds: boolean;
}

const REBUILD_SET: ReadonlySet<string> = new Set<string>(REBUILD_KEYS);

export function diffDraft(server: RagDraft, draft: RagDraft): DraftChange[] {
    const changes: DraftChange[] = [];

    for (const key of [...TEXT_KEYS, ...NUMERIC_KEYS]) {
        if (server[key].trim() !== draft[key].trim()) {
            changes.push({
                key,
                label: FIELD_LABELS[key],
                from: server[key] || "(empty)",
                to: draft[key] || "(empty)",
                rebuilds: REBUILD_SET.has(key),
            });
        }
    }

    return changes;
}

export interface BuiltUpdate {
    update: RagConfigUpdate;
    /** Local problems, keyed by field — the same 422s the server would send, only
     * sooner. Server-side validation still has the last word. */
    problems: { key: keyof RagDraft; message: string }[];
}

export function buildUpdate(
    changes: DraftChange[],
    draft: RagDraft
): BuiltUpdate {
    const update: RagConfigUpdate = {};
    const problems: BuiltUpdate["problems"] = [];

    for (const change of changes) {
        if ((TEXT_KEYS as string[]).includes(change.key)) {
            const value = draft[change.key as TextKey].trim();
            if (!value) {
                problems.push({
                    key: change.key,
                    message: "This cannot be empty.",
                });
                continue;
            }
            update[change.key as TextKey] = value;
            continue;
        }

        const raw = draft[change.key as NumericKey].trim();
        const value = Number(raw);
        const floor = ALLOWS_ZERO.has(change.key) ? 0 : 1;
        if (!/^\d+$/.test(raw) || value < floor) {
            problems.push({
                key: change.key,
                message: `Must be a whole number${floor === 1 ? " of 1 or more" : " of 0 or more"}.`,
            });
            continue;
        }
        update[change.key as NumericKey] = value;
    }

    const size = Number(draft.chunk_size);
    const overlap = Number(draft.chunk_overlap);
    if (Number.isFinite(size) && Number.isFinite(overlap) && overlap >= size) {
        problems.push({
            key: "chunk_overlap",
            message: "Overlap has to be smaller than the chunk size.",
        });
    }

    return { update, problems };
}
