import type { ModelInfo } from "../../api/types";

const MODE_ORDER = [
    "chat",
    "embedding",
    "reranker",
    "audio-stt",
    "audio-tts",
    "guardrail",
] as const;

const MODE_TITLES: Record<string, string> = {
    chat: "聊天模型",
    embedding: "嵌入模型",
    reranker: "重排序模型",
    "audio-stt": "語音轉文字模型",
    "audio-tts": "文字轉語音模型",
    guardrail: "防護模型",
};

const UNCLASSIFIED_MODE = "__unclassified__";

export interface ModelGroup {
    mode: string;
    title: string;
    models: ModelInfo[];
}

/** Group upstream models by their advertised mode, with the known modes in a
 * stable task-oriented order and any server extensions following by name. */
export function groupModelsByMode(models: ModelInfo[]): ModelGroup[] {
    const grouped = new Map<string, ModelInfo[]>();

    for (const model of models) {
        const mode = model.model_mode || UNCLASSIFIED_MODE;
        const group = grouped.get(mode) ?? [];
        group.push(model);
        grouped.set(mode, group);
    }

    const known = MODE_ORDER.filter((mode) => grouped.has(mode));
    const extensions = [...grouped.keys()]
        .filter(
            (mode) =>
                mode !== UNCLASSIFIED_MODE &&
                !(MODE_ORDER as readonly string[]).includes(mode)
        )
        .sort();
    const ordered = [
        ...known,
        ...extensions,
        ...(grouped.has(UNCLASSIFIED_MODE) ? [UNCLASSIFIED_MODE] : []),
    ];

    return ordered.map((mode) => ({
        mode,
        title:
            mode === UNCLASSIFIED_MODE
                ? "未分類模型"
                : (MODE_TITLES[mode] ?? `${mode} 模型`),
        models: grouped.get(mode) ?? [],
    }));
}

/** An empty allow-list is the server's unrestricted mode. */
export function isModelAllowed(
    modelId: string,
    allowedModels: string[]
): boolean {
    return allowedModels.length === 0 || allowedModels.includes(modelId);
}

/** Only chat models have a meaningful status in the /chat allow-list. */
export function isChatModelMode(mode: string): boolean {
    return mode === "chat";
}
