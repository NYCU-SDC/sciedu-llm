import assert from "node:assert/strict";
import test from "node:test";

import {
    groupModelsByMode,
    isChatModelMode,
    isModelAllowed,
} from "./modelGroups.ts";

test("treats an empty model allow-list as unrestricted", () => {
    assert.equal(isModelAllowed("any-upstream-model", []), true);
    assert.equal(isModelAllowed("chat-a", ["chat-a"]), true);
    assert.equal(isModelAllowed("chat-b", ["chat-a"]), false);
});

test("only shows chat allow-list status for chat models", () => {
    assert.equal(isChatModelMode("chat"), true);
    assert.equal(isChatModelMode("embedding"), false);
    assert.equal(isChatModelMode("audio-stt"), false);
});

test("groups models by upstream model_mode in the display order", () => {
    const groups = groupModelsByMode([
        { id: "voice", model_mode: "audio-tts" },
        { id: "guard", model_mode: "guardrail" },
        { id: "chat", model_mode: "chat" },
        { id: "embed", model_mode: "embedding" },
        { id: "speech", model_mode: "audio-stt" },
        { id: "rerank", model_mode: "reranker" },
        { id: "custom", model_mode: "vision" },
        { id: "legacy", model_mode: null },
    ]);

    assert.deepEqual(
        groups.map(({ mode, title, models }) => ({
            mode,
            title,
            models: models.map((model) => model.id),
        })),
        [
            { mode: "chat", title: "聊天模型", models: ["chat"] },
            { mode: "embedding", title: "嵌入模型", models: ["embed"] },
            { mode: "reranker", title: "重排序模型", models: ["rerank"] },
            {
                mode: "audio-stt",
                title: "語音轉文字模型",
                models: ["speech"],
            },
            {
                mode: "audio-tts",
                title: "文字轉語音模型",
                models: ["voice"],
            },
            { mode: "guardrail", title: "防護模型", models: ["guard"] },
            { mode: "vision", title: "vision 模型", models: ["custom"] },
            {
                mode: "__unclassified__",
                title: "未分類模型",
                models: ["legacy"],
            },
        ]
    );
});
