import assert from "node:assert/strict";
import test from "node:test";

import {
    enabledToolCount,
    normalisePreset,
    setRagSelection,
} from "./presetShape.ts";

test("normalises a legacy preset before the UI reads tool sections", () => {
    const preset = normalisePreset({
        name: "teacher-student",
        description: "legacy response",
        model: null,
        max_steps: 8,
        tool_choice: "auto",
        rag_mode: "off",
        orchestrator: "teacher",
        characters: [
            {
                id: "teacher",
                prompt_name: "agents/teacher-system",
                tools: ["rag_search", "summon_subagent"],
                max_steps: 3,
            },
            {
                id: "student",
                display_name: "Student",
                prompt_name: "agents/student",
                tools: ["rag_search"],
                max_steps: 4,
            },
        ],
    });

    assert.deepEqual(preset.tools.rag, {
        enable_tool: true,
        force: false,
        corpus_datasets: [],
    });
    assert.deepEqual(preset.tools.subagents, {
        enable_tool: true,
        character_forcing: true,
        prompt_name: "agents/student",
        max_steps: 4,
        personas: [],
    });
    assert.equal(preset.teacher_prompt_name, "agents/teacher-system");
    assert.equal(enabledToolCount(preset), 2);
});

test("fills absent tool sections instead of throwing", () => {
    const preset = normalisePreset({ name: "partial", max_steps: 8 });

    assert.equal(enabledToolCount(preset), 0);
    assert.deepEqual(preset.tools.rag, {
        enable_tool: false,
        force: false,
        corpus_datasets: [],
    });
    assert.deepEqual(preset.tools.subagents, {
        enable_tool: false,
        character_forcing: false,
        prompt_name: null,
        max_steps: 3,
        personas: [],
    });
});

test("preserves preset corpus selection while changing retrieval mode", () => {
    const preset = normalisePreset({
        name: "biology",
        tools: {
            rag: {
                enable_tool: true,
                force: false,
                corpus_datasets: ["corpus/biology"],
            },
        },
    });
    for (const mode of ["disabled", "enabled", "forced"]) {
        assert.deepEqual(
            setRagSelection(preset, mode).tools.rag.corpus_datasets,
            ["corpus/biology"]
        );
    }
    assert.deepEqual(normalisePreset(preset).tools.rag.corpus_datasets, [
        "corpus/biology",
    ]);
});
