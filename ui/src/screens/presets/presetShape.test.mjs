import assert from "node:assert/strict";
import test from "node:test";

import { enabledToolCount, normalisePreset } from "./presetShape.ts";

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
    });
    assert.deepEqual(preset.tools.subagents, {
        enable_tool: false,
        character_forcing: false,
        prompt_name: null,
        max_steps: 3,
        personas: [],
    });
});
