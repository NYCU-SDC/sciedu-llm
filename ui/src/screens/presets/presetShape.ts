/* Mapping and local validation for the tool-based preset document. */

import { locPath } from "../../api/errors.ts";
import {
    MAX_STEPS_CAP,
    PRESET_ID_PATTERN,
    type Preset,
    type PresetTools,
    type SubagentPersona,
    type SubagentToolConfig,
    type ToolChoice,
} from "../../api/types.ts";

export type RagSelection = "disabled" | "enabled" | "forced";
export const MAX_SUBAGENT_PERSONAS = 8;

export function ragSelection(preset: Preset): RagSelection {
    if (!preset.tools.rag.enable_tool) return "disabled";
    return preset.tools.rag.force ? "forced" : "enabled";
}

export function setRagSelection(
    preset: Preset,
    selection: RagSelection
): Preset {
    return {
        ...preset,
        // Forced retrieval supplies the teacher's system prompt itself.
        teacher_prompt_name:
            selection === "forced" ? null : preset.teacher_prompt_name,
        tools: {
            ...preset.tools,
            rag: {
                enable_tool: selection !== "disabled",
                force: selection === "forced",
            },
        },
    };
}

export function enabledToolCount(preset: Preset): number {
    return (
        Number(preset.tools.rag.enable_tool) +
        Number(preset.tools.subagents.enable_tool)
    );
}

export function blankPreset(): Preset {
    return {
        name: "",
        description: "",
        model: null,
        max_steps: 8,
        tool_choice: "auto",
        teacher_prompt_name: null,
        tools: {
            rag: { enable_tool: false, force: false },
            subagents: {
                enable_tool: false,
                character_forcing: false,
                prompt_name: null,
                max_steps: 3,
                personas: [],
            },
        },
    };
}

export interface ShapeProblem {
    path: string;
    message: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function checkPresetShape(value: unknown): ShapeProblem[] {
    const problems: ShapeProblem[] = [];
    const bad = (path: string, message: string) =>
        problems.push({ path, message });

    if (!isRecord(value))
        return [{ path: "(root)", message: "must be a JSON object" }];

    if (typeof value.name !== "string" || !PRESET_ID_PATTERN.test(value.name)) {
        bad(
            "name",
            'must be a lowercase slug: letters, digits, "-" and "_", up to 64 characters'
        );
    }
    if (
        value.description !== undefined &&
        typeof value.description !== "string"
    )
        bad("description", "must be a string");
    if (
        value.model !== undefined &&
        value.model !== null &&
        typeof value.model !== "string"
    )
        bad("model", "must be a model id, or null to use the server default");
    if (!wholeStep(value.max_steps))
        bad(
            "max_steps",
            `must be a whole number between 1 and ${MAX_STEPS_CAP}`
        );
    if (
        value.tool_choice !== undefined &&
        !["auto", "none", "required"].includes(String(value.tool_choice))
    )
        bad("tool_choice", 'must be "auto", "none" or "required"');
    if (
        value.teacher_prompt_name !== undefined &&
        value.teacher_prompt_name !== null &&
        typeof value.teacher_prompt_name !== "string"
    )
        bad("teacher_prompt_name", "must be a prompt name, or null");

    if (!isRecord(value.tools)) {
        bad("tools", "must contain the rag and subagents tool sections");
        return problems;
    }

    const rag = value.tools.rag;
    if (!isRecord(rag)) {
        bad("tools.rag", "is required");
    } else {
        if (typeof rag.enable_tool !== "boolean")
            bad("tools.rag.enable_tool", "must be true or false");
        if (typeof rag.force !== "boolean")
            bad("tools.rag.force", "must be true or false");
        if (rag.force === true && rag.enable_tool !== true)
            bad("tools.rag.force", "requires enable_tool=true");
    }

    const subagents = value.tools.subagents;
    if (!isRecord(subagents)) {
        bad("tools.subagents", "is required");
    } else {
        if (typeof subagents.enable_tool !== "boolean")
            bad("tools.subagents.enable_tool", "must be true or false");
        if (typeof subagents.character_forcing !== "boolean")
            bad("tools.subagents.character_forcing", "must be true or false");
        if (
            subagents.prompt_name !== undefined &&
            subagents.prompt_name !== null &&
            typeof subagents.prompt_name !== "string"
        )
            bad(
                "tools.subagents.prompt_name",
                "must be a prompt name, or null"
            );
        if (!wholeStep(subagents.max_steps))
            bad(
                "tools.subagents.max_steps",
                `must be a whole number between 1 and ${MAX_STEPS_CAP}`
            );
        if (
            subagents.character_forcing === true &&
            subagents.enable_tool !== true
        )
            bad(
                "tools.subagents.character_forcing",
                "requires enable_tool=true"
            );
        if (
            subagents.personas !== undefined &&
            !Array.isArray(subagents.personas)
        ) {
            bad("tools.subagents.personas", "must be an array");
        } else if (Array.isArray(subagents.personas)) {
            if (subagents.personas.length > MAX_SUBAGENT_PERSONAS)
                bad(
                    "tools.subagents.personas",
                    `may contain at most ${MAX_SUBAGENT_PERSONAS} personas`
                );
            const ids: string[] = [];
            subagents.personas.forEach((persona, index) => {
                const path = `tools.subagents.personas.${index}`;
                if (!isRecord(persona)) {
                    bad(path, "must be an object");
                    return;
                }
                if (
                    typeof persona.id !== "string" ||
                    !PRESET_ID_PATTERN.test(persona.id)
                )
                    bad(`${path}.id`, "must be a lowercase English slug");
                else ids.push(persona.id);
                if (
                    typeof persona.display_name !== "string" ||
                    persona.display_name.length === 0
                )
                    bad(`${path}.display_name`, "is required");
                if (
                    typeof persona.prompt_name !== "string" ||
                    persona.prompt_name.length === 0
                )
                    bad(`${path}.prompt_name`, "is required");
                if (!wholeStep(persona.max_steps))
                    bad(
                        `${path}.max_steps`,
                        `must be a whole number between 1 and ${MAX_STEPS_CAP}`
                    );
            });
            if (ids.includes("teacher"))
                bad(
                    "tools.subagents.personas",
                    'persona id "teacher" is reserved'
                );
            if (new Set(ids).size !== ids.length)
                bad("tools.subagents.personas", "persona ids must be unique");
        }
        if (
            subagents.character_forcing === true &&
            (!Array.isArray(subagents.personas) ||
                subagents.personas.length === 0) &&
            !subagents.prompt_name
        )
            bad(
                "tools.subagents.personas",
                "at least one persona is required when character forcing is enabled"
            );
    }

    if (
        isRecord(rag) &&
        rag.force === true &&
        value.teacher_prompt_name != null
    )
        bad(
            "teacher_prompt_name",
            "must be empty when RAG is forced because RAG supplies the system prompt"
        );

    return problems;
}

function wholeStep(value: unknown): boolean {
    return (
        typeof value === "number" &&
        Number.isInteger(value) &&
        value >= 1 &&
        value <= MAX_STEPS_CAP
    );
}

/** Normalise an API/import payload before any screen dereferences its shape.
 *
 * The legacy branch is intentionally retained while independently deployed UI
 * and backend containers can overlap during a rollout. A new UI may briefly
 * receive the pre-tool-section `characters`/`rag_mode` response from an older
 * backend; treating the TypeScript response annotation as runtime validation
 * made that ordinary deployment window crash the whole presets screen.
 */
export function normalisePreset(value: unknown): Preset {
    if (!isRecord(value)) return blankPreset();

    const tools = isRecord(value.tools) ? value.tools : {};
    const rag = isRecord(tools.rag) ? tools.rag : {};
    const subagents = isRecord(tools.subagents) ? tools.subagents : {};
    const legacy = !isRecord(value.tools) ? normaliseLegacyTools(value) : null;
    return {
        name: typeof value.name === "string" ? value.name : "",
        description:
            typeof value.description === "string" ? value.description : "",
        model: typeof value.model === "string" ? value.model : null,
        max_steps: typeof value.max_steps === "number" ? value.max_steps : 8,
        tool_choice: (["auto", "none", "required"] as const).includes(
            value.tool_choice as ToolChoice
        )
            ? (value.tool_choice as ToolChoice)
            : "auto",
        teacher_prompt_name:
            legacy !== null
                ? legacy.teacherPromptName
                : typeof value.teacher_prompt_name === "string"
                  ? value.teacher_prompt_name
                  : null,
        tools: legacy?.tools ?? {
            rag: normaliseRag(rag),
            subagents: normaliseSubagents(subagents),
        },
    };
}

function normaliseRag(value: Record<string, unknown>) {
    return {
        enable_tool: value.enable_tool === true,
        force: value.force === true,
    };
}

function normaliseSubagents(
    value: Record<string, unknown>
): SubagentToolConfig {
    return {
        enable_tool: value.enable_tool === true,
        character_forcing: value.character_forcing === true,
        prompt_name:
            typeof value.prompt_name === "string" ? value.prompt_name : null,
        max_steps: typeof value.max_steps === "number" ? value.max_steps : 3,
        personas: Array.isArray(value.personas)
            ? value.personas.filter(isRecord).map(normalisePersona)
            : [],
    };
}

function normalisePersona(value: Record<string, unknown>): SubagentPersona {
    return {
        id: typeof value.id === "string" ? value.id : "",
        display_name:
            typeof value.display_name === "string" ? value.display_name : "",
        prompt_name:
            typeof value.prompt_name === "string" ? value.prompt_name : "",
        max_steps: typeof value.max_steps === "number" ? value.max_steps : 3,
    };
}

function stringList(value: unknown): string[] {
    return Array.isArray(value)
        ? value.filter((entry): entry is string => typeof entry === "string")
        : [];
}

function normaliseLegacyTools(value: Record<string, unknown>): {
    teacherPromptName: string | null;
    tools: PresetTools;
} | null {
    if (!Array.isArray(value.characters)) return null;

    const characters = value.characters.filter(isRecord);
    const orchestrator =
        typeof value.orchestrator === "string"
            ? value.orchestrator
            : "assistant";
    const teacher =
        characters.find((character) => character.id === orchestrator) ??
        characters[0] ??
        {};
    const subagentCharacters = characters.filter(
        (character) => character !== teacher
    );
    const firstSubagent = subagentCharacters[0] ?? {};
    const teacherTools = stringList(teacher.tools);
    const allTools = characters.flatMap((character) =>
        stringList(character.tools)
    );
    const subagentsEnabled =
        teacherTools.includes("summon_subagent") &&
        subagentCharacters.length > 0;
    const promptName =
        typeof firstSubagent.prompt_name === "string"
            ? firstSubagent.prompt_name
            : null;

    return {
        teacherPromptName:
            typeof teacher.prompt_name === "string"
                ? teacher.prompt_name
                : null,
        tools: {
            rag: {
                enable_tool:
                    value.rag_mode === "forced" ||
                    allTools.includes("rag_search"),
                force: value.rag_mode === "forced",
            },
            subagents: {
                enable_tool: subagentsEnabled,
                character_forcing: subagentsEnabled && promptName !== null,
                prompt_name: promptName,
                max_steps:
                    typeof firstSubagent.max_steps === "number"
                        ? firstSubagent.max_steps
                        : 3,
                personas:
                    subagentCharacters.length > 1
                        ? subagentCharacters.map(normalisePersona)
                        : [],
            },
        },
    };
}

/** Materialise the legacy forced student as a persona without rewriting it. */
export function configuredSubagentPersonas(
    subagents: SubagentToolConfig
): SubagentPersona[] {
    if (subagents.personas?.length) return subagents.personas;
    if (!subagents.character_forcing || !subagents.prompt_name) return [];
    return [
        {
            id: "student",
            display_name: "學生",
            prompt_name: subagents.prompt_name,
            max_steps: subagents.max_steps,
        },
    ];
}

export function labelForLoc(loc: (string | number)[]): string {
    const path = locPath(loc);
    const labels: Record<string, string> = {
        name: "Preset name",
        model: "Model",
        description: "What this preset is for",
        max_steps: "Maximum steps per reply",
        tool_choice: "Tool choice",
        teacher_prompt_name: "Teacher prompt",
        "tools.rag": "RAG",
        "tools.rag.enable_tool": "RAG — enabled",
        "tools.rag.force": "RAG — force",
        "tools.subagents": "Subagents",
        "tools.subagents.enable_tool": "Subagents — enabled",
        "tools.subagents.character_forcing": "Subagents — character forcing",
        "tools.subagents.prompt_name": "Student prompt",
        "tools.subagents.max_steps": "Subagent maximum steps",
        "tools.subagents.personas": "Subagent personas",
    };
    const personaMatch = path.match(
        /^tools\.subagents\.personas\.(\d+)\.(id|display_name|prompt_name|max_steps)$/
    );
    if (personaMatch) {
        const fields: Record<string, string> = {
            id: "Persona ID",
            display_name: "Persona display name",
            prompt_name: "Persona prompt",
            max_steps: "Persona maximum steps",
        };
        return `Persona ${Number(personaMatch[1]) + 1} — ${fields[personaMatch[2]]}`;
    }
    return labels[path] ?? path;
}
