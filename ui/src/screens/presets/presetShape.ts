/* Reading and writing the preset document.
 *
 * The form is a view over the same JSON the JSON tab shows, so everything here
 * works on a `Preset` and returns a `Preset`. The server is the only authority
 * on whether a document is valid (`app/presets.py` holds the semantic rules);
 * what lives here is the mapping between the three plain-language
 * course-material choices and the `rag_mode` / `rag_search` pair the backend
 * actually stores, plus a shape check strong enough to catch a typo in the JSON
 * tab before it costs a round trip. */

import { locPath } from "../../api/errors";
import {
    MAX_STEPS_CAP,
    PRESET_ID_PATTERN,
    type Preset,
    type PresetCharacter,
    type ToolChoice,
} from "../../api/types";

export const RAG_SEARCH = "rag_search";
export const SUMMON_SUBAGENT = "summon_subagent";

/** The three ways a preset can relate to the course material, as the mockup
 * words them. They are not a backend field — they are a reading of `rag_mode`
 * together with whether anyone holds the `rag_search` tool. */
export type CourseMaterialMode = "never" | "always" | "decide";

export function courseMaterialMode(preset: Preset): CourseMaterialMode {
    if (preset.rag_mode === "forced") return "always";
    const hasSearch = preset.characters.some((character) =>
        character.tools.includes(RAG_SEARCH)
    );
    return hasSearch ? "decide" : "never";
}

export function describeRagMode(preset: Preset): string {
    switch (courseMaterialMode(preset)) {
        case "always":
            return "always searches";
        case "decide":
            return "model decides";
        default:
            return "never searches";
    }
}

function withoutSearch(character: PresetCharacter): PresetCharacter {
    return {
        ...character,
        tools: character.tools.filter((tool) => tool !== RAG_SEARCH),
    };
}

export function setCourseMaterialMode(
    preset: Preset,
    mode: CourseMaterialMode
): Preset {
    if (mode === "never") {
        return {
            ...preset,
            rag_mode: "off",
            characters: preset.characters.map(withoutSearch),
        };
    }
    if (mode === "always") {
        // `rag_mode: "forced"` supplies the retrieval itself, so the tool would only
        // fight it. The rest of the backend's rule (one character, no tools, no
        // prompt of its own) is reported to the user rather than enforced by
        // deleting their cast — see `forcedRagObjections`.
        return {
            ...preset,
            rag_mode: "forced",
            characters: preset.characters.map(withoutSearch),
        };
    }
    const characters = preset.characters.map((character) =>
        character.id === preset.orchestrator &&
        !character.tools.includes(RAG_SEARCH)
            ? { ...character, tools: [...character.tools, RAG_SEARCH] }
            : character
    );
    return { ...preset, rag_mode: "off", characters };
}

/** What still stands between this document and `rag_mode: "forced"`, in the
 * backend's own terms. Empty when the document would pass. */
export function forcedRagObjections(preset: Preset): string[] {
    if (preset.rag_mode !== "forced") return [];
    const objections: string[] = [];
    if (preset.characters.length !== 1) {
        objections.push(
            `"Always search" works with exactly one character; this preset has ${preset.characters.length}.`
        );
    }
    const only = preset.characters[0];
    if (only && only.tools.length > 0) {
        objections.push('"Always search" does not allow any tools.');
    }
    if (only && only.prompt_name) {
        objections.push(
            '"Always search" supplies the system prompt itself, so the character must not name one.'
        );
    }
    return objections;
}

export function blankPreset(): Preset {
    return {
        name: "",
        description: "",
        model: null,
        max_steps: 8,
        tool_choice: "auto",
        rag_mode: "off",
        orchestrator: "assistant",
        characters: [
            {
                id: "assistant",
                display_name: "Assistant",
                role: "assistant",
                prompt_name: null,
                tools: [],
                max_steps: 3,
            },
        ],
    };
}

export function blankCharacter(): PresetCharacter {
    return {
        id: "helper",
        display_name: "Helper",
        role: "assistant",
        prompt_name: null,
        tools: [],
        max_steps: 3,
    };
}

/** Comma-separated tool list ⇄ array, the way the mockup's "tools it may call"
 * field reads. */
export function parseTools(text: string): string[] {
    return text
        .split(",")
        .map((tool) => tool.trim())
        .filter(Boolean);
}

export function formatTools(tools: string[]): string {
    return tools.join(", ");
}

// ── the JSON tab's local check ────────────────────────────────────────────

export interface ShapeProblem {
    path: string;
    message: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A required-shape check, run locally when the user presses Validate in the
 * JSON tab. It catches the mistakes that are obvious without the server — a
 * missing field, a character id that is not a slug, an orchestrator naming
 * nobody. The semantic rules (which tools exist, what "forced" forbids) are the
 * server's, and run for real on Save. */
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
    ) {
        bad("description", "must be a string");
    }
    if (
        value.model !== undefined &&
        value.model !== null &&
        typeof value.model !== "string"
    ) {
        bad("model", "must be a model id, or null to use the server default");
    }
    if (
        value.max_steps !== undefined &&
        (typeof value.max_steps !== "number" ||
            !Number.isInteger(value.max_steps) ||
            value.max_steps < 1 ||
            value.max_steps > MAX_STEPS_CAP)
    ) {
        bad(
            "max_steps",
            `must be a whole number between 1 and ${MAX_STEPS_CAP}`
        );
    }
    if (
        value.tool_choice !== undefined &&
        !["auto", "none", "required"].includes(String(value.tool_choice))
    ) {
        bad("tool_choice", 'must be "auto", "none" or "required"');
    }
    if (
        value.rag_mode !== undefined &&
        !["off", "forced"].includes(String(value.rag_mode))
    ) {
        bad("rag_mode", 'must be "off" or "forced"');
    }

    const characters = value.characters;
    if (
        !Array.isArray(characters) ||
        characters.length < 1 ||
        characters.length > 2
    ) {
        bad("characters", "must be a list of one or two characters");
        return problems;
    }

    const ids: string[] = [];
    characters.forEach((character, index) => {
        const at = `characters[${index}]`;
        if (!isRecord(character)) {
            bad(at, "must be an object");
            return;
        }
        if (
            typeof character.id !== "string" ||
            !PRESET_ID_PATTERN.test(character.id)
        ) {
            bad(`${at}.id`, "must be a lowercase slug");
        } else {
            ids.push(character.id);
        }
        if (
            typeof character.display_name !== "string" ||
            !character.display_name
        ) {
            bad(`${at}.display_name`, "is required");
        }
        if (character.tools !== undefined) {
            if (
                !Array.isArray(character.tools) ||
                character.tools.some((tool) => typeof tool !== "string")
            ) {
                bad(`${at}.tools`, "must be a list of tool names");
            }
        }
        if (
            character.prompt_name !== undefined &&
            character.prompt_name !== null &&
            typeof character.prompt_name !== "string"
        ) {
            bad(`${at}.prompt_name`, "must be a prompt name, or null");
        }
    });

    const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
    if (duplicates.length > 0) {
        bad(
            "characters",
            `duplicate character ids: ${[...new Set(duplicates)].join(", ")}`
        );
    }
    if (typeof value.orchestrator !== "string") {
        bad("orchestrator", "is required");
    } else if (ids.length > 0 && !ids.includes(value.orchestrator)) {
        bad(
            "orchestrator",
            `must be one of the character ids: ${ids.join(", ")}`
        );
    }

    return problems;
}

/** Fill a parsed document out to a complete `Preset`, so the form tab can bind
 * to it without every field being optional. Only defaults the backend itself
 * declares are supplied. */
export function normalisePreset(value: Record<string, unknown>): Preset {
    const characters = Array.isArray(value.characters) ? value.characters : [];
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
        rag_mode: value.rag_mode === "forced" ? "forced" : "off",
        orchestrator:
            typeof value.orchestrator === "string"
                ? value.orchestrator
                : "assistant",
        characters: characters.map((raw): PresetCharacter => {
            const character = isRecord(raw) ? raw : {};
            return {
                id: typeof character.id === "string" ? character.id : "",
                display_name:
                    typeof character.display_name === "string"
                        ? character.display_name
                        : "",
                role:
                    typeof character.role === "string"
                        ? character.role
                        : "assistant",
                prompt_name:
                    typeof character.prompt_name === "string"
                        ? character.prompt_name
                        : null,
                tools: Array.isArray(character.tools)
                    ? character.tools.filter(
                          (tool): tool is string => typeof tool === "string"
                      )
                    : [],
                max_steps:
                    typeof character.max_steps === "number"
                        ? character.max_steps
                        : 3,
            };
        }),
    };
}

/** Turn a pydantic `loc` into the wording the form uses, so a 422 points at
 * something the user can see. Falls back to the dotted path. */
export function labelForLoc(loc: (string | number)[], preset: Preset): string {
    const path = locPath(loc);
    const [head, ...rest] = loc;
    if (head === "characters" && typeof rest[0] === "number") {
        const character = preset.characters[rest[0]];
        const who =
            character?.display_name ||
            character?.id ||
            `character ${rest[0] + 1}`;
        const field = rest[1];
        const fieldName =
            field === "prompt_name"
                ? "prompt"
                : field === "tools"
                  ? "tools"
                  : field === "display_name"
                    ? "display name"
                    : typeof field === "string"
                      ? field.replace(/_/g, " ")
                      : "";
        return fieldName ? `${who} — ${fieldName}` : String(who);
    }
    const simple: Record<string, string> = {
        name: "Preset name",
        model: "Model",
        description: "What this preset is for",
        max_steps: "Maximum steps per reply",
        tool_choice: "Tool choice",
        rag_mode: "Course material",
        orchestrator: "Orchestrator",
        characters: "Cast",
    };
    if (typeof head === "string" && simple[head] && loc.length === 1)
        return simple[head];
    return path;
}
