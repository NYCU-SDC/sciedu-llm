/* Reading FastAPI's two error shapes.
 *
 * `HTTPException(detail="…")` gives a string; request-body validation gives a
 * list of pydantic problems, each with a `loc` path into the submitted
 * document. The preset editor uses `loc` to point at the offending field, so we
 * keep the structure rather than flattening straight to a message. */

export interface ValidationProblem {
    /** The pydantic `loc` with the leading "body" segment dropped. */
    loc: (string | number)[];
    msg: string;
    type?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function detailToProblems(detail: unknown): ValidationProblem[] {
    if (!Array.isArray(detail)) return [];
    const problems: ValidationProblem[] = [];
    for (const entry of detail) {
        if (!isRecord(entry)) continue;
        const rawLoc = Array.isArray(entry.loc) ? entry.loc : [];
        const loc = rawLoc
            .filter(
                (part): part is string | number =>
                    typeof part === "string" || typeof part === "number"
            )
            // FastAPI prefixes body-validation locs with "body"; the UI thinks in
            // terms of the preset document, which starts one level in.
            .filter((part, index) => !(index === 0 && part === "body"));
        problems.push({
            loc,
            msg: typeof entry.msg === "string" ? entry.msg : "invalid",
            type: typeof entry.type === "string" ? entry.type : undefined,
        });
    }
    return problems;
}

/** A dotted path, the way pydantic writes it in the preset load report. */
export function locPath(loc: (string | number)[]): string {
    if (loc.length === 0) return "(root)";
    return loc.reduce<string>((path, part) => {
        if (typeof part === "number") return `${path}[${part}]`;
        return path ? `${path}.${part}` : part;
    }, "");
}

export function detailToMessage(detail: unknown): string {
    if (typeof detail === "string") return detail;
    const problems = detailToProblems(detail);
    if (problems.length > 0) {
        return problems.map((p) => `${locPath(p.loc)}: ${p.msg}`).join("; ");
    }
    if (detail === undefined || detail === null) return "";
    return typeof detail === "object" ? JSON.stringify(detail) : String(detail);
}

/** The single line to show in an error panel for any thrown value. */
export function errorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    if (typeof error === "string") return error;
    return "發生未預期的錯誤。";
}
