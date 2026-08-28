/* Small display helpers. Everything here formats a value the API actually
 * sent — none of it invents a figure the backend does not report. */

const DATE_TIME = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
});

const TIME = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
});

export function formatDateTime(iso: string | null | undefined): string {
    if (!iso) return "—";
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso : DATE_TIME.format(date);
}

export function formatTime(iso: string | null | undefined): string {
    if (!iso) return "—";
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso : TIME.format(date);
}

/** `fetched_at` on the preset load report is a unix timestamp in seconds. */
export function formatUnixSeconds(value: number | null | undefined): string {
    if (value === null || value === undefined) return "—";
    return DATE_TIME.format(new Date(value * 1000));
}

export function formatDuration(seconds: number | null | undefined): string {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds))
        return "—";
    const total = Math.max(0, Math.round(seconds));
    if (total < 60) return `${total}s`;
    const minutes = Math.floor(total / 60);
    const rest = total % 60;
    if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

/** Langfuse names carry a folder prefix; the label drops it. */
export function stripFolder(name: string): string {
    const slash = name.lastIndexOf("/");
    return slash === -1 ? name : name.slice(slash + 1);
}

export function joinNames(names: string[], empty = "—"): string {
    return names.length === 0 ? empty : names.join(", ");
}

export function pluralise(
    count: number,
    one: string,
    many = `${one}s`
): string {
    return `${count} ${count === 1 ? one : many}`;
}
