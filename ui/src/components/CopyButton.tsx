import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";

/** Copy-to-clipboard with a two-second acknowledgement. Falls back to a
 * hidden textarea where the async clipboard API is unavailable (plain http
 * origins, which is how this console is often reached in a lab). */
export function CopyButton({
    text,
    label = "複製",
    className = "btn btn-secondary",
}: {
    text: string;
    label?: string;
    className?: string;
}) {
    const [copied, setCopied] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

    useEffect(() => () => clearTimeout(timer.current), []);

    const copy = async () => {
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const scratch = document.createElement("textarea");
                scratch.value = text;
                scratch.style.position = "fixed";
                scratch.style.opacity = "0";
                document.body.appendChild(scratch);
                scratch.select();
                document.execCommand("copy");
                document.body.removeChild(scratch);
            }
            setCopied(true);
            clearTimeout(timer.current);
            timer.current = setTimeout(() => setCopied(false), 2000);
        } catch {
            // Nothing useful to say — the text is on screen either way.
        }
    };

    return (
        <button type="button" className={className} onClick={() => void copy()}>
            {copied ? (
                <Check size={14} strokeWidth={2.75} aria-hidden />
            ) : (
                <Copy size={14} strokeWidth={2.75} aria-hidden />
            )}
            {copied ? "已複製" : label}
        </button>
    );
}
