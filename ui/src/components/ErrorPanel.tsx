import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

import { ApiError } from "../api/client";
import { errorMessage } from "../api/errors";
import { CopyButton } from "./CopyButton";

/** The mockup's warm-red "rejected document" panel, reused for every failure
 * the service reports. `detail` is always the API's own words — a FastAPI
 * `{"detail": …}` body, verbatim — so the message a user copies is the message
 * an operator can act on. */
export function ErrorPanel({
    title,
    detail,
    children,
    actions,
    copyText,
}: {
    title: ReactNode;
    detail?: ReactNode;
    children?: ReactNode;
    actions?: ReactNode;
    copyText?: string;
}) {
    return (
        <div className="alarm" role="alert">
            <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <AlertTriangle
                    size={17}
                    strokeWidth={2.75}
                    color="var(--color-alarm-ink)"
                    style={{ flex: "none", marginTop: 2 }}
                    aria-hidden
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="alarm-title">{title}</div>
                    {detail !== undefined &&
                        detail !== null &&
                        detail !== "" && (
                            <div className="alarm-body mono">{detail}</div>
                        )}
                    {children}
                </div>
            </div>
            {(actions || copyText) && (
                <div className="alarm-actions">
                    {actions}
                    {copyText && (
                        <CopyButton
                            text={copyText}
                            className="btn btn-secondary"
                            label="複製錯誤"
                        />
                    )}
                </div>
            )}
        </div>
    );
}

/** The same panel driven straight off a thrown value, with the HTTP status
 * spelled out because 502 (Langfuse unreachable) and 503 (RAG switched off)
 * mean very different things to the person reading it. */
export function QueryError({
    what,
    error,
    actions,
}: {
    what: string;
    error: unknown;
    actions?: ReactNode;
}) {
    const message = errorMessage(error);
    const status = error instanceof ApiError ? error.status : null;
    return (
        <ErrorPanel
            title={status ? `${what} — 服務回應 ${status}` : what}
            detail={message}
            copyText={status ? `${status} — ${message}` : message}
            actions={actions}
        />
    );
}
