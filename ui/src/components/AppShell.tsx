import { NavLink, Outlet } from "react-router-dom";

import { LANGFUSE_URL } from "../api/client";
import { errorMessage } from "../api/errors";
import { useHealth } from "../api/hooks";

/** Exactly five destinations. The preset editor and the run detail are
 * sub-screens of their list, reached by opening a row — not nav entries. */
const NAV = [
    { to: "/rag", label: "檢索設定" },
    { to: "/presets", label: "行為預設" },
    { to: "/evals", label: "評估" },
    // Not part of docs/admin-ui-spec.md: a manual tester for POST /agents.
    { to: "/playground", label: "測試區" },
    { to: "/reference", label: "可用資源" },
];

export function AppShell() {
    return (
        <div className="shell">
            <header className="topbar">
                <div className="topbar-inner">
                    <NavLink to="/rag" className="brand">
                        <span className="brand-dot" />
                        <span className="brand-name">SciLLM Console</span>
                    </NavLink>

                    <nav className="topnav">
                        {NAV.map((item) => (
                            // NavLink is not `end`, so /presets stays lit while its editor is
                            // open, and sets aria-current="page" — which is what the active
                            // tab's accent underline keys off in app.css.
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className="topnav-link"
                            >
                                {item.label}
                            </NavLink>
                        ))}
                    </nav>

                    <BackendStatus />
                </div>
            </header>

            <main className="main">
                <Outlet />
            </main>
        </div>
    );
}

/** What the console can actually know about the backend: whether `GET /healthz`
 * answered, a few seconds ago. Not which host it went to — that is a build-time
 * setting and says nothing about whether anything is listening. */
function BackendStatus() {
    const health = useHealth();
    const waiting = health.isPending;
    const up = health.isSuccess;

    return (
        <div className="status" aria-live="polite">
            <span
                className={`status-dot ${waiting ? "status-wait" : up ? "status-on" : "status-off"}`}
                aria-hidden
            />
            <span title={health.error ? errorMessage(health.error) : undefined}>
                {waiting
                    ? "正在檢查後端"
                    : up
                      ? "後端已連線"
                      : "無法連線至後端"}
            </span>
            {LANGFUSE_URL && (
                <>
                    <span className="status-sep" aria-hidden>
                        ·
                    </span>
                    <a href={LANGFUSE_URL} target="_blank" rel="noreferrer">
                        在 Langfuse 查看追蹤紀錄
                    </a>
                </>
            )}
        </div>
    );
}
