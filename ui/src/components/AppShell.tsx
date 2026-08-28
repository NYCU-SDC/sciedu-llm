import { NavLink, Outlet } from 'react-router-dom'

import { LANGFUSE_URL } from '../api/client'

/** Exactly four destinations. The preset editor and the run detail are
 * sub-screens of their list, reached by opening a row — not nav entries. */
const NAV = [
  { to: '/rag', label: 'Retrieval settings' },
  { to: '/presets', label: 'Behaviour presets' },
  { to: '/evals', label: 'Evaluations' },
  { to: '/reference', label: "What's available" },
]

/** Whichever host the API calls actually go to, so the footer is not a claim
 * about a deployment this build may not be talking to. */
function connectedHost(): string {
  const base = import.meta.env.VITE_API_BASE_URL
  if (!base) return window.location.host
  try {
    return new URL(base, window.location.origin).host
  } catch {
    return base
  }
}

export function AppShell() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="sidebar-dot" />
          <div>
            <div className="sidebar-name">sciedu-llm</div>
            <div className="sidebar-kicker">Service console</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map((item) => (
            // NavLink is not `end`, so /presets stays lit while its editor is
            // open, and sets aria-current="page" — which is what the active
            // dot and tint key off in app.css.
            <NavLink key={item.to} to={item.to} className="btn nav-btn">
              <span className="nav-marker" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          Connected to <span className="mono">{connectedHost()}</span>
          {LANGFUSE_URL && (
            <>
              {' · traces in '}
              <a href={LANGFUSE_URL} target="_blank" rel="noreferrer">
                Langfuse
              </a>
            </>
          )}
          <div style={{ marginTop: 8 }}>
            Settings you change here live in the service's memory. A restart puts the
            server's own defaults back.
          </div>
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
