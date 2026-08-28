import type { ReactNode } from 'react'

/** Loading says so in one quiet line. The design system prefers calm, so there
 * are no spinners anywhere in this console. */
export function Loading({ what }: { what: string }) {
  return <p className="quiet">Loading {what}…</p>
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="quiet">{children}</p>
}

/** The page header shared by every screen. */
export function PageHeader({
  kicker,
  title,
  lede,
  actions,
  back,
  mono,
}: {
  kicker?: ReactNode
  title: ReactNode
  lede?: ReactNode
  actions?: ReactNode
  back?: ReactNode
  /** Titles that are identifiers (a preset name, a run id) set in the body
   * face at bold rather than in Caprasimo. */
  mono?: boolean
}) {
  return (
    <div className={actions ? 'page-head page-head-split' : 'page-head'}>
      <div>
        {back}
        {kicker && <div className="page-kicker">{kicker}</div>}
        <h2 className={mono ? 'page-title page-title-mono mono' : 'page-title'}>{title}</h2>
        {lede && <p className="page-lede">{lede}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}
