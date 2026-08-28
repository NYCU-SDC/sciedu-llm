import { useId, type ReactNode } from 'react'

/** The mockup's `.panel` + `.sect` pair: a bordered card with a small uppercase
 * accent heading. */
export function Panel({
  title,
  actions,
  children,
  className,
  style,
}: {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <section className={className ? `panel ${className}` : 'panel'} style={style}>
      {(title || actions) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          {title ? <h5 className="sect">{title}</h5> : <span />}
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

/** A `.field` wrapper — label above the control, optional note below.
 *
 * Pass a function as the child to receive a generated id, so the label and the
 * control it names are actually associated. */
export function Field({
  label,
  hint,
  children,
  style,
}: {
  label: ReactNode
  hint?: ReactNode
  children: ReactNode | ((id: string) => ReactNode)
  style?: React.CSSProperties
}) {
  const id = useId()
  return (
    <div className="field" style={style}>
      <label htmlFor={id}>{label}</label>
      {typeof children === 'function' ? children(id) : children}
      {hint && (
        <span className="note" style={{ display: 'block', marginTop: 7 }}>
          {hint}
        </span>
      )}
    </div>
  )
}
