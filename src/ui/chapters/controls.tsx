import type { ReactNode } from 'react'

/** Small segmented selector used by chapters (layer / head / mode). */
export function Seg<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <span className="field">
      <span>{label}</span>
      <span className="segmented">
        {options.map((o) => (
          <button key={String(o.value)} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </span>
    </span>
  )
}

/** Chapter-level selectors that stay visible while the stage scrolls. */
export function StickyBar({ children }: { children: ReactNode }) {
  return <div className="sticky-controls">{children}</div>
}

export function Arrow({ children }: { children: ReactNode }) {
  return (
    <div className="muted" style={{ margin: '6px 0 6px 4px' }}>
      {children}
    </div>
  )
}
