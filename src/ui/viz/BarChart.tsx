import { useState } from 'react'
import { fmt } from './colors'

export interface BarItem {
  label: string
  value: number
  sub?: string
  highlight?: boolean
  muted?: boolean
  color?: string
}

interface Props {
  items: BarItem[]
  min?: number
  max?: number
  width?: number
  barHeight?: number
  gap?: number
  labelWidth?: number
  valueWidth?: number
  format?: (v: number) => string
  title?: string
  onClick?: (i: number) => void
}

export function BarChart({ items, width = 380, barHeight = 16, gap = 6, labelWidth = 76, valueWidth = 60, format = (v) => fmt(v, 2), title, onClick, ...p }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  let lo = 0
  let hi = 0
  for (const it of items) {
    if (it.value < lo) lo = it.value
    if (it.value > hi) hi = it.value
  }
  lo = p.min ?? lo
  hi = p.max ?? hi
  if (hi === lo) hi = lo + 1
  const plotW = width - labelWidth - valueWidth
  const x = (v: number) => labelWidth + ((v - lo) / (hi - lo)) * plotW
  const x0 = x(0)
  const height = items.length * (barHeight + gap) + 4
  const r = 4
  return (
    <div className="viz-wrap">
      {title && <div className="viz-title">{title}</div>}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
        <line x1={x0} x2={x0} y1={0} y2={height} stroke="var(--line-2)" strokeWidth={1} />
        {items.map((it, i) => {
          const y = i * (barHeight + gap) + 2
          const x1 = x(it.value)
          const pos = it.value >= 0
          const w = Math.abs(x1 - x0)
          const rr = Math.min(r, w)
          const d = pos
            ? `M${x0},${y} H${x1 - rr} Q${x1},${y} ${x1},${y + rr} V${y + barHeight - rr} Q${x1},${y + barHeight} ${x1 - rr},${y + barHeight} H${x0} Z`
            : `M${x0},${y} H${x1 + rr} Q${x1},${y} ${x1},${y + rr} V${y + barHeight - rr} Q${x1},${y + barHeight} ${x1 + rr},${y + barHeight} H${x0} Z`
          const fill = it.color ?? (it.highlight ? 'var(--accent)' : it.muted ? 'var(--line-2)' : 'var(--indigo)')
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => onClick?.(i)} style={{ cursor: onClick ? 'pointer' : 'default' }}>
              <rect x={0} y={y - gap / 2} width={width} height={barHeight + gap} fill="transparent" />
              <text x={labelWidth - 8} y={y + barHeight / 2 + 3.5} textAnchor="end" fontSize={11} className={it.highlight ? 'lbl-strong' : ''}>
                {it.label}
              </text>
              <path d={d} fill={fill} opacity={hover === i ? 0.85 : 1} style={{ transition: 'd 200ms' }} />
              <text x={pos ? x1 + 6 : x1 - 6} y={y + barHeight / 2 + 3.5} textAnchor={pos ? 'start' : 'end'} fontSize={10} className={it.highlight ? 'lbl-strong' : ''}>
                {format(it.value)}
                {it.sub ? `  ${it.sub}` : ''}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
