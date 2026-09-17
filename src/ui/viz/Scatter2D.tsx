import { useState } from 'react'

interface Props {
  points: [number, number][]
  labels: string[]
  highlight?: Set<number>
  width?: number
  height?: number
  title?: string
}

export function Scatter2D({ points, labels, highlight, width = 560, height = 400, title }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const [x, y] of points) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const pad = 28
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (width - 2 * pad)
  const sy = (y: number) => height - pad - ((y - minY) / (maxY - minY || 1)) * (height - 2 * pad)
  return (
    <div className="viz-wrap">
      {title && <div className="viz-title">{title}</div>}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
        <line x1={sx(0)} x2={sx(0)} y1={pad} y2={height - pad} stroke="var(--line)" />
        <line x1={pad} x2={width - pad} y1={sy(0)} y2={sy(0)} stroke="var(--line)" />
        {points.map(([x, y], i) => {
          const hi = highlight?.has(i)
          const hv = hover === i
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <circle cx={sx(x)} cy={sy(y)} r={hi ? 6 : 4} fill={hi ? 'var(--accent)' : 'var(--indigo)'} opacity={hi || hv ? 1 : 0.55} stroke="var(--paper)" strokeWidth={1.5} />
              <text x={sx(x) + 8} y={sy(y) + 3.5} fontSize={hi || hv ? 12 : 9} className={hi || hv ? 'lbl-strong' : ''} opacity={hi || hv ? 1 : 0.7}>
                {labels[i]}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
