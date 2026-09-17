import { useMemo, useState } from 'react'
import { fmt } from './colors'

interface Props {
  history: ArrayLike<number>
  width?: number
  height?: number
  baseline?: number
  baselineLabel?: string
  smooth?: number
  title?: string
  /** x positions to mark (e.g. where the pretrained run ends) */
  marks?: { x: number; label: string }[]
}

export function LossChart({ history, width = 620, height = 220, baseline, baselineLabel, smooth = 25, title, marks }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const n = history.length
  const padL = 44
  const padR = 12
  const padT = 12
  const padB = 26
  const plotW = width - padL - padR
  const plotH = height - padT - padB

  const { yMax, smoothed } = useMemo(() => {
    let mx = baseline ?? 0
    const sm = new Float64Array(n)
    let acc = 0
    for (let i = 0; i < n; i++) {
      const v = history[i]
      if (v > mx) mx = v
      acc += v
      if (i >= smooth) acc -= history[i - smooth]
      sm[i] = acc / Math.min(i + 1, smooth)
    }
    return { yMax: mx * 1.06 || 1, smoothed: sm }
  }, [history, n, smooth, baseline])

  const x = (i: number) => padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW)
  const y = (v: number) => padT + plotH - (v / yMax) * plotH
  const path = (arr: ArrayLike<number>) => {
    let d = ''
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(arr[i]).toFixed(1)
    return d
  }
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax)
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * Math.max(0, n - 1)))

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (n === 0) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * width
    const i = Math.round(((px - padL) / plotW) * (n - 1))
    setHover(Math.max(0, Math.min(n - 1, i)))
  }

  return (
    <div className="viz-wrap" style={{ width: '100%' }}>
      {title && <div className="viz-title">{title}</div>}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={padL} x2={width - padR} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={1} />
            <text x={padL - 8} y={y(v) + 3.5} textAnchor="end" fontSize={10}>
              {fmt(v, 1)}
            </text>
          </g>
        ))}
        {xTicks.map((i, k) => (
          <text key={k} x={x(i)} y={height - 8} textAnchor="middle" fontSize={10}>
            {i}
          </text>
        ))}
        {baseline !== undefined && (
          <g>
            <line x1={padL} x2={width - padR} y1={y(baseline)} y2={y(baseline)} stroke="var(--ink-3)" strokeDasharray="4 4" strokeWidth={1} />
            <text x={width - padR} y={y(baseline) - 5} textAnchor="end" fontSize={10}>
              {baselineLabel ?? `ln V = ${fmt(baseline, 2)}`}
            </text>
          </g>
        )}
        {marks?.map((m, i) => (
          <g key={i}>
            <line x1={x(m.x)} x2={x(m.x)} y1={padT} y2={padT + plotH} stroke="var(--ink-3)" strokeDasharray="2 4" />
            <text x={x(m.x) + 4} y={padT + 10} fontSize={10}>
              {m.label}
            </text>
          </g>
        ))}
        {n > 0 && <path d={path(history)} fill="none" stroke="var(--indigo)" strokeWidth={1} opacity={0.3} />}
        {n > 0 && <path d={path(smoothed)} fill="none" stroke="var(--indigo)" strokeWidth={2} strokeLinejoin="round" />}
        {hover !== null && n > 0 && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={padT} y2={padT + plotH} stroke="var(--ink-3)" strokeWidth={1} />
            <circle cx={x(hover)} cy={y(smoothed[hover])} r={4} fill="var(--indigo)" stroke="var(--paper)" strokeWidth={2} />
          </g>
        )}
      </svg>
      {hover !== null && n > 0 && (
        <div className="tooltip" style={{ left: `${(x(hover) / width) * 100}%`, top: `${(y(smoothed[hover]) / height) * 100}%` }}>
          step {hover + 1} · loss {fmt(history[hover], 3)} · 平均 {fmt(smoothed[hover], 3)}
        </div>
      )}
    </div>
  )
}
