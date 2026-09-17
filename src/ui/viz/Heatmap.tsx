import { useMemo, useState } from 'react'
import { fmt, useHeatPalette } from './colors'
import { Tag, type ValueKind } from './Tag'

export interface ColGroup {
  label: string
  from: number
  to: number
  color: string
}

export interface HeatmapProps {
  values: ArrayLike<number>
  rows: number
  cols: number
  mode?: 'diverging' | 'sequential'
  cell?: number
  gap?: number
  rowLabels?: string[]
  colLabels?: string[]
  rowLabelWidth?: number
  highlightRows?: number[]
  highlightCols?: number[]
  highlightCells?: [number, number][]
  /** dim every cell outside highlighted rows / cols */
  dimOthers?: boolean
  masked?: (r: number, c: number) => boolean
  maxAbs?: number
  max?: number
  title?: string
  showValues?: boolean
  digits?: number
  onCellClick?: (r: number, c: number) => void
  onRowClick?: (r: number) => void
  legend?: boolean
  /** name shown in tooltips for the row axis / col axis */
  rowName?: string
  colName?: string
  /** human explanation of what a row / a column means, shown under the title */
  rowAxis?: string
  colAxis?: string
  /** coloured bands above column ranges (e.g. which columns are Q / K / V) */
  colGroups?: ColGroup[]
  /** where the numbers come from */
  tag?: ValueKind
}

export function Heatmap(p: HeatmapProps) {
  const pal = useHeatPalette()
  const { rows, cols, values } = p
  const mode = p.mode ?? 'diverging'
  const cell = p.cell ?? 14
  const gap = p.gap ?? 1
  const digits = p.digits ?? 2
  const rlw = p.rowLabelWidth ?? (p.rowLabels ? 52 : 0)
  const longCols = !!p.colLabels && p.colLabels.some((l) => l.length > 2)
  const groupH = p.colGroups ? 16 : 0
  const clh = (p.colLabels ? (longCols ? 44 : 14) : 0) + groupH
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null)

  const { maxAbs, max } = useMemo(() => {
    let ma = 0
    let mx = -Infinity
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (!Number.isFinite(v)) continue
      if (Math.abs(v) > ma) ma = Math.abs(v)
      if (v > mx) mx = v
    }
    return { maxAbs: p.maxAbs ?? (ma || 1), max: p.max ?? (mx > 0 ? mx : 1) }
  }, [values, p.maxAbs, p.max])

  const hr = useMemo(() => new Set(p.highlightRows ?? []), [p.highlightRows])
  const hc = useMemo(() => new Set(p.highlightCols ?? []), [p.highlightCols])
  const hcell = useMemo(() => new Set((p.highlightCells ?? []).map(([r, c]) => r * cols + c)), [p.highlightCells, cols])
  const anyHighlight = hr.size > 0 || hc.size > 0 || hcell.size > 0

  const width = rlw + cols * (cell + gap) + 2
  const height = clh + rows * (cell + gap) + 2
  const step = cell + gap
  const fontSize = Math.min(10, Math.max(7, cell * 0.42))

  const rects = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r * cols + c]
      const masked = p.masked?.(r, c) ?? false
      const t = mode === 'diverging' ? v / maxAbs : v / max
      const fill = masked ? 'var(--paper-3)' : mode === 'diverging' ? pal.div(t) : pal.seq(t)
      const strong = hcell.has(r * cols + c) || (hr.has(r) && hc.has(c))
      const inRowOrCol = hr.has(r) || hc.has(c)
      const dim = p.dimOthers && anyHighlight && !strong && !inRowOrCol
      const x = rlw + c * step + 1
      const y = clh + r * step + 1
      rects.push(
        <g key={r * cols + c} opacity={dim ? 0.22 : 1}>
          <rect
            x={x}
            y={y}
            width={cell}
            height={cell}
            fill={fill}
            rx={cell >= 12 ? 1.5 : 0}
            stroke={strong ? 'var(--accent)' : hover && hover.r === r && hover.c === c ? 'var(--ink)' : 'none'}
            strokeWidth={strong ? 1.5 : 1}
            onMouseEnter={() => setHover({ r, c })}
            onMouseLeave={() => setHover(null)}
            onClick={() => (p.onCellClick ? p.onCellClick(r, c) : p.onRowClick?.(r))}
            style={{ cursor: p.onCellClick || p.onRowClick ? 'pointer' : 'default', transition: 'fill 200ms' }}
          />
          {p.showValues && cell >= 18 && !masked && (
            <text x={x + cell / 2} y={y + cell / 2 + fontSize * 0.36} textAnchor="middle" fontSize={fontSize} fill={Math.abs(t) > 0.55 ? 'var(--paper)' : 'var(--ink)'} pointerEvents="none">
              {fmt(v, digits)}
            </text>
          )}
        </g>,
      )
    }
  }

  const tip = hover
    ? (() => {
        const rl = p.rowLabels?.[hover.r] ?? String(hover.r)
        const cl = p.colLabels?.[hover.c] ?? String(hover.c)
        const v = values[hover.r * cols + hover.c]
        const rn = p.rowName ? `${p.rowName} ` : ''
        const cn = p.colName ? `${p.colName} ` : ''
        return `${rn}${rl} · ${cn}${cl} = ${fmt(v, 4)}`
      })()
    : null

  return (
    <div className="viz-wrap">
      {(p.title || p.legend || p.tag) && (
        <div className="viz-title">
          <span>
            {p.title}
            {p.tag && <Tag kind={p.tag} />}
          </span>
          {p.legend && (
            <span className="legend-bar">
              <span>{mode === 'diverging' ? fmt(-maxAbs, 2) : '0'}</span>
              <i style={{ background: mode === 'diverging' ? `linear-gradient(90deg, ${pal.div(-1)}, ${pal.div(0)}, ${pal.div(1)})` : `linear-gradient(90deg, ${pal.seq(0)}, ${pal.seq(1)})` }} />
              <span>{fmt(mode === 'diverging' ? maxAbs : max, 2)}</span>
            </span>
          )}
        </div>
      )}
      {(p.rowAxis || p.colAxis) && (
        <div className="axis-note">
          {p.rowAxis && <span>行 = {p.rowAxis}</span>}
          {p.colAxis && <span>列 = {p.colAxis}</span>}
        </div>
      )}
      <svg width={width} height={height} style={{ display: 'block', maxWidth: '100%', height: 'auto' }} viewBox={`0 0 ${width} ${height}`}>
        {p.colGroups?.map((g) => (
          <g key={g.label}>
            <rect x={rlw + g.from * step + 1} y={1} width={(g.to - g.from) * step - gap} height={5} rx={2} fill={g.color} />
            <text x={rlw + ((g.from + g.to) / 2) * step} y={13} textAnchor="middle" fontSize={9} fill={g.color} fontWeight={600}>
              {g.label}
            </text>
          </g>
        ))}
        {p.colLabels?.map((l, c) =>
          longCols ? (
            <text key={c} transform={`translate(${rlw + c * step + cell / 2 + 3}, ${clh - 4}) rotate(-50)`} fontSize={9} className={hc.has(c) ? 'lbl-strong' : ''}>
              {l}
            </text>
          ) : (
            <text key={c} x={rlw + c * step + cell / 2} y={clh - 4} textAnchor="middle" fontSize={8} className={hc.has(c) ? 'lbl-strong' : ''}>
              {l}
            </text>
          ),
        )}
        {p.rowLabels?.map((l, r) => (
          <text
            key={r}
            x={rlw - 6}
            y={clh + r * step + cell / 2 + 3.5}
            textAnchor="end"
            fontSize={Math.min(11, Math.max(8, cell * 0.7))}
            className={hr.has(r) ? 'lbl-strong' : ''}
            style={{ cursor: p.onRowClick ? 'pointer' : 'default' }}
            onClick={() => p.onRowClick?.(r)}
          >
            {l}
          </text>
        ))}
        {rects}
        {[...hr].map((r) => (
          <rect key={'r' + r} x={rlw + 0.5} y={clh + r * step + 0.5} width={cols * step - gap + 1} height={cell + 1} fill="none" stroke="var(--accent)" strokeWidth={1.5} pointerEvents="none" />
        ))}
        {[...hc].map((c) => (
          <rect key={'c' + c} x={rlw + c * step + 0.5} y={clh + 0.5} width={cell + 1} height={rows * step - gap + 1} fill="none" stroke="var(--accent)" strokeWidth={1.5} pointerEvents="none" />
        ))}
      </svg>
      {hover && tip && (
        <div className="tooltip" style={{ left: rlw + hover.c * step + cell / 2 + 1, top: clh + hover.r * step + 1 }}>
          {tip}
        </div>
      )}
    </div>
  )
}

/** A single row vector drawn as a strip of cells. */
export function VectorStrip(p: Omit<HeatmapProps, 'rows' | 'cols'> & { length?: number; highlight?: number[] }) {
  const n = p.length ?? p.values.length
  return <Heatmap {...p} rows={1} cols={n} highlightCols={p.highlight} showValues={p.showValues ?? true} cell={p.cell ?? 30} />
}
