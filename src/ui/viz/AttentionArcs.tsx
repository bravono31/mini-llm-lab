import { displayToken } from '../../engine/tokenizer'
import { fmt } from './colors'

interface Props {
  tokens: string[]
  /** attention weights of the query over every key position (length T) */
  weights: ArrayLike<number>
  query: number
  onSelect?: (t: number) => void
  width?: number
}

export function AttentionArcs({ tokens, weights, query, onSelect, width = 640 }: Props) {
  const T = tokens.length
  const slot = Math.min(72, Math.max(40, (width - 24) / T))
  const w = slot * T + 24
  const baseY = 118
  const boxH = 30
  const cx = (i: number) => 12 + slot * i + slot / 2
  return (
    <div className="viz-wrap">
      <svg width={w} height={baseY + boxH + 6} viewBox={`0 0 ${w} ${baseY + boxH + 6}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
        {Array.from({ length: T }, (_, k) => {
          if (k > query) return null
          const a = weights[k]
          const x1 = cx(k)
          const x2 = cx(query)
          const lift = Math.min(100, 18 + Math.abs(x2 - x1) * 0.45)
          const d = k === query ? `M${x1},${baseY} C${x1 - 22},${baseY - 46} ${x1 + 22},${baseY - 46} ${x1},${baseY}` : `M${x1},${baseY} Q${(x1 + x2) / 2},${baseY - lift} ${x2},${baseY}`
          return (
            <g key={k}>
              <path d={d} fill="none" stroke="var(--accent)" strokeWidth={1 + a * 7} opacity={0.18 + a * 0.82} strokeLinecap="round" style={{ transition: 'all 300ms' }} />
              {a >= 0.08 && (
                <text x={k === query ? x1 : (x1 + x2) / 2} y={(k === query ? baseY - 36 : baseY - lift * 0.75) - 4} textAnchor="middle" fontSize={10} className="lbl-strong">
                  {fmt(a, 2)}
                </text>
              )}
            </g>
          )
        })}
        {tokens.map((t, i) => {
          const isQ = i === query
          const future = i > query
          return (
            <g key={i} onClick={() => onSelect?.(i)} style={{ cursor: onSelect ? 'pointer' : 'default' }} opacity={future ? 0.3 : 1}>
              <rect x={cx(i) - slot / 2 + 3} y={baseY} width={slot - 6} height={boxH} rx={5} fill={isQ ? 'var(--accent-soft)' : 'var(--paper-2)'} stroke={isQ ? 'var(--accent)' : 'var(--line-2)'} />
              <text x={cx(i)} y={baseY + boxH / 2 + 4} textAnchor="middle" fontSize={12} className="lbl-strong">
                {displayToken(t)}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
