import type { ReactElement } from 'react'
import { useLab } from '../state/LabProvider'

interface Props {
  panel?: 'mlp' | 'stack' | 'both'
}

/** Nodes-and-edges pictures: the MLP as a classic 3-layer network, and blocks stacked like hidden layers. */
export function NetworkDiagram({ panel = 'both' }: Props) {
  const { params, ids } = useLab()
  const { dModel: D, dFF: F, nLayers: L, nHeads: H, vocabSize: V } = params.config
  const T = ids.length
  return (
    <div className="row" style={{ gap: 28, alignItems: 'flex-start' }}>
      {(panel === 'mlp' || panel === 'both') && <MlpNet D={D} F={F} />}
      {(panel === 'stack' || panel === 'both') && <Stack L={L} H={H} D={D} T={T} V={V} />}
    </div>
  )
}

function MlpNet({ D, F }: { D: number; F: number }) {
  const w = 420
  const h = 300
  const cols = [
    { x: 60, n: 7, label: `入力 ${D} 次元`, sub: 'LN₂(x₁) のベクトル', color: 'var(--indigo)' },
    { x: 210, n: 10, label: `中間層 ${F} 個`, sub: 'GELU で発火 / 沈黙', color: 'var(--accent)' },
    { x: 360, n: 7, label: `出力 ${D} 次元`, sub: '残差に足し戻す', color: 'var(--indigo)' },
  ]
  const ys = (n: number) => Array.from({ length: n }, (_, i) => 50 + (i * (h - 130)) / (n - 1))
  const edges: ReactElement[] = []
  for (let c = 0; c < cols.length - 1; c++) {
    const a = ys(cols[c].n)
    const b = ys(cols[c + 1].n)
    a.forEach((y1, i) =>
      b.forEach((y2, j) => {
        edges.push(<line key={`${c}-${i}-${j}`} x1={cols[c].x} y1={y1} x2={cols[c + 1].x} y2={y2} stroke="var(--line-2)" strokeWidth={0.6} opacity={0.7} />)
      }),
    )
  }
  return (
    <div className="viz-wrap">
      <div className="viz-title">MLP そのものが、教科書どおりの 3 層ニューラルネットワーク</div>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
        {edges}
        {cols.map((c, ci) => (
          <g key={ci}>
            {ys(c.n).map((y, i) => (
              <circle key={i} cx={c.x} cy={y} r={7} fill={c.color} stroke="var(--paper)" strokeWidth={1.5} opacity={i === Math.floor(c.n / 2) ? 0.35 : 1} />
            ))}
            <text x={c.x} y={50 + (Math.floor(c.n / 2) * (h - 130)) / (c.n - 1) + 4} textAnchor="middle" fontSize={11} className="lbl-strong">
              ⋮
            </text>
            <text x={c.x} y={h - 52} textAnchor="middle" fontSize={12} className="lbl-strong">
              {c.label}
            </text>
            <text x={c.x} y={h - 36} textAnchor="middle" fontSize={10}>
              {c.sub}
            </text>
          </g>
        ))}
        <text x={135} y={22} textAnchor="middle" fontSize={10}>
          W₁：線 1 本 = 重み 1 個（{D}×{F} = {D * F} 本）
        </text>
        <text x={285} y={22} textAnchor="middle" fontSize={10}>
          W₂：{F}×{D} = {F * D} 本
        </text>
        <text x={w / 2} y={h - 10} textAnchor="middle" fontSize={10}>
          ノード = 数値 1 つ、線 = 掛ける重み。各ノードは前の層の全ノードから重み付きで集めた和（+ バイアス）
        </text>
      </svg>
    </div>
  )
}

function Stack({ L, H, D, T, V }: { L: number; H: number; D: number; T: number; V: number }) {
  const w = 440
  const boxH = 30
  const gap = 10
  const blockPad = 12
  const items: { kind: 'plain' | 'block'; label: string; sub?: string }[] = [
    { kind: 'plain', label: '入力文 → トークン ID', sub: `${T} 個の整数` },
    { kind: 'plain', label: '埋め込み + 位置', sub: `${T} × ${D} の行列（残差ストリームの出発点）` },
  ]
  for (let l = 0; l < L; l++) items.push({ kind: 'block', label: `ブロック ${l + 1}`, sub: `= 一般のニューラルネットの「中間層 ${l + 1}」に相当` })
  items.push({ kind: 'plain', label: '最終 LayerNorm → 語彙との内積', sub: `${D} 次元 → ${V} 個のスコア` })
  items.push({ kind: 'plain', label: 'softmax → 次トークンの確率', sub: `${V} 個、合計 1` })

  let y = 8
  const nodes: ReactElement[] = []
  const x0 = 60
  const bw = w - 100
  items.forEach((it, i) => {
    if (it.kind === 'plain') {
      nodes.push(
        <g key={i}>
          <rect x={x0} y={y} width={bw} height={boxH} rx={6} fill="var(--paper-2)" stroke="var(--line-2)" />
          <text x={x0 + 12} y={y + 19} fontSize={12} className="lbl-strong">
            {it.label}
          </text>
          <text x={x0 + bw - 10} y={y + 19} fontSize={10} textAnchor="end">
            {it.sub}
          </text>
        </g>,
      )
      y += boxH + gap
    } else {
      const innerH = boxH * 2 + gap + blockPad * 2 + 18
      nodes.push(
        <g key={i}>
          <rect x={x0 - 14} y={y} width={bw + 28} height={innerH} rx={10} fill="none" stroke="var(--accent)" strokeDasharray="4 3" />
          <text x={x0 - 4} y={y + 14} fontSize={11} fill="var(--accent)" fontWeight={600}>
            {it.label}
          </text>
          <text x={x0 + bw + 10} y={y + 14} fontSize={10} textAnchor="end">
            {it.sub}
          </text>
          <rect x={x0} y={y + 20 + blockPad} width={bw} height={boxH} rx={6} fill="var(--accent-soft)" stroke="var(--accent)" />
          <text x={x0 + 12} y={y + 20 + blockPad + 19} fontSize={12} className="lbl-strong">
            注意機構（{H} ヘッド）
          </text>
          <text x={x0 + bw - 10} y={y + 20 + blockPad + 19} fontSize={10} textAnchor="end">
            他の位置から情報を集める
          </text>
          <rect x={x0} y={y + 20 + blockPad + boxH + gap} width={bw} height={boxH} rx={6} fill="var(--indigo-soft)" stroke="var(--indigo)" />
          <text x={x0 + 12} y={y + 20 + blockPad + boxH + gap + 19} fontSize={12} className="lbl-strong">
            MLP（{D} → {D * 4} → {D}）
          </text>
          <text x={x0 + bw - 10} y={y + 20 + blockPad + boxH + gap + 19} fontSize={10} textAnchor="end">
            位置ごとに変換して書き足す
          </text>
          {/* residual skip arrows */}
          {[0, 1].map((k) => {
            const top = y + 20 + blockPad + k * (boxH + gap)
            return (
              <path key={k} d={`M${x0 - 6},${top - 4} C${x0 - 34},${top - 4} ${x0 - 34},${top + boxH + 4} ${x0 - 6},${top + boxH + 4}`} fill="none" stroke="var(--ink-3)" strokeWidth={1.2} markerEnd="url(#arr)" />
            )
          })}
          <text x={x0 - 40} y={y + 20 + blockPad + boxH + 4} fontSize={9} textAnchor="middle" transform={`rotate(-90 ${x0 - 40} ${y + 20 + blockPad + boxH + 4})`}>
            残差（足し戻す）
          </text>
        </g>,
      )
      y += innerH + gap
    }
  })
  const h = y + 4
  // vertical spine
  return (
    <div className="viz-wrap">
      <div className="viz-title">ブロックを中間層のように積む（入力から出力への 1 本の流れ）</div>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
          </marker>
        </defs>
        <line x1={x0 + bw / 2} y1={8} x2={x0 + bw / 2} y2={h - 8} stroke="var(--line-2)" strokeWidth={1} />
        {nodes}
      </svg>
      <div className="muted small" style={{ marginTop: 6 }}>
        各ブロックの出力（{T} × {D}）が次のブロックの入力になります。一般のニューラルネットと違うのは、各ノードがベクトル（行列の 1 行）で、注意機構が行どうしをつなぐ点です。
      </div>
    </div>
  )
}
