import { fmt } from './colors'
import { Heatmap, VectorStrip, type ColGroup } from './Heatmap'

interface Props {
  x: ArrayLike<number>
  W: ArrayLike<number>
  C: number
  OC: number
  bias?: ArrayLike<number>
  out: ArrayLike<number>
  j: number
  onSelect: (j: number) => void
  xLabel: string
  wLabel: string
  outLabel: string
  colGroups?: ColGroup[]
  cell?: number
  /** what an input dimension i / output column j means */
  rowAxis?: string
  colAxis?: string
}

/** x (1×C) · W (C×OC) + b = out (1×OC), with output column j spelled out. */
export function MatrixMul({ x, W, C, OC, bias, out, j, onSelect, xLabel, wLabel, outLabel, colGroups, cell = 11, rowAxis, colAxis }: Props) {
  const terms: string[] = []
  let sum = 0
  for (let i = 0; i < C; i++) {
    const t = x[i] * W[i * OC + j]
    sum += t
    if (i < 3) terms.push(`${fmt(x[i])}×${fmt(W[i * OC + j])}`)
  }
  const b = bias ? bias[j] : 0
  const colLabels = Array.from({ length: OC }, (_, c) => String(c))
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row" style={{ alignItems: 'center', gap: 14 }}>
        <div>
          <VectorStrip values={x} cell={26} digits={1} showValues={false} title={xLabel} tag="computed" colLabels={Array.from({ length: C }, (_, i) => String(i))} colAxis={rowAxis ?? '入力の次元 i'} />
        </div>
        <div className="muted" style={{ fontSize: 20 }}>
          ×
        </div>
        <div>
          <Heatmap values={W} rows={C} cols={OC} cell={cell} title={wLabel} tag="param" legend highlightCols={[j]} colLabels={OC <= 64 ? colLabels : undefined} colGroups={colGroups} onCellClick={(_, c) => onSelect(c)} rowName="i" colName="j" rowAxis={rowAxis ?? '入力の次元 i（x の何番目に掛かるか）'} colAxis={colAxis ?? '出力の次元 j（どの出力を作るか）'} />
          <div className="muted small" style={{ marginTop: 6 }}>
            セルの色 = 重みの値（藍が負、朱が正）。{colGroups ? '上の帯 = その列が作る出力の種類。' : ''}列をクリックすると下の式が変わります。
          </div>
        </div>
      </div>
      <div className="row" style={{ alignItems: 'center', gap: 14 }}>
        <div className="muted" style={{ fontSize: 20 }}>
          =
        </div>
        <div>
          <VectorStrip values={out} cell={Math.max(cell, 14)} showValues={false} title={outLabel} tag="computed" highlight={[j]} onCellClick={(_, c) => onSelect(c)} colLabels={colLabels} colGroups={colGroups} colAxis={colAxis ?? '出力の次元 j'} />
        </div>
      </div>
      <div className="formula">
        {outLabel}[{j}] = Σᵢ x[i]·W[i,{j}]{bias ? ` + b[${j}]` : ''}
        {'\n'}= {terms.join(' + ')} + …{bias ? ` + ${fmt(b)}` : ''}
        {'\n'}= {fmt(sum + b, 4)}
      </div>
    </div>
  )
}
