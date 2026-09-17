import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import { gelu } from '../../engine/kernels'
import { view } from '../../engine/params'
import { displayToken } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { fmt } from '../viz/colors'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import { MatrixMul } from '../viz/MatrixMul'
import { NetworkDiagram } from '../viz/NetworkDiagram'
import { TokenChips } from '../viz/TokenChips'
import { Arrow, Seg, StickyBar } from './controls'

function GeluCurve({ pre }: { pre: ArrayLike<number> }) {
  const w = 360
  const hgt = 200
  const xs = -4
  const xe = 4
  const ys = -0.5
  const ye = 4
  const sx = (x: number) => ((x - xs) / (xe - xs)) * (w - 20) + 10
  const sy = (y: number) => hgt - 10 - ((y - ys) / (ye - ys)) * (hgt - 20)
  let d = ''
  for (let i = 0; i <= 160; i++) {
    const x = xs + ((xe - xs) * i) / 160
    d += (i ? 'L' : 'M') + sx(x).toFixed(1) + ',' + sy(gelu(x)).toFixed(1)
  }
  return (
    <svg width={w} height={hgt} viewBox={`0 0 ${w} ${hgt}`} style={{ display: 'block', maxWidth: '100%', height: 'auto' }}>
      <line x1={sx(xs)} x2={sx(xe)} y1={sy(0)} y2={sy(0)} stroke="var(--line-2)" />
      <line x1={sx(0)} x2={sx(0)} y1={sy(ys)} y2={sy(ye)} stroke="var(--line-2)" />
      <path d={d} fill="none" stroke="var(--indigo)" strokeWidth={2} />
      {Array.from(pre as ArrayLike<number>, (x, i) => (
        <circle key={i} cx={sx(Math.max(xs, Math.min(xe, x)))} cy={sy(gelu(x))} r={4} fill="var(--accent)" opacity={0.7} stroke="var(--paper)" strokeWidth={1} />
      ))}
      <text x={sx(xe) - 4} y={sy(0) + 12} textAnchor="end">
        入力
      </text>
      <text x={sx(0) + 6} y={sy(ye) + 10}>
        GELU(入力)
      </text>
    </svg>
  )
}

export function Mlp(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, tokenizer, ids, acts } = lab
  const { dModel: D, dFF: F, nLayers: L } = params.config
  const T = ids.length
  const [l, setL] = useState(0)
  const [tSel, setT] = useState<number | null>(null)
  const [j1, setJ1] = useState(0)
  const [j2, setJ2] = useState(0)
  const t = Math.min(tSel ?? T - 1, T - 1)
  const tokens = tokenizer.tokensOf(ids)
  const disp = tokens.map(displayToken)
  const a = acts.layers[l]
  const p = `layer${l}.`
  const dimLabels = useMemo(() => Array.from({ length: D }, (_, i) => String(i)), [D])
  const x1 = a.x1.subarray(t * D, (t + 1) * D)
  const ln2 = a.ln2.subarray(t * D, (t + 1) * D)
  const fc = a.fc.subarray(t * F, (t + 1) * F)
  const ge = a.gelu.subarray(t * F, (t + 1) * F)
  const proj = a.mlpProj.subarray(t * D, (t + 1) * D)
  const x2 = a.x2.subarray(t * D, (t + 1) * D)
  const active = Array.from(ge).filter((v) => v > 0.05).length
  const geluMax = Math.max(...Array.from(fc, Math.abs))
  const scale = Math.max(...Array.from(x2, Math.abs), ...Array.from(x1, Math.abs))

  const stream = useMemo(() => {
    const items: { label: string; data: Float32Array | Float64Array }[] = [{ label: 'x₀ 埋め込み', data: acts.encoded as Float32Array }]
    acts.layers.forEach((la, i) => {
      items.push({ label: `層 ${i + 1} 注意の後`, data: la.x1 as Float32Array })
      items.push({ label: `層 ${i + 1} MLP の後`, data: la.x2 as Float32Array })
    })
    let mx = 0
    for (const it of items) for (let i = 0; i < it.data.length; i++) mx = Math.max(mx, Math.abs(it.data[i]))
    return { items, mx }
  }, [acts])

  const norm = (v: ArrayLike<number>) => Math.sqrt(Array.from(v).reduce((s, x) => s + x * x, 0))

  const controls = (
    <StickyBar>
      <div className="row" style={{ alignItems: 'center', gap: 18 }}>
        <Seg label="層" value={l} options={Array.from({ length: L }, (_, i) => ({ value: i, label: `${i + 1}` }))} onChange={setL} />
        <span className="field">トークン</span>
        <TokenChips tokens={tokens} active={t} onSelect={setT} positions />
      </div>
    </StickyBar>
  )

  const steps: Step[] = [
    {
      id: 'ln2',
      title: 'もう一度 LayerNorm',
      body: (
        <>
          <p>注意機構の出力を足した x₁ を、MLP に入れる前にもう一度正規化します。γ・β は注意機構側とは別の、この層専用のものです。</p>
        </>
      ),
      formula: `LN₂(x₁) = (x₁ − μ) / σ · γ₂ + β₂`,
    },
    {
      id: 'fc',
      title: '拡大：16 → 64 次元',
      body: (
        <>
          <p>
            <Term id="mlp">MLP（多層パーセプトロン）</Term>は<strong>位置ごとに独立</strong>に働きます。まず W₁（{D} × {F}）で {F} 次元に広げます。{F} 個の「特徴検出器」それぞれが、入力ベクトルとの内積を取っていると読めます。{F} という数は GPT-2 以来の慣例「d_model の 4 倍」で、{D} × 4 = {F} です。
          </p>
          <p>列をクリックすると、そのニューロンの計算式が出ます。</p>
        </>
      ),
      formula: `h = LN₂(x₁) · W₁ + b₁   ∈ ℝ^${F}`,
    },
    {
      id: 'gelu',
      title: 'GELU：非線形にする',
      body: (
        <>
          <p>
            行列の掛け算だけでは、何層重ねても 1 つの行列と同じ。<strong>非線形な関数</strong>を挟むことで初めて複雑な関係を表せます。GELU は負の入力をほぼ 0 に潰し、正の入力はそのまま通します。
          </p>
          <p>
            位置 {t} では {F} ニューロン中 {active} 個が明確に活性化（&gt; 0.05）しています。曲線上の点が {F} 個の値です。
          </p>
        </>
      ),
      formula: `GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`,
    },
    {
      id: 'proj',
      title: '縮小：64 → 16 次元',
      body: (
        <>
          <p>
            活性化した {F} 次元を W₂（{F} × {D}）で {D} 次元に戻します。W₂ の各行は「そのニューロンが発火したとき、残差ストリームに何を書き込むか」を表す方向ベクトルです。
          </p>
        </>
      ),
      formula: `mlp = GELU(h) · W₂ + b₂   ∈ ℝ^${D}`,
    },
    {
      id: 'residual',
      title: '残差に足してブロック完了',
      body: (
        <>
          <p>
            MLP の出力も入力に<strong>足し戻します</strong>。これで Transformer ブロック 1 つ分が終わり、次のブロックへ同じ形（{T} × {D}）のまま渡されます。
          </p>
        </>
      ),
      formula: `x₂[t] = x₁[t] + mlp[t]`,
    },
    {
      id: 'stream',
      title: '残差ストリームの変化',
      body: (
        <>
          <p>
            埋め込み x₀ から、注意・MLP が交互に情報を<strong>書き足していく</strong>様子です。同じ色スケールで並べています。層を経るほど値が大きく、模様がはっきりしていきます。
          </p>
          <p>
            位置 {t} のノルム：{stream.items.map((it) => fmt(norm(it.data.subarray(t * D, (t + 1) * D)), 2)).join(' → ')}
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="05"
      title="MLP と残差ストリーム"
      lede="注意機構が集めた情報を、位置ごとに変換して残差ストリームへ書き足す。"
      purpose={
        <>
          注意機構は「誰の情報を集めるか」を決めるだけで、集めた情報の加工はほとんどしません。<Term id="mlp">MLP</Term> は各位置のベクトルを一度 {F} 次元に広げ、<Term id="gelu">GELU</Term> で非線形にしてから {D} 次元に戻す 2 段の変換で、情報を整理し直します。「ねこ→たべる→さかな」のような知識はここに蓄えられると考えられています。結果は入力に<strong>足し戻す</strong>（<Term id="residual">残差接続</Term>）ので、各<Term id="layer">層</Term>は差分だけを学べばよく、深く積んでも情報が消えません。
        </>
      }
      io={`${T} × ${D} の行列 → ${T} × ${D} の行列（各行が位置ごとに書き換わる）`}
      terms={['mlp', 'gelu', 'residual', 'layer', 'layernorm']}
      steps={steps}
    >
      {(step) => (
        <div className="col">
          {controls}
          <p className="controls-help">
            <strong>層</strong>：注意機構 + MLP のブロックを中間層のように {L} 段積んでいます。同じ MLP が別の重みで {L} 回繰り返され、どの段を見るかの切替。<strong>トークン</strong>：MLP は位置ごとに独立なので、どの位置の計算を見るかを選びます。
          </p>
          {step.id === 'ln2' && (
            <div className="card">
              <div className="viz-title">x₁[{t}]</div>
              <VectorStrip values={x1} cell={34} colLabels={dimLabels} />
              <Arrow>↓ LayerNorm（γ₂, β₂）</Arrow>
              <VectorStrip values={ln2} cell={34} colLabels={dimLabels} />
            </div>
          )}
          {step.id === 'fc' && (
            <div className="card">
              <NetworkDiagram panel="mlp" />
              <p className="muted small" style={{ margin: '8px 0 16px' }}>
                上の図の左 2 層分が W₁ です。行列で書くと下のようになります。列 j = 中間層のノード j に入る線の重み {D} 本。
              </p>
              <MatrixMul x={ln2} W={view(params, p + 'mlp.fc.w')} C={D} OC={F} bias={view(params, p + 'mlp.fc.b')} out={fc} j={j1} onSelect={setJ1} xLabel={`LN₂(x₁[${t}])`} wLabel={`W₁（${D} 行 × ${F} 列）`} outLabel="h" cell={9} rowAxis={`入力の次元 i（${D} 個）`} colAxis={`ニューロン j（${F} 個の特徴検出器、1 列 = 1 個）`} />
            </div>
          )}
          {step.id === 'gelu' && (
            <div className="row">
              <div className="card grow">
                <div className="viz-title">h[{t}]（活性化前、{F} 次元を 16 列ずつ折り返し）</div>
                <Heatmap values={fc} rows={F / 16} cols={16} cell={26} rowLabels={Array.from({ length: F / 16 }, (_, r) => `${r * 16}–`)} rowLabelWidth={36} showValues maxAbs={geluMax} rowName="行" colName="列" />
                <Arrow>↓ GELU</Arrow>
                <Heatmap values={ge} rows={F / 16} cols={16} cell={26} rowLabels={Array.from({ length: F / 16 }, (_, r) => `${r * 16}–`)} rowLabelWidth={36} showValues maxAbs={geluMax} rowName="行" colName="列" />
              </div>
              <div className="card">
                <div className="card-title">GELU 曲線と {F} 個の値</div>
                <GeluCurve pre={fc} />
              </div>
            </div>
          )}
          {step.id === 'proj' && (
            <div className="card">
              <MatrixMul x={ge} W={view(params, p + 'mlp.proj.w')} C={F} OC={D} bias={view(params, p + 'mlp.proj.b')} out={proj} j={j2} onSelect={setJ2} xLabel={`GELU(h[${t}])`} wLabel={`W₂（${F} 行 × ${D} 列）`} outLabel="mlp" cell={9} rowAxis={`ニューロン i（${F} 個。行 i = そのニューロンが発火したとき書き込む方向）`} colAxis={`残差ストリームの次元 j（${D} 個）`} />
            </div>
          )}
          {step.id === 'residual' && (
            <div className="card">
              <div className="viz-title">x₁[{t}]</div>
              <VectorStrip values={x1} cell={34} maxAbs={scale} />
              <Arrow>＋ mlp[{t}]</Arrow>
              <VectorStrip values={proj} cell={34} maxAbs={scale} />
              <Arrow>＝ x₂[{t}]</Arrow>
              <VectorStrip values={x2} cell={34} maxAbs={scale} colLabels={dimLabels} />
            </div>
          )}
          {step.id === 'stream' && (
            <div className="col">
              <NetworkDiagram panel="stack" />
              <div className="row" style={{ gap: 28 }}>
              {stream.items.map((it) => (
                <Heatmap key={it.label} values={it.data} rows={T} cols={D} cell={14} maxAbs={stream.mx} rowLabels={disp} rowLabelWidth={56} highlightRows={[t]} title={it.label} rowName="t" colName="dim" onRowClick={setT} tag="computed" />
              ))}
              </div>
            </div>
          )}
        </div>
      )}
    </ChapterLayout>
  )
}
