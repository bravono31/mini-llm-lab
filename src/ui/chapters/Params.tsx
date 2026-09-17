import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import { totalSize, viewIn } from '../../engine/params'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmt, fmtPct } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import { SpecHeat, stats } from './SpecHeat'

export function Params(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, session } = lab
  const cfg = params.config
  const total = totalSize(params.specs)
  const [selected, setSelected] = useState('wte')
  const spec = params.byName.get(selected)!
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const st = useMemo(() => stats(viewIn(params.data, spec)), [params, spec, lab.version])
  const diff = useMemo(() => {
    const a = viewIn(params.data, spec)
    const b = viewIn(session.pretrainedParams.data, spec)
    const d = new Float32Array(spec.size)
    let m = 0
    for (let i = 0; i < spec.size; i++) {
      d[i] = a[i] - b[i]
      m = Math.max(m, Math.abs(d[i]))
    }
    return { d, m }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, spec, session, lab.version])

  const perLayer = useMemo(() => {
    const items: { label: string; value: number }[] = []
    let emb = 0
    for (const s of params.specs) if (s.layer === null && s.group === 'embedding') emb += s.size
    items.push({ label: '埋め込み', value: emb })
    for (let l = 0; l < cfg.nLayers; l++) {
      let attn = 0
      let mlp = 0
      let ln = 0
      for (const s of params.specs) {
        if (s.layer !== l) continue
        if (s.name.includes('attn')) attn += s.size
        else if (s.name.includes('mlp')) mlp += s.size
        else ln += s.size
      }
      items.push({ label: `層${l + 1} 注意`, value: attn }, { label: `層${l + 1} MLP`, value: mlp }, { label: `層${l + 1} LN`, value: ln })
    }
    items.push({ label: '最終 LN', value: 2 * cfg.dModel })
    return items
  }, [params.specs, cfg])

  const steps: Step[] = [
    {
      id: 'overview',
      title: 'どこに何個あるか',
      body: (
        <>
          <p>
            パラメータは全部で <strong>{total.toLocaleString()} 個</strong>。ひとつの Transformer ブロックには注意機構 4 行列 + MLP 2 行列 + LayerNorm 2 組が入っていて、それが {cfg.nLayers} 回繰り返されます。
          </p>
          <p>行列の大きさは「入力次元 × 出力次元」で決まるので、d_model を 2 倍にすると重み行列は 4 倍になります。</p>
        </>
      ),
      formula: `W_te: V·d = ${cfg.vocabSize}·${cfg.dModel} = ${cfg.vocabSize * cfg.dModel}\nW_qkv: d·3d = ${cfg.dModel * 3 * cfg.dModel}   W_o: d² = ${cfg.dModel ** 2}\nW₁: d·d_ff = ${cfg.dModel * cfg.dFF}   W₂: d_ff·d = ${cfg.dFF * cfg.dModel}`,
    },
    {
      id: 'table',
      title: 'すべてのテンソル',
      body: (
        <>
          <p>名前・形・個数と、値の統計です。行をクリックすると次のステップで中身を見られます。</p>
          <p>学習済みモデルでは LayerNorm の γ が 1 から、β が 0 から動き、行列は初期の N(0, 0.02²) より広がっています。</p>
        </>
      ),
    },
    {
      id: 'inspect',
      title: '中身を見る',
      body: (
        <>
          <p>
            <strong>{spec.name}</strong>（{spec.label}）の値です。右は同梱の学習済み重みとの差分で、第 7 章で学習を進めるとここが変わります。
            {session.origin === 'random' && ' いまはランダム初期化なので差分はそのまま「学習済みからの距離」です。'}
          </p>
          <p>
            平均 {fmt(st.mean, 4)} · 標準偏差 {fmt(st.std, 4)} · 最大絶対値 {fmt(st.absMax, 4)}
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="08"
      title="パラメータ一覧"
      lede="学習で決まる数値のすべて。名前と形と、いまの値。"
      purpose={
        <>
          モデルの「知識」は、ここに並ぶ数値の中にしかありません。どの行列がどこで使われ、いくつあるかを一覧で把握します。形（行 × 列）はすべて設計値（次元数・語彙数・層数）から決まり、値は<Term id="parameter">学習</Term>で決まります。
        </>
      }
      io="—"
      terms={['parameter', 'dimension', 'layer', 'embedding', 'tying']}
      steps={steps}
    >
      {(step) => (
        <div className="col">
          {step.id === 'overview' && (
            <div className="row">
              <div className="card">
                <BarChart title="内訳" items={perLayer.map((it) => ({ ...it, sub: fmtPct(it.value / total, 0) }))} width={420} labelWidth={96} valueWidth={90} format={(v) => v.toLocaleString()} />
              </div>
              <div className="card grow" style={{ minWidth: 240 }}>
                <div className="stat">
                  <span className="label">合計</span>
                  <span className="value">{total.toLocaleString()}</span>
                </div>
                <p className="muted small" style={{ marginTop: 12 }}>
                  参考：GPT-2 small は 1 億 2400 万、GPT-3 は 1750 億。構造は同じで、d_model・層数・語彙が大きいだけです。
                </p>
              </div>
            </div>
          )}
          {step.id === 'table' && (
            <div className="card" style={{ maxHeight: 640, overflow: 'auto' }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>名前</th>
                    <th>役割</th>
                    <th>形</th>
                    <th>個数</th>
                    <th>平均</th>
                    <th>標準偏差</th>
                    <th>最大|w|</th>
                  </tr>
                </thead>
                <tbody>
                  {params.specs.map((s) => {
                    const v = stats(viewIn(params.data, s))
                    return (
                      <tr key={s.name} className={s.name === selected ? 'hi' : ''} onClick={() => setSelected(s.name)} style={{ cursor: 'pointer' }}>
                        <td className="mono">{s.name}</td>
                        <td>{s.label}</td>
                        <td className="mono">[{s.shape.join(' × ')}]</td>
                        <td className="num">{s.size.toLocaleString()}</td>
                        <td className="num">{fmt(v.mean, 3)}</td>
                        <td className="num">{fmt(v.std, 3)}</td>
                        <td className="num">{fmt(v.absMax, 3)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
          {step.id === 'inspect' && (
            <>
              <span className="field">
                テンソル
                <select value={selected} onChange={(e) => setSelected(e.target.value)}>
                  {params.specs.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name} [{s.shape.join('×')}] — {s.label}
                    </option>
                  ))}
                </select>
              </span>
              <div className="row">
                <div className="card">
                  <SpecHeat data={params.data} spec={spec} title={`${spec.name}  [${spec.shape.join(' × ')}]`} tag="param" />
                </div>
                <div className="card">
                  <Heatmap values={diff.d} rows={spec.shape.length === 2 ? spec.shape[0] : 1} cols={spec.shape.length === 2 ? spec.shape[1] : spec.shape[0]} cell={Math.max(4, Math.min(14, Math.floor(560 / (spec.shape.length === 2 ? spec.shape[1] : spec.shape[0]))))} gap={1} maxAbs={diff.m || 1} title={`学習済みとの差分（最大 |Δ| ${fmt(diff.m, 4)}）`} legend rowName="i" colName="j" />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </ChapterLayout>
  )
}
