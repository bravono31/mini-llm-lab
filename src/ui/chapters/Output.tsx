import { useEffect, useMemo, useRef, useState } from 'react'
import { Term } from '../../content/glossary'
import { forward } from '../../engine/forward'
import { contextWindow, lastLogits, makeDistribution, sampleIndex, type SampleMode } from '../../engine/generate'
import { view } from '../../engine/params'
import { mulberry32 } from '../../engine/rng'
import { displayToken, EOS_ID } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmt, fmtPct } from '../viz/colors'
import { VectorStrip } from '../viz/Heatmap'
import { TokenChips } from '../viz/TokenChips'
import { Arrow, Seg } from './controls'

interface Drawn {
  chosen: number
  prob: number
}

export function Output(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, tokenizer, ids, acts } = lab
  const { vocabSize: V, dModel: D, nLayers: L, ctxLen } = params.config
  const T = ids.length
  const [temperature, setTemperature] = useState(1)
  const [mode, setMode] = useState<SampleMode>('greedy')
  const [k, setK] = useState(5)
  const [drawn, setDrawn] = useState<Drawn[]>([])
  const seed = useRef(11)
  useEffect(() => setDrawn([]), [ids, lab.version, lab.lang])

  const tokens = tokenizer.tokensOf(ids)
  const disp = tokens.map(displayToken)
  const vocabDisp = useMemo(() => tokenizer.vocab.map(displayToken), [tokenizer])
  const dimLabels = useMemo(() => Array.from({ length: D }, (_, i) => String(i)), [D])
  const xLast = acts.layers[L - 1].x2.subarray((T - 1) * D, T * D)
  const lnf = acts.lnf.subarray((T - 1) * D, T * D)
  const logits = acts.logits.subarray((T - 1) * V, T * V)
  const wte = view(params, 'wte')

  const order = useMemo(() => [...Array(V).keys()].sort((a, b) => logits[b] - logits[a]), [logits, V])
  const top = order[0]
  let dot = 0
  for (let i = 0; i < D; i++) dot += lnf[i] * wte[top * D + i]
  const dist = useMemo(() => makeDistribution(logits, { temperature, mode, k }), [logits, temperature, mode, k])
  const base = useMemo(() => makeDistribution(logits, { temperature: 1, mode: 'random', k }), [logits, k])

  // generation state: context = ids + drawn
  const allIds = useMemo(() => [...ids, ...drawn.map((d) => d.chosen)], [ids, drawn])
  const cur = useMemo(() => {
    const ctx = contextWindow(allIds, ctxLen)
    const a = forward(params, ctx, 1, ctx.length)
    const lg = lastLogits(a)
    return { ctx, dist: makeDistribution(lg, { temperature, mode, k }) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allIds, params, ctxLen, temperature, mode, k, lab.version])
  const ended = drawn.length > 0 && drawn[drawn.length - 1].chosen === EOS_ID

  const drawOne = () => {
    const rng = mulberry32(seed.current++)
    const chosen = sampleIndex(cur.dist.sampling, rng)
    setDrawn([...drawn, { chosen, prob: cur.dist.probs[chosen] }])
  }
  const drawUntilEnd = () => {
    const out = [...drawn]
    let ctxIds = [...allIds]
    for (let i = 0; i < 14 && !(out.length && out[out.length - 1].chosen === EOS_ID); i++) {
      const ctx = contextWindow(ctxIds, ctxLen)
      const a = forward(params, ctx, 1, ctx.length)
      const d = makeDistribution(lastLogits(a), { temperature, mode, k })
      const chosen = sampleIndex(d.sampling, mulberry32(seed.current++))
      out.push({ chosen, prob: d.probs[chosen] })
      ctxIds = [...ctxIds, chosen]
    }
    setDrawn(out)
  }

  const bars = (d: { probs: Float64Array; kept: boolean[] }, n = 10, useKept = false) => {
    const ord = [...Array(V).keys()].sort((a, b) => d.probs[b] - d.probs[a]).slice(0, n)
    return ord.map((v, i) => ({ label: vocabDisp[v], value: d.probs[v], highlight: i === 0 && (!useKept || d.kept[v]), muted: useKept && !d.kept[v] }))
  }

  const samplingCtl = (
    <div className="row" style={{ alignItems: 'center', gap: 18 }}>
      <span className="field">
        温度 T
        <input type="range" min={0.2} max={3} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
        <span className="mono">{fmt(temperature, 1)}</span>
      </span>
      <Seg
        label="方式"
        value={mode}
        options={[
          { value: 'greedy', label: '最大 (greedy)' },
          { value: 'topk', label: 'top-k' },
          { value: 'random', label: '確率どおり' },
        ]}
        onChange={setMode}
      />
      {mode === 'topk' && (
        <span className="field">
          k
          <input type="range" min={1} max={10} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ width: 90 }} />
          <span className="mono">{k}</span>
        </span>
      )}
    </div>
  )

  const steps: Step[] = [
    {
      id: 'lnf',
      title: '最後の位置だけを見る',
      body: (
        <>
          <p>
            {L} ブロックを抜けた残差ストリームのうち、<strong>最後の位置 t = {T - 1}</strong>（{disp[T - 1]}）のベクトルが「次のトークン」を決めます。最終 LayerNorm で整えます。
          </p>
          <p>ほかの位置も同時に「その次」を予測していますが（最後のステップで見ます）、生成に使うのはこの 1 行です。</p>
        </>
      ),
      formula: `y = LN_f(x_${L}[${T - 1}])`,
    },
    {
      id: 'logits',
      title: 'ロジット：語彙との内積',
      body: (
        <>
          <p>
            y と<strong>埋め込み表の各行</strong>との内積を取り、語彙 {V} 個ぶんのスコア（<Term id="logits">ロジット</Term>）を出します。出力層の重みは入力の埋め込み表と共有しています（<Term id="tying">weight tying</Term>）。専用の {V} × {D} 行列を持つ設計もありますが、「埋め込みに近いベクトルほど高得点」という自然な意味になり、パラメータも減るため GPT-2 ではこの方式です。
          </p>
          <p>
            最大は「{vocabDisp[top]}」で、y · W<sub>te</sub>[{top}] = {fmt(dot, 3)}。上位 10 件を表示しています。
          </p>
        </>
      ),
      formula: `logit[v] = y · W_te[v]   (v = 0 … ${V - 1})`,
    },
    {
      id: 'softmax',
      title: 'softmax で確率に',
      body: (
        <>
          <p>
            ロジットを指数にして合計 1 に正規化すると、<strong>次トークンの確率分布</strong>になります。差が大きいロジットほど確率の差が極端になります。
          </p>
          <p>
            「{vocabDisp[top]}」の確率は {fmtPct(base.probs[top])} です。
          </p>
        </>
      ),
      formula: `p[v] = exp(logit[v]) / Σ_u exp(logit[u])`,
    },
    {
      id: 'temperature',
      title: '温度：分布の尖り方',
      body: (
        <>
          <p>
            ロジットを温度 T で割ってから softmax にかけます。<strong>T が小さいほど 1 位に集中</strong>（決定的）、大きいほど平らに（多様だが雑）なります。スライダーで確かめてください。
          </p>
          <p>
            T = {fmt(temperature, 1)} のとき、最大確率は {fmtPct(dist.probs[order[0]])} です。
          </p>
        </>
      ),
      formula: `p[v] = softmax(logit / T)[v]`,
    },
    {
      id: 'sample',
      title: 'サンプリングと自己回帰',
      body: (
        <>
          <p>
            分布から 1 つ引き、<strong>文末に付け足して、また最初から</strong>forward を回す。これが「生成」です。greedy は常に最大、top-k は上位 k 個だけから確率どおりに、「確率どおり」は全語彙から引きます。
          </p>
          <p>⟨eos⟩ が出たら文の終わりです。灰色のバーは top-k で除外された候補です。</p>
        </>
      ),
    },
    {
      id: 'positions',
      title: 'すべての位置が「次」を予測している',
      body: (
        <>
          <p>
            forward は全位置で同時に予測を出しています。各位置 t の予測と、実際の次のトークン t+1 を並べました。学習では<strong>この全位置ぶんの誤差</strong>をまとめて使います（第 7 章）。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="06"
      title="出力と次トークン"
      lede="残差ストリームの最後の行を、語彙全体の確率分布に変えて 1 つ引く。それを繰り返すのが「生成」です。"
      purpose={
        <>
          ここまでの計算結果は {D} 次元のベクトルで、まだ「どの語か」の形をしていません。最後の位置のベクトルを語彙 {V} 個それぞれの点数（<Term id="logits">ロジット</Term>）に変え、<Term id="softmax">softmax</Term> で<Term id="probability">確率分布</Term>にし、そこから 1 つ引きます。引いたトークンを文末に足してまた最初から計算する（<Term id="autoregressive">自己回帰</Term>）ことで、文章が 1 語ずつ伸びていきます。
        </>
      }
      io={`${D} 次元のベクトル（最後の位置）→ ${V} 個の確率 → 1 トークン`}
      terms={['logits', 'softmax', 'probability', 'temperature', 'sampling', 'autoregressive', 'tying', 'eos', 'context']}
      steps={steps}
    >
      {(step) => (
        <div className="col">
          {step.id === 'lnf' && (
            <>
              <TokenChips tokens={tokens} positions active={T - 1} />
              <div className="card">
                <div className="viz-title">x_{L}[{T - 1}]（最終ブロックの出力）</div>
                <VectorStrip values={xLast} cell={34} colLabels={dimLabels} tag="computed" />
                <Arrow>↓ 最終 LayerNorm</Arrow>
                <VectorStrip values={lnf} cell={34} colLabels={dimLabels} tag="computed" />
              </div>
            </>
          )}
          {step.id === 'logits' && (
            <div className="row">
              <div className="card">
                <div className="viz-title">y = LN_f(x[{T - 1}])</div>
                <VectorStrip values={lnf} cell={30} />
                <Arrow>· W_te[{top}]（「{vocabDisp[top]}」の埋め込み）</Arrow>
                <VectorStrip values={wte.subarray(top * D, (top + 1) * D)} cell={30} tag="param" />
                <div className="formula">logit[{top}] = {fmt(dot, 4)}</div>
              </div>
              <div className="card grow">
                <BarChart title={`ロジット（上位 10 / ${V}）`} items={order.slice(0, 10).map((v, i) => ({ label: vocabDisp[v], value: logits[v], highlight: i === 0 }))} width={420} />
              </div>
            </div>
          )}
          {step.id === 'softmax' && (
            <div className="row">
              <div className="card grow">
                <BarChart title="ロジット" items={order.slice(0, 10).map((v, i) => ({ label: vocabDisp[v], value: logits[v], highlight: i === 0 }))} width={360} />
              </div>
              <div className="card grow">
                <BarChart title="確率（T = 1）" items={bars(base)} max={1} format={(v) => fmtPct(v)} width={360} />
              </div>
            </div>
          )}
          {step.id === 'temperature' && (
            <>
              {samplingCtl}
              <div className="row">
                <div className="card grow">
                  <BarChart title={`logit / T　(T = ${fmt(temperature, 1)})`} items={order.slice(0, 10).map((v, i) => ({ label: vocabDisp[v], value: dist.scaled[v], highlight: i === 0 }))} width={360} />
                </div>
                <div className="card grow">
                  <BarChart title="確率" items={bars(dist)} max={1} format={(v) => fmtPct(v)} width={360} />
                </div>
              </div>
            </>
          )}
          {step.id === 'sample' && (
            <>
              {samplingCtl}
              <div className="card">
                <div className="card-title">生成中の列（朱枠 = 引いたトークン）</div>
                <TokenChips tokens={tokenizer.tokensOf(allIds)} newSet={new Set(drawn.map((_, i) => ids.length + i))} positions />
                <div className="row" style={{ marginTop: 14, alignItems: 'center' }}>
                  <button className="ctl accent" onClick={drawOne} disabled={ended}>
                    次のトークンを 1 つ引く
                  </button>
                  <button className="ctl" onClick={drawUntilEnd} disabled={ended}>
                    文末まで生成
                  </button>
                  <button className="ctl" onClick={() => setDrawn([])} disabled={drawn.length === 0}>
                    リセット
                  </button>
                  {drawn.length > 0 && (
                    <span className="muted small">
                      直前に引いた「{vocabDisp[drawn[drawn.length - 1].chosen]}」の確率は {fmtPct(drawn[drawn.length - 1].prob)}
                    </span>
                  )}
                </div>
                <div className="muted small" style={{ marginTop: 8 }}>
                  文脈：{cur.ctx.length} トークン{allIds.length > ctxLen ? `（文脈長 ${ctxLen} を超えた分は先頭から捨てています）` : ''}
                </div>
              </div>
              <div className="card">
                <BarChart title={ended ? '⟨eos⟩ が出たので終了' : '次のトークンの分布（現在の文脈）'} items={bars(cur.dist, 10, true)} max={1} format={(v) => fmtPct(v)} width={480} />
              </div>
            </>
          )}
          {step.id === 'positions' && (
            <div className="card">
              <table className="data">
                <thead>
                  <tr>
                    <th>t</th>
                    <th>入力</th>
                    <th>予測（最大）</th>
                    <th>確率</th>
                    <th>実際の次</th>
                    <th>その確率</th>
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((tk, t) => {
                    const row = acts.probs.subarray(t * V, (t + 1) * V)
                    let am = 0
                    for (let v = 1; v < V; v++) if (row[v] > row[am]) am = v
                    const next = t + 1 < T ? ids[t + 1] : null
                    return (
                      <tr key={t} className={next !== null && next === am ? 'hi' : ''}>
                        <td className="num">{t}</td>
                        <td className="mono">{displayToken(tk)}</td>
                        <td className="mono">{vocabDisp[am]}</td>
                        <td className="num">{fmtPct(row[am])}</td>
                        <td className="mono">{next === null ? '—' : vocabDisp[next]}</td>
                        <td className="num">{next === null ? '—' : fmtPct(row[next])}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <p className="muted small" style={{ marginTop: 10 }}>
                朱の行 = 予測が実際の次トークンと一致。
              </p>
            </div>
          )}
        </div>
      )}
    </ChapterLayout>
  )
}
