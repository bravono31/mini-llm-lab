import { useEffect, useMemo, useRef, useState } from 'react'
import { Term } from '../../content/glossary'
import { listActGrads } from '../../engine/backward'
import { forward } from '../../engine/forward'
import { generate } from '../../engine/generate'
import { viewIn } from '../../engine/params'
import { mulberry32 } from '../../engine/rng'
import { displayToken } from '../../engine/tokenizer'
import type { Batch, TrainStepResult } from '../../engine/trainer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmt, fmtPct } from '../viz/colors'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import { LossChart } from '../viz/LossChart'
import { TokenChips } from '../viz/TokenChips'
import { Arrow } from './controls'
import { SpecHeat, stats } from './SpecHeat'

export function Train(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, tokenizer, session, pretrained, corpus } = lab
  const { trainer } = session
  const { vocabSize: V, ctxLen: T } = params.config
  const [preview, setPreview] = useState<Batch>(() => trainer.sampleBatch())
  const [last, setLast] = useState<TrainStepResult | null>(null)
  const [specName, setSpecName] = useState('layer0.attn.qkv.w')
  const [lr, setLr] = useState(trainer.lr)
  const [running, setRunning] = useState(false)
  const [b, setB] = useState(0)
  const raf = useRef(0)

  // a new session (reset / language switch) invalidates everything local
  useEffect(() => {
    setPreview(trainer.sampleBatch())
    setLast(null)
    setLr(trainer.lr)
    setRunning(false)
  }, [trainer])

  const runSteps = (n: number) => {
    let r: TrainStepResult | null = null
    for (let i = 0; i < n; i++) r = trainer.trainStep(i === 0 ? preview : undefined)
    setLast(r)
    setPreview(trainer.sampleBatch())
    lab.notifyParamsChanged()
  }

  useEffect(() => {
    if (!running) return
    let alive = true
    const tick = () => {
      if (!alive) return
      let r: TrainStepResult | null = null
      for (let i = 0; i < 4; i++) r = trainer.trainStep()
      setLast(r)
      lab.notifyParamsChanged()
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => {
      alive = false
      cancelAnimationFrame(raf.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, trainer])

  const B = preview.inputs.length / T
  const batchTokens = (arr: Int32Array, bi: number) => tokenizer.tokensOf(arr.subarray(bi * T, (bi + 1) * T))
  // loss preview on the pending batch (no update)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const previewActs = useMemo(() => forward(params, preview.inputs, B, T, preview.targets), [params, preview, B, T, lab.version])
  const shown = last ?? null
  const lossActs = shown ? shown.acts : previewActs
  const lossBatch = shown ? shown.batch : preview
  const bb = Math.min(b, B - 1)
  const lossRow = lossActs.losses!.subarray(bb * T, (bb + 1) * T)
  const inTok = batchTokens(lossBatch.inputs, bb).map(displayToken)
  const tgTok = batchTokens(lossBatch.targets, bb).map(displayToken)

  const spec = params.byName.get(specName)!
  const gradNorms = useMemo(() => {
    if (!last) return []
    return listActGrads(last.acts, last.actGrads)
      .map((nb) => ({ label: nb.label, value: stats(nb.data).norm }))
      .reverse()
  }, [last])
  const paramGradGroups = useMemo(() => {
    if (!last) return []
    return params.specs.filter((s) => s.shape.length === 2).map((s) => ({ label: s.name.replace('layer', 'L').replace('.attn', '').replace('.mlp', ''), value: stats(viewIn(last.grads, s)).norm }))
  }, [last, params.specs])

  const history = useMemo(
    () => (session.origin === 'pretrained' ? [...pretrained.lossHistory, ...trainer.lossHistory] : [...trainer.lossHistory]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [session, pretrained, trainer, lab.version],
  )
  const sampleNow = useMemo(() => {
    const ids = tokenizer.encode(corpus.prompts[0])
    const out = generate(params, ids, 12, { mode: 'greedy', temperature: 1, k: 5 }, mulberry32(3))
    return tokenizer.decode([...ids, ...out.map((s) => s.chosen)]).replace(/\n$/, '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, tokenizer, corpus, lab.version])

  const diffMax = useMemo(() => {
    if (!last) return 1
    let m = 0
    const before = viewIn(last.before, spec)
    const after = viewIn(params.data, spec)
    for (let i = 0; i < spec.size; i++) m = Math.max(m, Math.abs(after[i] - before[i]))
    return m || 1
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [last, spec, lab.version])

  const trainButtons = (
    <div className="row" style={{ alignItems: 'center', gap: 10 }}>
      <button className="ctl accent" onClick={() => runSteps(1)} disabled={running}>
        1 ステップ学習
      </button>
      <button className="ctl" onClick={() => runSteps(10)} disabled={running}>
        10 ステップ
      </button>
      <button className="ctl" onClick={() => runSteps(100)} disabled={running}>
        100 ステップ
      </button>
      <button className={'ctl' + (running ? ' primary' : '')} onClick={() => setRunning(!running)}>
        {running ? '停止' : '自動'}
      </button>
      <span className="field">
        学習率
        <input
          type="range"
          min={-4}
          max={-1.5}
          step={0.1}
          value={Math.log10(lr)}
          onChange={(e) => {
            const v = Math.pow(10, Number(e.target.value))
            setLr(v)
            trainer.setLr(v)
          }}
        />
        <span className="mono">{lr.toExponential(1)}</span>
      </span>
      <span className="muted small">
        このセッションで {trainer.step} ステップ · 現在 {session.origin === 'pretrained' ? '学習済み重み' : 'ランダム初期化'}
      </span>
    </div>
  )

  const needStep = (
    <div className="card">
      <p className="muted">まだ学習ステップを実行していません。</p>
      <button className="ctl accent" onClick={() => runSteps(1)}>
        1 ステップ実行して勾配を見る
      </button>
    </div>
  )

  const steps: Step[] = [
    {
      id: 'data',
      title: '学習データ = ずらした自分',
      body: (
        <>
          <p>
            コーパスの連続 {T + 1} トークンを切り出し、前 {T} 個を<strong>入力</strong>、1 つずらした後ろ {T} 個を<strong>正解</strong>にします。人手のラベルは要りません。これを {B} 本束ねたものが 1 バッチです。
          </p>
          <p>位置 t の正解は「t+1 番目のトークン」。前章で見た「全位置が次を予測する」形と対応しています。</p>
        </>
      ),
      formula: `inputs  = tokens[s : s+${T}]\ntargets = tokens[s+1 : s+${T + 1}]`,
    },
    {
      id: 'loss',
      title: '損失：正解の確率の −log',
      body: (
        <>
          <p>
            forward で各位置の確率分布を出し、<strong>正解トークンに割り当てた確率</strong> p の −log を損失にします（交差エントロピー）。p = 1 なら 0、p が小さいほど大きい。全位置・全バッチの平均が 1 ステップの損失です。
          </p>
          <p>
            このバッチの平均損失は {fmt(lossActs.meanLoss!, 3)}（一様分布なら ln {V} = {fmt(Math.log(V), 2)}）。
          </p>
        </>
      ),
      formula: `loss[t] = −log p(target[t])\nL = mean over B×T`,
    },
    {
      id: 'backward',
      title: '逆伝播：勾配が出力から遡る',
      body: (
        <>
          <p>
            損失 L を各中間バッファで微分した値（勾配）を、<strong>出力側から入力側へ</strong>連鎖律で順に求めます。ロジットの勾配は単に「確率 − 正解の one-hot」。それが softmax・W_o・注意・LayerNorm…と逆順に伝わります。
          </p>
          <p>棒はバッファごとの勾配のノルム。層を遡っても消えず（残差接続のおかげ）、埋め込みまで届いています。</p>
        </>
      ),
      formula: `∂L/∂logit = p − onehot(target)\n∂L/∂x = (∂L/∂y)(∂y/∂x)   ← 連鎖律`,
    },
    {
      id: 'grads',
      title: 'パラメータの勾配',
      body: (
        <>
          <p>
            各重み行列について「その要素を少し増やしたら損失がどう変わるか」が求まりました。<strong>符号と大きさ</strong>が、次の更新の方向と強さを決めます。
          </p>
          <p>右のバーは行列ごとの勾配ノルム。どの層が今いちばん「直したがっているか」が読めます。</p>
        </>
      ),
      formula: `g = ∂L/∂W   （W と同じ形）`,
    },
    {
      id: 'update',
      title: 'AdamW で重みを更新',
      body: (
        <>
          <p>
            勾配の逆方向へ少し動かします。AdamW は勾配の<strong>移動平均 m</strong>と<strong>二乗平均 v</strong>を持ち、要素ごとに歩幅を自動調整します。学習率 η が全体の歩幅、weight decay が重みをわずかに 0 へ戻します。
          </p>
          <p>更新前・勾配・更新後・差分を同じ行列で並べました。差分は勾配と逆符号になっています。</p>
        </>
      ),
      formula: `m ← β₁m + (1−β₁)g      v ← β₂v + (1−β₂)g²\nW ← W − η·( m̂/(√v̂+ε) + λW )`,
    },
    {
      id: 'curve',
      title: '学習曲線と生成の変化',
      body: (
        <>
          <p>
            ステップを重ねると損失が下がり、固定プロンプト「{corpus.prompts[0]}」からの生成が文らしくなっていきます。<strong>「ランダム初期化」</strong>でゼロから学習を始めることもできます。
          </p>
          <p>学習済み重みは {pretrained.trainSteps} ステップ学習したものです。点線は一様分布の損失 ln V。</p>
        </>
      ),
    },
  ]

  const resetButtons = (
    <div className="row" style={{ gap: 10 }}>
      <button className="ctl" onClick={lab.resetToRandom}>
        ランダム初期化
      </button>
      <button className="ctl" onClick={lab.resetToPretrained}>
        学習済みに戻す
      </button>
    </div>
  )

  return (
    <ChapterLayout
      num="07"
      title="学習"
      lede="正解との誤差を損失にし、勾配を逆向きに流して、重みを少し動かす。それを何千回も繰り返します。"
      purpose={
        <>
          ここまでの章の<Term id="parameter">パラメータ</Term>は、最初はすべて乱数です。乱数のままでは予測はでたらめなので、コーパスの文を使って「次のトークン」を当てさせ、外れ具合を<Term id="loss">損失</Term>という 1 つの数にします。損失を各パラメータで微分した<Term id="gradient">勾配</Term>が「どちらに動かせば損失が減るか」を教えてくれるので、その方向に少し動かします（<Term id="adamw">AdamW</Term>）。これを数千<Term id="step">ステップ</Term>繰り返すと、乱数だった重みがコーパスの規則を写します。
        </>
      }
      io="バッチ（入力と正解）→ 損失 → 勾配 → 更新された重み"
      terms={['loss', 'gradient', 'backprop', 'adamw', 'lr', 'batch', 'step', 'parameter', 'forward', 'pretrain']}
      steps={steps}
      aside={resetButtons}
    >
      {(step) => (
        <div className="col">
          {step.id === 'data' && (
            <>
              <div className="row" style={{ alignItems: 'center', gap: 10 }}>
                <span className="field">
                  バッチ内の例
                  <input type="range" min={0} max={B - 1} value={bb} onChange={(e) => setB(Number(e.target.value))} style={{ width: 120 }} />
                  <span className="mono">
                    {bb + 1} / {B}
                  </span>
                </span>
                <button className="ctl" onClick={() => setPreview(trainer.sampleBatch())}>
                  別のバッチを引く
                </button>
              </div>
              <div className="card">
                <div className="viz-title">入力（コーパス位置 {preview.starts[bb]} から）</div>
                <TokenChips tokens={batchTokens(preview.inputs, bb)} positions />
                <Arrow>↓ 1 つずらす</Arrow>
                <div className="viz-title">正解</div>
                <TokenChips tokens={batchTokens(preview.targets, bb)} positions />
              </div>
            </>
          )}
          {step.id === 'loss' && (
            <>
              <div className="row" style={{ alignItems: 'center', gap: 10 }}>
                <span className="field">
                  バッチ内の例
                  <input type="range" min={0} max={B - 1} value={bb} onChange={(e) => setB(Number(e.target.value))} style={{ width: 120 }} />
                  <span className="mono">
                    {bb + 1} / {B}
                  </span>
                </span>
                <span className="muted small">{shown ? `直近の学習ステップ ${shown.step} のバッチ` : '次に学習するバッチ（更新前の重みで forward）'}</span>
              </div>
              <div className="row" style={{ gap: 36 }}>
                <div className="stat">
                  <span className="label">この例の平均</span>
                  <span className="value">{fmt(Array.from(lossRow).reduce((a, c) => a + c, 0) / T, 3)}</span>
                </div>
                <div className="stat">
                  <span className="label">バッチ全体の平均 L</span>
                  <span className="value">{fmt(lossActs.meanLoss!, 3)}</span>
                </div>
                <div className="stat">
                  <span className="label">一様分布なら ln V</span>
                  <span className="value">{fmt(Math.log(V), 3)}</span>
                </div>
              </div>
              <div className="card">
                <BarChart
                  title="位置ごとの損失 −log p(正解)　（入力 → 正解）"
                  items={Array.from(lossRow, (v, t) => ({ label: `${inTok[t]} → ${tgTok[t]}`, value: v, sub: `p=${fmtPct(Math.exp(-v), 1)}`, highlight: v === Math.max(...lossRow) }))}
                  width={620}
                  barHeight={18}
                  labelWidth={150}
                  valueWidth={120}
                  max={Math.max(Math.log(V), ...lossRow)}
                />
              </div>
            </>
          )}
          {step.id === 'backward' &&
            (last ? (
              <div className="col">
                <div className="card">
                  <BarChart title="‖∂L/∂(中間バッファ)‖ 出力側 → 入力側" items={gradNorms} width={560} labelWidth={130} format={(v) => fmt(v, 3)} />
                </div>
                <div className="card">
                  <div className="card-title">例 {bb + 1}：ロジットの勾配 p − onehot（各位置 × 語彙、上位数語）</div>
                  {(() => {
                    const dl = last.actGrads.dlogits
                    const rows = T
                    const top = [...Array(V).keys()]
                      .sort((x, y) => {
                        let sx = 0, sy = 0
                        for (let t = 0; t < T; t++) {
                          sx += Math.abs(dl[(bb * T + t) * V + x])
                          sy += Math.abs(dl[(bb * T + t) * V + y])
                        }
                        return sy - sx
                      })
                      .slice(0, 12)
                    const sub = new Float32Array(rows * top.length)
                    for (let t = 0; t < rows; t++) top.forEach((v, j) => (sub[t * top.length + j] = dl[(bb * T + t) * V + v]))
                    return <Heatmap values={sub} rows={rows} cols={top.length} cell={22} rowLabels={inTok.map((s, t) => `${t} ${s}`)} rowLabelWidth={70} colLabels={top.map((v) => displayToken(tokenizer.vocab[v]))} legend rowName="t" colName="v" />
                  })()}
                  <div className="muted small" style={{ marginTop: 8 }}>
                    正解トークンの列が負（確率を上げたい）、他は正（下げたい）。値は 1/(B·T) でスケールされています。
                  </div>
                </div>
              </div>
            ) : (
              needStep
            ))}
          {step.id === 'grads' &&
            (last ? (
              <>
                <span className="field">
                  テンソル
                  <select value={specName} onChange={(e) => setSpecName(e.target.value)}>
                    {params.specs.map((s) => (
                      <option key={s.name} value={s.name}>
                        {s.name} [{s.shape.join('×')}] — {s.label}
                      </option>
                    ))}
                  </select>
                </span>
                <div className="row">
                  <div className="card">
                    <SpecHeat data={last.grads} spec={spec} title={`∂L/∂${spec.name}`} />
                    <div className="muted small" style={{ marginTop: 8 }}>
                      ノルム {fmt(stats(viewIn(last.grads, spec)).norm, 4)} · 最大 |g| {fmt(stats(viewIn(last.grads, spec)).absMax, 4)}
                    </div>
                  </div>
                  <div className="card grow">
                    <BarChart title="行列ごとの勾配ノルム" items={paramGradGroups.map((g) => ({ ...g, highlight: spec.name.replace('layer', 'L').replace('.attn', '').replace('.mlp', '') === g.label }))} width={400} labelWidth={110} format={(v) => fmt(v, 3)} onClick={(i) => setSpecName(params.specs.filter((s) => s.shape.length === 2)[i].name)} />
                  </div>
                </div>
              </>
            ) : (
              needStep
            ))}
          {step.id === 'update' &&
            (last ? (
              <>
                <div className="row" style={{ alignItems: 'center', gap: 16 }}>
                  <span className="field">
                    テンソル
                    <select value={specName} onChange={(e) => setSpecName(e.target.value)}>
                      {params.specs.map((s) => (
                        <option key={s.name} value={s.name}>
                          {s.name} [{s.shape.join('×')}]
                        </option>
                      ))}
                    </select>
                  </span>
                  {trainButtons}
                </div>
                <div className="row">
                  <div className="card">
                    <SpecHeat data={last.before} spec={spec} title="更新前 W" />
                  </div>
                  <div className="card">
                    <SpecHeat data={last.grads} spec={spec} title="勾配 g" />
                  </div>
                  <div className="card">
                    <SpecHeat data={params.data} spec={spec} title="更新後 W′" />
                  </div>
                  <div className="card">
                    {(() => {
                      const before = viewIn(last.before, spec)
                      const after = viewIn(params.data, spec)
                      const diff = new Float32Array(spec.size)
                      for (let i = 0; i < spec.size; i++) diff[i] = after[i] - before[i]
                      const rows = spec.shape.length === 2 ? spec.shape[0] : 1
                      const cols = spec.shape.length === 2 ? spec.shape[1] : spec.shape[0]
                      const c = Math.max(4, Math.min(14, Math.floor(560 / cols)))
                      return <Heatmap values={diff} rows={rows} cols={cols} cell={c} gap={c >= 8 ? 1 : 0} maxAbs={diffMax} title="差分 W′ − W" legend rowName="i" colName="j" />
                    })()}
                  </div>
                </div>
                <div className="muted small">
                  ステップ {last.step}：損失 {fmt(last.loss, 3)} · 勾配ノルム {fmt(last.gradNorm, 3)} · 更新量ノルム {fmt(last.updateNorm, 4)}
                </div>
              </>
            ) : (
              needStep
            ))}
          {step.id === 'curve' && (
            <>
              {trainButtons}
              <div className="card">
                <LossChart history={history} baseline={Math.log(V)} marks={session.origin === 'pretrained' ? [{ x: pretrained.lossHistory.length - 1, label: '学習済み' }] : undefined} title={`損失（${history.length} ステップ）`} />
              </div>
              <div className="row">
                <div className="card grow">
                  <div className="card-title">いまの重みで「{corpus.prompts[0]}」から greedy 生成</div>
                  <div className="mono" style={{ fontSize: 18 }}>
                    {sampleNow}
                  </div>
                  <div className="muted small" style={{ marginTop: 6 }}>
                    直近の損失 {history.length ? fmt(history[history.length - 1], 3) : '—'}
                  </div>
                </div>
                <div className="card">
                  <div className="card-title">学習済み重みの学習中サンプル</div>
                  <table className="data">
                    <tbody>
                      {pretrained.samples.map((s) => (
                        <tr key={s.step}>
                          <td className="num">{s.step}</td>
                          <td className="mono">{s.text}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              {last && (
                <div className="card">
                  <div className="viz-title">直近ステップの位置ごとの損失（例 {bb + 1}）</div>
                  <VectorStrip values={lossRow} cell={30} mode="sequential" max={Math.log(V)} colLabels={inTok} />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </ChapterLayout>
  )
}
