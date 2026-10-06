import { useState } from 'react'
import { Term } from '../../content/glossary'
import { accuracy, segment } from '../../engine/classifier'
import { totalSize } from '../../engine/params'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import { ML_LR, ML_PRETRAIN_STEPS, useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { fmt, fmtPct } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import { LossChart } from '../viz/LossChart'
import type { ChapterProps } from './index'
import { ClassBadge, ProbBars, useClassNames, WordChips } from './MlParts'

export function MlTrain(_: ChapterProps) {
  const lab = useLab()
  const ml = useMl()
  const { ds, model, session } = ml
  const names = useClassNames()
  const F = ds.vocab.length
  const C = ds.classes.length
  const [ex, setEx] = useState(0)
  const i = ds.train[Math.min(ex, ds.train.length - 1)]
  const p = model.probs(ds.X[i])
  const onehot = ds.classes.map((_, c) => (c === ds.y[i] ? 1 : 0))
  const used = ds.vocab.flatMap((_, f) => (ds.X[i][f] ? [f] : []))
  // this one example's gradient, rows = words that occur
  const exGrad = Float64Array.from(used.flatMap((f) => ds.classes.map((_, c) => (p[c] - onehot[c]) * ds.X[i][f])))

  const transpose = (m: Float64Array) => {
    const t = new Float64Array(F * C)
    for (let f = 0; f < F; f++) for (let c = 0; c < C; c++) t[f * C + c] = m[c * F + f]
    return t
  }
  const trainLoss = session.trainLoss.at(-1)!
  const testLoss = session.testLoss.at(-1)!
  const ln3 = Math.log(C)

  const buttons = (
    <div className="row" style={{ alignItems: 'center', gap: 10 }}>
      <button className="ctl accent" onClick={() => ml.train(1)}>
        1 ステップ学習
      </button>
      <button className="ctl" onClick={() => ml.train(10)}>
        10 ステップ
      </button>
      <button className="ctl" onClick={() => ml.train(100)}>
        100 ステップ
      </button>
      <button className="ctl" onClick={ml.resetToZero}>
        重みを 0 に戻す
      </button>
      <button className="ctl" onClick={ml.resetToTrained}>
        学習済み（{ML_PRETRAIN_STEPS} ステップ）に戻す
      </button>
      <span className="muted small">
        いま {ml.steps} ステップ ・ 学習率 {ML_LR}
      </span>
    </div>
  )

  const status = (
    <div className="card" style={{ maxWidth: 520 }}>
      <table className="data">
        <thead>
          <tr>
            <th></th>
            <th>損失</th>
            <th>正解率</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>学習用 {ds.train.length} 文</td>
            <td className="num">{fmt(trainLoss, 3)}</td>
            <td className="num">{fmtPct(accuracy(model, ds, ds.train), 0)}</td>
          </tr>
          <tr>
            <td>テスト用 {ds.test.length} 文</td>
            <td className="num">{fmt(testLoss, 3)}</td>
            <td className="num">{fmtPct(accuracy(model, ds, ds.test), 0)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )

  const steps: Step[] = [
    {
      id: 'loss',
      title: '損失：予測の悪さを 1 つの数に',
      body: (
        <>
          <p>
            学習の目標は、学習用の各文で<strong>正解クラスの確率を 1 に近づける</strong>ことです。その悪さを測るのが<Term id="loss">損失</Term>で、正解クラスに付けた確率 p の −log を全文で平均します。LLM-7 の損失と<strong>まったく同じ式</strong>（交差エントロピー）で、違いは「正解」が次のトークンかラベルかだけです。
          </p>
          <p>
            重みがすべて 0 なら 3 クラスとも確率 1/3 で、損失は −log(1/3) = {fmt(ln3, 3)}。これが出発点です。いまの損失は {fmt(trainLoss, 3)} です。
          </p>
        </>
      ),
      formula: 'L = −(1/N) Σ_i log p_{y_i}(x_i)',
    },
    {
      id: 'grad',
      title: '勾配：どの点数を、どちらへ動かすか',
      body: (
        <>
          <p>
            損失を減らすには、各重みをどちらへ動かせばよいか（<Term id="gradient">勾配</Term>）が要ります。この分類器は層が 1 つしかないので、勾配が<strong>1 行の式</strong>で求まります：「予測の確率 − 正解（正解クラスだけ 1）」に、その語の回数を掛けるだけ。
          </p>
          <p>
            左は学習用の 1 文での例です。正解クラスは p − y が負なので、文に出た語の点数が<strong>上がり</strong>、ほかのクラスの点数は<strong>下がり</strong>ます（更新では勾配の逆向きに動かすため）。文に出ていない語の点数は動きません。
          </p>
          <p>
            LLM は層が何段も重なっているので、この計算を出力側から入力側へ連鎖律で順にたどる<Term id="backprop">逆伝播</Term>が必要でした（LLM-7）。やっていることの意味は同じです。
          </p>
        </>
      ),
      formula: '∂L/∂W[c, f] = (p_c − y_c) · x_f',
    },
    {
      id: 'run',
      title: '学習させてみる',
      body: (
        <>
          <p>
            学習用 {ds.train.length} 文全部の勾配を平均し、重みを勾配の逆向きに <strong>学習率 {ML_LR}</strong> 倍だけ動かす。これが 1 ステップ（<Term id="step">ステップ</Term>）で、<strong>勾配降下法</strong>と呼びます。
          </p>
          <p>「重みを 0 に戻す」を押してから、1 ステップずつ進めてみてください。損失が {fmt(ln3, 2)} から下がり、下の重みの表に色が付いていきます。100 ステップもすれば学習用の文はほぼ全問正解になります。</p>
          <p>計算は一瞬で終わります。LLM 編のミニモデルは同じことを 3,000 ステップで数十秒、実際の LLM は数千台の GPU で数か月かけます。</p>
        </>
      ),
    },
    {
      id: 'compare',
      title: 'LLM-7 の学習との対応',
      body: (
        <>
          <p>「損失を測る → 勾配を求める → 重みを少し動かす」を繰り返す、という骨組みは LLM の学習とまったく同じです。違うのは、何をどのくらいの規模でやるかです（左の表）。</p>
          <p>
            この共通の骨組みこそが「機械学習」で、LLM はその上に、深い<Term id="nn">ニューラルネットワーク</Term>と膨大なデータを載せたものと言えます。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="学習"
      lede="損失を測り、勾配の逆向きに重みを少し動かす。LLM と同じ手順を、点数表 1 枚でやります。"
      purpose={<>重みは最初はすべて 0 で、何も分かりません。正解とのずれを少しずつ減らす方向に重みを動かすことで、例に合った点数表ができあがります。</>}
      io="学習用の文とラベル → 損失 → 勾配 → 更新された重み"
      terms={['loss', 'gradient', 'step', 'lr', 'backprop', 'nn']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'loss')
          return (
            <div className="col">
              {status}
              <LossChart history={session.trainLoss} baseline={ln3} baselineLabel="重み 0（ln 3）" smooth={1} title="学習用の文の損失（横軸 = ステップ）" />
            </div>
          )
        if (step.id === 'grad')
          return (
            <div className="col">
              <span className="field">
                学習用の文
                <select value={ex} onChange={(e) => setEx(Number(e.target.value))}>
                  {ds.train.map((j, k) => (
                    <option key={j} value={k}>
                      {ds.texts[j]}
                    </option>
                  ))}
                </select>
              </span>
              <div className="row">
                <div className="card">
                  <div className="card-title">
                    文と正解：<ClassBadge c={ds.y[i]} />
                  </div>
                  <WordChips words={segment(ds.texts[i], ds.data)} />
                  <div style={{ marginTop: 10 }}>
                    <ProbBars probs={p} title="いまの予測 p" width={320} />
                  </div>
                </div>
                <div className="col" style={{ gap: 12 }}>
                  <Heatmap values={Float64Array.from(ds.classes.map((_, c) => p[c] - onehot[c]))} rows={1} cols={C} cell={72} cellH={24} rowLabels={['p − y']} rowLabelWidth={100} colLabels={names} showValues digits={2} title="予測 − 正解" tag="computed" />
                  {used.length > 0 && <Heatmap values={exGrad} rows={used.length} cols={C} cell={72} cellH={24} rowLabels={used.map((f) => ds.vocab[f])} rowLabelWidth={100} colLabels={names} showValues digits={2} title="この 1 文の勾配（出た語の行だけ。他の行は 0）" tag="computed" />}
                </div>
              </div>
            </div>
          )
        if (step.id === 'run')
          return (
            <div className="col">
              {buttons}
              {status}
              <div className="row">
                <LossChart history={session.trainLoss} baseline={ln3} baselineLabel="重み 0" smooth={1} width={420} height={170} title="学習用の損失" />
                <LossChart history={session.testLoss} baseline={ln3} baselineLabel="重み 0" smooth={1} width={420} height={170} title="テスト用の損失（学習には使っていない）" />
              </div>
              <div className="row">
                <Heatmap values={transpose(model.W)} rows={F} cols={C} cell={56} cellH={13} rowLabels={ds.vocab} rowLabelWidth={76} colLabels={names} title="重み W" rowName="語" colName="クラス" tag="param" />
                {ml.last && <Heatmap values={transpose(ml.last.dW)} rows={F} cols={C} cell={56} cellH={13} rowLabels={ds.vocab} rowLabelWidth={76} colLabels={names} title="直前のステップの勾配" rowName="語" colName="クラス" tag="computed" />}
              </div>
            </div>
          )
        return (
          <div className="card" style={{ maxWidth: 720 }}>
            <table className="data">
              <thead>
                <tr>
                  <th></th>
                  <th>この編</th>
                  <th>LLM 編（LLM-7）</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>損失</td>
                  <td>交差エントロピー（正解 = ラベル）</td>
                  <td>交差エントロピー（正解 = 次のトークン）</td>
                </tr>
                <tr>
                  <td>勾配の求め方</td>
                  <td>1 行の式 (p − y)·x</td>
                  <td>逆伝播で層を逆にたどる</td>
                </tr>
                <tr>
                  <td>更新</td>
                  <td>勾配降下（学習率 {ML_LR} を掛けて引く）</td>
                  <td>AdamW（要素ごとに歩幅を調整）</td>
                </tr>
                <tr>
                  <td>1 ステップで見る例</td>
                  <td>学習用 {ds.train.length} 文すべて</td>
                  <td>16 本 × 16 トークンのバッチ</td>
                </tr>
                <tr>
                  <td>動かす重み</td>
                  <td>{model.paramCount} 個</td>
                  <td>{totalSize(lab.params.specs).toLocaleString()} 個（実際の LLM は数千億）</td>
                </tr>
                <tr>
                  <td>かかる時間</td>
                  <td>一瞬</td>
                  <td>数十秒（実際の LLM は数か月）</td>
                </tr>
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
