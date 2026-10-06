import { useMemo } from 'react'
import { Term } from '../../content/glossary'
import { accuracy, confusion, SoftmaxRegression } from '../../engine/classifier'
import { ChapterLayout } from '../shell/ChapterLayout'
import { ML_LR, useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { fmt, fmtPct } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import type { ChapterProps } from './index'
import { ClassBadge, useClassNames } from './MlParts'

const CHECKPOINTS = [0, 10, 30, 100, 300, 1000, 3000]

export function MlEval(_: ChapterProps) {
  const ml = useMl()
  const { ds, model } = ml
  const names = useClassNames()
  const F = ds.vocab.length
  const C = ds.classes.length
  const trainAcc = accuracy(model, ds, ds.train)
  const testAcc = accuracy(model, ds, ds.test)
  const conf = confusion(model, ds, ds.test)
  const majority = Math.max(...ds.classes.map((_, c) => ds.test.filter((i) => ds.y[i] === c).length)) / ds.test.length

  // misclassified sentences, with the word that pushed hardest toward the wrong class
  const errors = [...ds.test, ...ds.train].flatMap((i) => {
    const pred = model.predict(ds.X[i])
    if (pred === ds.y[i]) return []
    let best = -1
    let bestGap = -Infinity
    for (let f = 0; f < F; f++) {
      if (!ds.X[i][f]) continue
      const gap = (model.W[pred * F + f] - model.W[ds.y[i] * F + f]) * ds.X[i][f]
      if (gap > bestGap) {
        bestGap = gap
        best = f
      }
    }
    const word = best >= 0 ? ds.vocab[best] : null
    const seenWith = word ? ds.train.filter((j) => ds.X[j][best] > 0).map((j) => ds.texts[j]) : []
    return [{ i, pred, test: ds.test.includes(i), word, seenWith }]
  })

  // a fresh run to show how train / test accuracy move with longer training
  const curve = useMemo(() => {
    const m = new SoftmaxRegression(C, F)
    const rows: { steps: number; train: number; test: number; trainLoss: number; testLoss: number }[] = []
    let s = 0
    for (const target of CHECKPOINTS) {
      for (; s < target; s++) m.step(ds.X, ds.y, ds.train, ML_LR)
      rows.push({ steps: target, train: accuracy(m, ds, ds.train), test: accuracy(m, ds, ds.test), trainLoss: m.loss(ds.X, ds.y, ds.train), testLoss: m.loss(ds.X, ds.y, ds.test) })
    }
    return rows
  }, [ds, C, F])

  const steps: Step[] = [
    {
      id: 'acc',
      title: '正解率：見たことのない文で測る',
      body: (
        <>
          <p>
            いまの分類器（{ml.steps} ステップ学習）の<Term id="accuracy">正解率</Term>は、学習用の文で {fmtPct(trainAcc, 0)}、<Term id="testdata">テスト用</Term>の文で {fmtPct(testAcc, 0)} です。
          </p>
          <p>
            大事なのはテスト用の方です。学習用の文は答えを見ながら重みを決めたので、当たって当然です。テスト用の文は一度も見せていないので、その成績が「新しい文にも通用するか」の目安になります。
          </p>
          <p>比べる相手も必要です。テスト用の文で「いつもいちばん多いクラスと答える」だけでも {fmtPct(majority, 0)} 当たります。それを上回って初めて、何かを学んだと言えます。</p>
        </>
      ),
    },
    {
      id: 'confusion',
      title: '混同行列：何を何と間違えたか',
      body: (
        <>
          <p>
            正解率は 1 つの数なので、どのクラスで間違えやすいかは分かりません。左の<Term id="confusion">混同行列</Term>は、テスト用の文を「正解（行）× 予測（列）」で数えたものです。対角線上が正解、それ以外が間違いです。
          </p>
          <p>上部の入力文や ML-5 の学習ステップを変えてから戻ってくると、表も変わります。</p>
        </>
      ),
    },
    {
      id: 'errors',
      title: '間違えた文と、その理由',
      body: (
        <>
          <p>間違えた文について、間違ったクラスへいちばん強く引っ張った語を調べると、理由がたいてい読めます（左の表）。</p>
          <p>
            多いのは、ある語が<strong>学習用の文ではたまたま一方のクラスとしか一緒に出てこなかった</strong>パターンです。モデルは語の意味を知らないので、「この語が出たらこのクラス」という見かけの関係をそのまま覚えます。直すには、その語が別のクラスで出てくる例を学習データに足すしかありません。
          </p>
          <p>
            LLM が事実と違うことをもっともらしく言う（ハルシネーション）のも、根は同じです。モデルは学習データの中の<strong>統計的なつながり</strong>を覚えているだけで、それが本当かどうかは確かめていません。
          </p>
        </>
      ),
    },
    {
      id: 'overfit',
      title: '学習しすぎると：過学習',
      body: (
        <>
          <p>
            左の表は、重み 0 から学習し直して、ステップ数ごとに成績を測ったものです。学習用の損失は進めるほど 0 に近づきますが、テスト用の成績はある所から伸びなくなり、場合によっては悪くなります。
          </p>
          <p>
            学習用の文に<strong>合わせすぎて</strong>、たまたまの特徴まで覚えてしまう現象を<Term id="overfit">過学習</Term>と呼びます。データが少ないほど起きやすく、テストデータを取りのけておく一番の理由はこれを見つけるためです。
          </p>
        </>
      ),
    },
    {
      id: 'llm-eval',
      title: 'LLM の評価はなぜ難しいか',
      body: (
        <>
          <p>この分類器は答えが 3 択なので、正解と照らして ○✕ が付けられます。LLM の出力は文章で、正解が 1 つに決まりません。「ねこは」の続きは「さかなをたべる」でも「しろい」でもよいのです。</p>
          <p>そのため LLM は、次の語の損失（LLM-7）、選択式の問題集（ベンチマーク）での正解率、人やほかの LLM による採点など、いくつもの方法を組み合わせて評価します。どれも一長一短で、「LLM の性能をどう測るか」はそれ自体が大きな研究分野です。</p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="評価"
      lede="学習に使わなかった文で、どれだけ当たるかを測ります。"
      purpose={<>学習用の文で当たるのは当然です。知りたいのは新しい文に通用するかどうかで、それを測るためにテストデータを取りのけておきました。</>}
      io={`テスト用 ${ds.test.length} 文 → 予測 → 正解率・混同行列`}
      terms={['accuracy', 'testdata', 'confusion', 'overfit']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'acc')
          return (
            <div className="card" style={{ maxWidth: 520 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th></th>
                    <th>正解率</th>
                    <th>損失</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>学習用 {ds.train.length} 文</td>
                    <td className="num">{fmtPct(trainAcc, 0)}</td>
                    <td className="num">{fmt(model.loss(ds.X, ds.y, ds.train), 3)}</td>
                  </tr>
                  <tr>
                    <td>テスト用 {ds.test.length} 文</td>
                    <td className="num">{fmtPct(testAcc, 0)}</td>
                    <td className="num">{fmt(model.loss(ds.X, ds.y, ds.test), 3)}</td>
                  </tr>
                  <tr>
                    <td className="muted">いちばん多いクラスと答えるだけ</td>
                    <td className="num muted">{fmtPct(majority, 0)}</td>
                    <td></td>
                  </tr>
                </tbody>
              </table>
            </div>
          )
        if (step.id === 'confusion')
          return <Heatmap values={conf} rows={C} cols={C} cell={72} cellH={36} mode="sequential" showValues digits={0} rowLabels={names.map((n) => `正解 ${n}`)} rowLabelWidth={110} colLabels={names.map((n) => `予測 ${n}`)} highlightCells={ds.classes.map((_, c) => [c, c] as [number, number])} title={`テスト用 ${ds.test.length} 文の混同行列`} rowName="正解" colName="予測" tag="computed" />
        if (step.id === 'errors')
          return errors.length ? (
            <div className="card" style={{ maxWidth: 820 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>文</th>
                    <th>正解</th>
                    <th>予測</th>
                    <th>引っ張った語</th>
                    <th>その語が学習用で出てきた文</th>
                  </tr>
                </thead>
                <tbody>
                  {errors.map((e) => (
                    <tr key={e.i}>
                      <td className="mono">
                        {ds.texts[e.i]} <span className="muted small">{e.test ? 'テスト' : '学習'}</span>
                      </td>
                      <td>
                        <ClassBadge c={ds.y[e.i]} />
                      </td>
                      <td>
                        <ClassBadge c={e.pred} />
                      </td>
                      <td className="mono">{e.word ?? '—'}</td>
                      <td className="mono small muted">{e.seenWith.length ? e.seenWith.join('　') : 'なし'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="card muted">いまの分類器は、学習用・テスト用ともに 1 文も間違えていません。ML-5 で重みを 0 に戻して数ステップだけ学習させると、間違いが見られます。</div>
          )
        if (step.id === 'overfit')
          return (
            <div className="card" style={{ maxWidth: 620 }}>
              <div className="card-title">重み 0 から学習し直したときの成績</div>
              <table className="data">
                <thead>
                  <tr>
                    <th>ステップ</th>
                    <th>学習用 正解率</th>
                    <th>テスト用 正解率</th>
                    <th>学習用 損失</th>
                    <th>テスト用 損失</th>
                  </tr>
                </thead>
                <tbody>
                  {curve.map((r) => (
                    <tr key={r.steps}>
                      <td className="num">{r.steps.toLocaleString()}</td>
                      <td className="num">{fmtPct(r.train, 0)}</td>
                      <td className="num">{fmtPct(r.test, 0)}</td>
                      <td className="num">{fmt(r.trainLoss, 3)}</td>
                      <td className="num">{fmt(r.testLoss, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        return (
          <div className="card" style={{ maxWidth: 640 }}>
            <table className="data">
              <thead>
                <tr>
                  <th></th>
                  <th>この分類器</th>
                  <th>LLM</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>出力</td>
                  <td>3 択のどれか</td>
                  <td>自由な文章</td>
                </tr>
                <tr>
                  <td>正解</td>
                  <td>1 つ（ラベル）</td>
                  <td>たくさんありうる</td>
                </tr>
                <tr>
                  <td>測り方</td>
                  <td>テストデータの正解率</td>
                  <td>損失、ベンチマーク、人や LLM による採点</td>
                </tr>
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
