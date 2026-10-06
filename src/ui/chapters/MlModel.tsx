import { Term } from '../../content/glossary'
import { totalSize } from '../../engine/params'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmt } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import type { ChapterProps } from './index'
import { CLASS_COLORS, ProbBars, useClassNames, WordChips } from './MlParts'

export function MlModel(_: ChapterProps) {
  const lab = useLab()
  const ml = useMl()
  const { ds, model } = ml
  const names = useClassNames()
  const F = ds.vocab.length
  const C = ds.classes.length
  const used = ds.vocab.flatMap((_, f) => (ml.x[f] ? [f] : []))
  const logits = model.logits(ml.x)

  // rows: each word that occurs, then the bias, then the total (= logit)
  const contrib = Float64Array.from([
    ...used.flatMap((f) => ds.classes.map((_, c) => model.W[c * F + f] * ml.x[f])),
    ...model.b,
    ...logits,
  ])
  const contribRows = [...used.map((f) => (ml.x[f] > 1 ? `${ds.vocab[f]} ×${ml.x[f]}` : ds.vocab[f])), 'b（切片）', '合計 z']

  // W transposed so that one row = one word
  const wT = new Float64Array(F * C)
  for (let f = 0; f < F; f++) for (let c = 0; c < C; c++) wT[f * C + c] = model.W[c * F + f]

  const top = ds.classes.map((_, c) =>
    ds.vocab
      .map((w, f) => ({ label: w, value: model.W[c * F + f] }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6),
  )
  const llmParams = totalSize(lab.params.specs)

  const steps: Step[] = [
    {
      id: 'score',
      title: '語ごとの点数を足すだけ',
      body: (
        <>
          <p>
            この分類器（<Term id="logreg">ロジスティック回帰</Term>）は、<strong>語 × クラスの点数表</strong>（重み W）を持っています。たとえば「ねこ」なら いきもの +2、しぜん −1、ひと −1 のような 3 つの点数です。
          </p>
          <p>
            文が来たら、出てきた語の点数をクラスごとに足し、最後にクラスごとの下駄 b（切片）を足します。これがクラスごとの合計点 z です。左の表は上部の入力文での内訳で、いちばん下の行が合計です。
          </p>
          <p>計算は掛け算と足し算だけで、層もありません。LLM 編の 1 つの行列積（<Term id="projection">射影</Term>）とまったく同じ形です。</p>
        </>
      ),
      formula: 'z_c = b_c + Σ_f W[c, f] · x_f　（c = クラス、f = 辞書の語）',
    },
    {
      id: 'softmax',
      title: '合計点を確率にする',
      body: (
        <>
          <p>
            合計点 z を <Term id="softmax">softmax</Term> で合計 1 の確率に直し、いちばん大きいクラスを答えにします。LLM-6 で次のトークンの確率を出したのと<strong>同じ関数</strong>です。違うのは選択肢の数だけで、LLM は語彙の {lab.params.config.vocabSize} 個、この分類器は {C} 個です。
          </p>
          <p>上の入力文を書き換えると、確率がその場で変わります。</p>
        </>
      ),
      formula: 'p_c = exp(z_c) / Σ_k exp(z_k)',
    },
    {
      id: 'weights',
      title: '重みの表の全体',
      body: (
        <>
          <p>
            左が重み W の全体です（1 行 = 辞書の 1 語、1 列 = 1 クラス）。朱が正（そのクラスらしさを上げる）、藍が負（下げる）。この値は人が書いたものではなく、ML-5 の学習で決まりました。
          </p>
          <p>{F} 語 × {C} クラス = {F * C} 個と切片 {C} 個、合わせて <strong>{model.paramCount} 個</strong>がこのモデルの<Term id="parameter">パラメータ</Term>のすべてです。</p>
        </>
      ),
    },
    {
      id: 'reason',
      title: '判断の理由が読める',
      body: (
        <>
          <p>クラスごとに重みの大きい語を並べると、モデルが何を手がかりにしているかがそのまま読めます。いきもの なら動物の名前、ひと なら わたし・あなた が上に来ているはずです。</p>
          <p>
            重みが 1 語に 1 つずつ対応しているので、「なぜこの答えか」を「この語が +2.1 点入れたから」と説明できます。これを<strong>解釈性</strong>と呼び、古典的な機械学習の大きな長所です。
          </p>
          <p>LLM の重みは 1 つ 1 つが何を意味するのか分かりません（LLM-8）。知識は何千もの重みに分散していて、「なぜこの語を出したか」を重みから直接読み取ることは、いまも研究課題です。</p>
        </>
      ),
    },
    {
      id: 'size',
      title: 'パラメータの数を比べる',
      body: (
        <>
          <p>左の表は、この分類器と LLM のパラメータ数の比較です。このミニ LLM でも分類器の約 {Math.round(llmParams / model.paramCount)} 倍、実際の LLM はさらに桁違いです。</p>
          <p>パラメータが多いほど複雑な規則を表せますが、学習に必要なデータと計算も増えます。「何の話か」程度の問題なら {model.paramCount} 個で十分、というのがこの編の分類器です。</p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="モデル"
      lede="語ごとの点数を足して、確率に直す。モデルの中身は点数表 1 枚です。"
      purpose={<>特徴量（数の並び）を、クラスごとの確率に変えるのがモデルの役目です。どう変えるかを決める数（重み）が、学習で調整される対象になります。</>}
      io={`${F} 個の回数 → ${C} 個の合計点 z → ${C} 個の確率`}
      terms={['logreg', 'parameter', 'softmax', 'projection', 'feature']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'score')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">上部の入力文</div>
                <WordChips words={ml.words} />
              </div>
              <Heatmap values={contrib} rows={contribRows.length} cols={C} cell={72} cellH={24} rowLabels={contribRows} rowLabelWidth={110} colLabels={names} showValues digits={2} highlightRows={[contribRows.length - 1]} title="語ごとの点数（W[c, f] · x_f）と合計" rowName="語" colName="クラス" tag="computed" />
            </div>
          )
        if (step.id === 'softmax')
          return (
            <div className="col">
              <Heatmap values={logits} rows={1} cols={C} cell={72} cellH={24} rowLabels={['合計 z']} rowLabelWidth={110} colLabels={names} showValues digits={2} title="合計点 z" tag="computed" />
              <div className="card" style={{ maxWidth: 420 }}>
                <ProbBars probs={ml.probs} title="softmax 後の確率" />
              </div>
            </div>
          )
        if (step.id === 'weights')
          return <Heatmap values={wT} rows={F} cols={C} cell={60} cellH={13} rowLabels={ds.vocab} rowLabelWidth={80} colLabels={names} highlightRows={used} title={`重み W（${F} 語 × ${C} クラス。太枠 = 入力文に出た語）`} rowName="語" colName="クラス" legend tag="param" />
        if (step.id === 'reason')
          return (
            <div className="row">
              {top.map((items, c) => (
                <div key={c} className="card">
                  <BarChart title={`${names[c]} の点数が高い語`} items={items.map((it) => ({ ...it, color: CLASS_COLORS[c] }))} min={0} width={250} labelWidth={70} format={(v) => fmt(v, 2)} />
                </div>
              ))}
            </div>
          )
        return (
          <div className="card" style={{ maxWidth: 640 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>モデル</th>
                  <th>パラメータ数</th>
                  <th>中身</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>この編の分類器</td>
                  <td className="num">{model.paramCount.toLocaleString()}</td>
                  <td className="muted">点数表 1 枚。全部画面に出せる</td>
                </tr>
                <tr>
                  <td>LLM 編のミニ LLM</td>
                  <td className="num">{llmParams.toLocaleString()}</td>
                  <td className="muted">埋め込み・注意機構・MLP を 2 層（LLM-8）</td>
                </tr>
                <tr>
                  <td>GPT-2 small（2019）</td>
                  <td className="num">約 1.2 億</td>
                  <td className="muted">同じ構造で 12 層・768 次元</td>
                </tr>
                <tr>
                  <td>最近の大型 LLM</td>
                  <td className="num">数千億〜</td>
                  <td className="muted">同じ構造をさらに大きく、深く</td>
                </tr>
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
