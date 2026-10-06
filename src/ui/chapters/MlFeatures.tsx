import { useMemo } from 'react'
import { Term } from '../../content/glossary'
import { featurize, segment } from '../../engine/classifier'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import type { ChapterProps } from './index'
import { ProbBars, WordChips } from './MlParts'

export function MlFeatures(_: ChapterProps) {
  const ml = useMl()
  const { ds, model } = ml
  const F = ds.vocab.length
  const N = ds.texts.length
  const ja = ds.data.lang === 'ja'
  const used = ds.vocab.flatMap((_, f) => (ml.x[f] ? [f] : []))

  const matrix = useMemo(() => {
    const m = new Float64Array(N * F)
    ds.X.forEach((x, i) => m.set(x, i * F))
    return m
  }, [ds, N, F])
  const nonzero = matrix.reduce((a, v) => a + (v ? 1 : 0), 0)

  const [sa, sb] = ja ? ['ねこはいぬをみる', 'いぬはねこをみる'] : ['the cat sees the dog', 'the dog sees the cat']
  const xa = featurize(sa, ds.data, ds.vocab)
  const xb = featurize(sb, ds.data, ds.vocab)
  const pairCols = ds.vocab.flatMap((_, f) => (xa[f] || xb[f] ? [f] : []))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const pa = useMemo(() => model.probs(xa), [model, sa, ml.version])

  const steps: Step[] = [
    {
      id: 'segment',
      title: '単語に区切る（辞書は人が作る）',
      body: (
        <>
          <p>文字列のままでは計算できないので、まず文を単語に区切ります。{ja ? 'ひらがなの文には区切りの空白がないので、' : ''}このアプリの作者が作った<strong>{F} 語の辞書</strong>{ja ? 'と照らし合わせて区切ります' : 'を使います（英語は空白で区切るだけ）'}。</p>
          <p>
            「{ja ? 'は・が・を' : 'the・is・in'}」のような語はどの話題の文にも出てくるので、手がかりにならないと判断して<strong>捨てます</strong>（薄い表示）。辞書にない語（点線）は無視されます。上の入力文に辞書にない語を入れて試してみてください。
          </p>
          <p>
            LLM 編のトークン化（LLM-2）も文を区切りますが、単位（トークン）は <Term id="bpe">BPE</Term> がデータから<strong>自動で</strong>決め、助詞も捨てません。区切り方を人が決めるか、データから決めるか。ここに最初の違いがあります。
          </p>
        </>
      ),
    },
    {
      id: 'count',
      title: '数える：Bag-of-Words',
      body: (
        <>
          <p>
            区切った単語を、辞書の {F} 語それぞれについて<strong>何回出たか</strong>数えます。これで 1 文が {F} 個の数の並び（<Term id="vector">ベクトル</Term>）になります。モデルへの入力はこのベクトルで、文そのものではありません。
          </p>
          <p>
            このように、生のデータを計算できる数に直したものを<Term id="feature">特徴量</Term>、この数え方を <Term id="bow">Bag-of-Words</Term>（単語の袋）と呼びます。文を袋に入れて振ったように、<strong>語の順番は消えます</strong>。
          </p>
          <p>
            LLM-9（RAG）の検索でも、同じ考え方でトークンの回数を数えていました。
          </p>
        </>
      ),
      formula: `x = (辞書の語 1 の回数, 語 2 の回数, …, 語 ${F} の回数)`,
    },
    {
      id: 'matrix',
      title: '全データを並べると、ほとんど 0',
      body: (
        <>
          <p>
            {N} 文すべてを特徴量にして 1 行ずつ並べると、左の {N} × {F} の表になります。0 でないのは {nonzero} マスだけ（全体の {((nonzero / (N * F)) * 100).toFixed(1)}%）。1 文に出てくる語は辞書のごく一部なので、こうした<strong>ほとんど 0 の行列</strong>（疎行列）になるのが普通です。
          </p>
          <p>モデルが学習で見るのは、この表と各行のラベルだけです。</p>
        </>
      ),
    },
    {
      id: 'order',
      title: '語順を捨てた代償',
      body: (
        <>
          <p>
            「{sa}」と「{sb}」は意味が違う文ですが、出てくる語と回数は同じなので、特徴量は<strong>まったく同じベクトル</strong>になります。モデルにはこの 2 文を区別する手段がありません。
          </p>
          <p>
            今回の「何の話か」という問題なら、語順はほとんど要らないので困りません。でも「だれが・だれを」が大事な問題や、次の語を当てる問題には使えません。
          </p>
          <p>
            LLM は、トークンに<Term id="positional">位置埋め込み</Term>を足して「何番目か」を残し（LLM-3）、<Term id="attention">注意機構</Term>で語どうしの関係を計算します（LLM-4）。語順を捨てないことが、文章を生成できる前提になっています。
          </p>
        </>
      ),
    },
    {
      id: 'design',
      title: '特徴は人が設計する、LLM は自分で学ぶ',
      body: (
        <>
          <p>ここまでで人が決めたことを並べると、左の表になります。どれも「何が手がかりになりそうか」という人の見立てで、これを<strong>特徴量エンジニアリング</strong>と呼びます。古典的な機械学習の性能は、この設計の良し悪しに大きく左右されます。</p>
          <p>
            LLM（<Term id="deeplearning">深層学習</Term>）はこの部分も学習に任せます。トークンを 16 次元の<Term id="embedding">埋め込み</Term>ベクトルにする表も、その後の注意機構や MLP の変換も、すべて学習で決まる重みです。人は「数える語」を選ばず、<strong>どんな特徴が役に立つかまで、データから学ばせます</strong>。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="特徴量"
      lede="文を、計算できる数の並びに直します。何を数えるかは人が決めます。"
      purpose={<>モデルは数しか扱えません。文のどの性質を数として取り出すか（特徴量）が、モデルに見えるもののすべてになります。</>}
      io={`文字列 → 単語の列 → ${F} 個の回数（ベクトル）`}
      terms={['feature', 'bow', 'vector', 'bpe', 'positional', 'attention', 'embedding', 'deeplearning']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'segment')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">上部の入力文を区切ると</div>
                <WordChips words={ml.words} />
              </div>
              <div className="card">
                <div className="card-title">辞書（{F} 語）</div>
                <div className="mono small" style={{ lineHeight: 1.9 }}>
                  {ds.vocab.join('　')}
                </div>
                <div className="card-title" style={{ marginTop: 12 }}>
                  捨てる語
                </div>
                <div className="mono small">{ds.data.stop.join('　')}</div>
              </div>
            </div>
          )
        if (step.id === 'count')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">上部の入力文</div>
                <WordChips words={ml.words} />
              </div>
              {used.length > 0 ? (
                <VectorStrip values={used.map((f) => ml.x[f])} colLabels={used.map((f) => ds.vocab[f])} mode="sequential" max={2} digits={0} cell={44} title="0 でない列だけ" tag="computed" />
              ) : (
                <div className="card muted small">辞書にある語がないので、すべて 0 のベクトルになります。</div>
              )}
              <Heatmap values={ml.x} rows={1} cols={F} cell={11} mode="sequential" max={2} title={`ベクトル全体（${F} 列。色の付いた列以外はすべて 0）`} colName="語" tag="computed" />
            </div>
          )
        if (step.id === 'matrix')
          return <Heatmap values={matrix} rows={N} cols={F} cell={9} cellH={11} mode="sequential" max={2} rowLabels={ds.texts} rowLabelWidth={ja ? 150 : 210} title={`全 ${N} 文の特徴量（1 行 = 1 文、1 列 = 辞書の 1 語）`} rowName="文" colName="語" tag="computed" />
        if (step.id === 'order')
          return (
            <div className="col">
              <div className="row">
                <div className="card">
                  <div className="card-title">{sa}</div>
                  <WordChips words={segment(sa, ds.data)} />
                </div>
                <div className="card">
                  <div className="card-title">{sb}</div>
                  <WordChips words={segment(sb, ds.data)} />
                </div>
              </div>
              <Heatmap
                values={Float64Array.from([...pairCols.map((f) => xa[f]), ...pairCols.map((f) => xb[f])])}
                rows={2}
                cols={pairCols.length}
                cell={44}
                mode="sequential"
                max={2}
                showValues
                digits={0}
                rowLabels={[sa, sb]}
                rowLabelWidth={ja ? 140 : 170}
                colLabels={pairCols.map((f) => ds.vocab[f])}
                title="2 文の特徴量（0 でない列）：同じ"
                tag="computed"
              />
              <div className="card" style={{ maxWidth: 420 }}>
                <ProbBars probs={pa} title="分類の結果も当然同じ（どちらの文でも）" width={360} />
              </div>
            </div>
          )
        return (
          <div className="card" style={{ maxWidth: 720 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>決めたこと</th>
                  <th>この編（人が決めた）</th>
                  <th>LLM 編</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>区切りの単位</td>
                  <td>人が作った {F} 語の辞書</td>
                  <td>BPE がデータから自動で作る（LLM-2）</td>
                </tr>
                <tr>
                  <td>捨てる語</td>
                  <td>{ds.data.stop.join('・')}</td>
                  <td>捨てない</td>
                </tr>
                <tr>
                  <td>数にする方法</td>
                  <td>出現回数を数える</td>
                  <td>学習した埋め込み表を引く（LLM-3）</td>
                </tr>
                <tr>
                  <td>語順</td>
                  <td>捨てる</td>
                  <td>位置埋め込みで残す</td>
                </tr>
                <tr>
                  <td>語の関係</td>
                  <td>見ない</td>
                  <td>注意機構で計算する（LLM-4）</td>
                </tr>
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
