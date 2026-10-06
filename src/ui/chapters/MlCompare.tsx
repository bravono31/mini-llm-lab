import { Term } from '../../content/glossary'
import { totalSize } from '../../engine/params'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import type { ChapterProps } from './index'

export function MlCompare({ onNavigate }: ChapterProps) {
  const lab = useLab()
  const { ds, model } = useMl()
  const cfg = lab.params.config
  const F = ds.vocab.length
  const C = ds.classes.length

  const rows: [string, string, string][] = [
    ['解く問題', `文を ${C} つの話題に分ける（分類）`, '次のトークンを当てる → 繰り返して文章を生成'],
    ['正解（ラベル）', '人が 1 文ずつ付ける', '不要。文の続きが正解（自己教師あり）'],
    ['データの量', `${ds.texts.length} 文（ラベル付けの手間で増やしにくい）`, `このミニ版は ${lab.corpus.sentences.length} 文、実物はネット規模の文章`],
    ['特徴量', `人が作った辞書の ${F} 語の出現回数`, '埋め込みとして学習で決まる'],
    ['語順', '捨てる', '位置埋め込み＋注意機構で使う'],
    ['モデルの形', '点数表 1 枚（層なし）', `Transformer ブロックを ${cfg.nLayers} 層（実物は数十〜百層超）`],
    ['パラメータ数', model.paramCount.toLocaleString(), `${totalSize(lab.params.specs).toLocaleString()}（実物は数千億〜）`],
    ['学習', '勾配が 1 行の式。一瞬で終わる', '逆伝播と AdamW。実物は数か月'],
    ['出力', `${C} つのクラスの確率`, `語彙 ${cfg.vocabSize} 個の確率 → 1 つ選んで次へ`],
    ['評価', 'テストデータの正解率で明快', '正解が 1 つでなく、測り方自体が難しい'],
    ['判断の理由', '重みからそのまま読める', '重みから読むのは難しい'],
  ]

  const steps: Step[] = [
    {
      id: 'table',
      title: '並べて比べる',
      body: (
        <>
          <p>この編の分類器と、LLM 編のミニ LLM を項目ごとに並べました。</p>
          <p>
            大きな違いは 3 つにまとめられます。<strong>①正解を人が用意するか、データ自身から作るか</strong>。<strong>②特徴を人が設計するか、学習で決めるか</strong>。<strong>③決められた選択肢から選ぶか、文章を生み出すか</strong>。
          </p>
          <p>①のおかげで LLM は膨大なデータを使え、②のおかげでそのデータから人が思いつかない特徴まで学べ、その結果③のように自由な文章を書けるようになりました。</p>
        </>
      ),
    },
    {
      id: 'same',
      title: '同じところ：LLM の最後は「分類」',
      body: (
        <>
          <p>
            違いばかりではありません。<Term id="softmax">softmax</Term> で確率にする、正解の確率の −log を<Term id="loss">損失</Term>にする、<Term id="gradient">勾配</Term>の逆向きに重みを動かす。この 3 つは両者でまったく同じです。
          </p>
          <p>
            実は LLM の出力層（LLM-6）は、<strong>語彙 {cfg.vocabSize} クラスのロジスティック回帰</strong>そのものです。ベクトル × 重み → softmax で、「次のトークンはどれか」を {cfg.vocabSize} 択で分類しています。違うのは、入力のベクトルが人の数えた回数ではなく、Transformer が文脈から計算した {cfg.dModel} 次元のベクトルであることです。
          </p>
          <p>つまり LLM は「特徴を自分で作る巨大な前処理（Transformer）」＋「この編と同じ分類器」と見ることもできます。</p>
        </>
      ),
    },
    {
      id: 'when',
      title: 'どちらを使うか',
      body: (
        <>
          <p>LLM の方がなんでもできそうに見えますが、古典的な機械学習が向いている場面はいまも多くあります。売上の予測、不正な取引の検知、在庫の見積もりなど、<strong>表の形のデータ</strong>で、答えが数値や選択肢のものです。</p>
          <p>
            一方、LLM に分類をさせることもできます。「次の文はいきもの・しぜん・ひとのどれの話？」と頼めば、ラベル付きデータを 1 つも用意せずに答えます（ゼロショット）。大量の文章で言葉の使われ方を学んでいるからです。
          </p>
          <p>左の表は、選ぶときの目安です。</p>
        </>
      ),
    },
    {
      id: 'next',
      title: 'LLM 編へ',
      body: (
        <>
          <p>LLM 編では、LLM がどうやって「特徴を自分で作る」のかを、トークン化 → 埋め込み → 注意機構 → MLP の順に、実際の数値で見ていきます。</p>
          <p>この編で見た「損失・勾配・softmax」は LLM 編でもそのまま出てきます。▶▶ で LLM-1 へ進んでください。</p>
          <p>
            機械学習・深層学習・LLM・生成 AI の関係は、
            <button className="gloss-link" onClick={() => onNavigate('map')}>
              → 全体像の地図
            </button>
            にまとめています。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="LLM との違い"
      lede="同じ文集・同じ「確率で答える」仕組みでも、正解の作り方と特徴の作り方がまるで違います。"
      purpose={<>この編の分類器と LLM を並べると、LLM の何が新しく、何が昔ながらの機械学習と同じなのかが見えます。</>}
      io="この編の分類器 ⇔ LLM 編のミニ LLM"
      terms={['ml', 'supervised', 'selfsupervised', 'feature', 'deeplearning', 'llm', 'softmax', 'loss', 'gradient']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'table')
          return (
            <div className="card" style={{ maxWidth: 860 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th></th>
                    <th>この編の分類器（機械学習）</th>
                    <th>LLM 編のミニ LLM</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([k, a, b]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{a}</td>
                      <td>{b}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        if (step.id === 'same')
          return (
            <div className="col">
              <div className="pipeline">
                {[
                  ['文', '入力'],
                  [`人が数えた ${F} 個の回数`, '特徴（人が設計）'],
                  [`W（${C} × ${F}）＋ b`, '点数表'],
                  [`softmax → ${C} 択`, '話題の確率'],
                ].map(([name, shape], k) => (
                  <div key={k} className="pipe-node">
                    <span className="num">ML</span>
                    <span className="name">{name}</span>
                    <span className="shape">{shape}</span>
                  </div>
                ))}
              </div>
              <div className="pipeline">
                {[
                  ['文', '入力'],
                  [`Transformer が計算した ${cfg.dModel} 次元`, '特徴（学習で決まる）'],
                  [`埋め込み表（${cfg.vocabSize} × ${cfg.dModel}）`, '点数表（LLM-6）'],
                  [`softmax → ${cfg.vocabSize} 択`, '次のトークンの確率'],
                ].map(([name, shape], k) => (
                  <div key={k} className="pipe-node">
                    <span className="num">LLM</span>
                    <span className="name">{name}</span>
                    <span className="shape">{shape}</span>
                  </div>
                ))}
              </div>
              <p className="muted small">後ろ 2 段（点数表 → softmax）は同じ形。違いは、点数表に入れるベクトルを誰が作るか。</p>
            </div>
          )
        if (step.id === 'when')
          return (
            <div className="card" style={{ maxWidth: 760 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>こんなとき</th>
                    <th>向いているもの</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>表の形のデータ（売上、センサー値、取引履歴）</td>
                    <td>古典的な機械学習（回帰、決定木など）</td>
                  </tr>
                  <tr>
                    <td>判断の理由を説明する必要がある（審査、医療）</td>
                    <td>古典的な機械学習（重みが読める）</td>
                  </tr>
                  <tr>
                    <td>大量に・安く・速く処理したい</td>
                    <td>古典的な機械学習（計算が軽い）</td>
                  </tr>
                  <tr>
                    <td>画像・音声の認識</td>
                    <td>深層学習（CNN など。特徴を自分で学ぶ）</td>
                  </tr>
                  <tr>
                    <td>文章を書く・要約する・対話する</td>
                    <td>LLM</td>
                  </tr>
                  <tr>
                    <td>ラベル付きデータがない分類</td>
                    <td>LLM に頼む（ゼロショット）</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )
        return (
          <div className="pipeline">
            {[
              ['tokenize', 'LLM-2', 'トークン化', '区切りをデータから決める'],
              ['embed', 'LLM-3', '埋め込み', '特徴を学習で決める'],
              ['attention', 'LLM-4', '注意機構', '語どうしの関係を使う'],
              ['output', 'LLM-6', '出力', '語彙の数だけの分類'],
              ['train', 'LLM-7', '学習', 'ラベルなしで学ぶ'],
            ].map(([id, n, name, shape]) => (
              <button key={n} className="pipe-node" onClick={() => onNavigate(id)}>
                <span className="num">{n}</span>
                <span className="name">{name}</span>
                <span className="shape">{shape}</span>
              </button>
            ))}
          </div>
        )
      }}
    </ChapterLayout>
  )
}
