import type { ReactNode } from 'react'
import { Term } from '../../content/glossary'
import { ChapterLayout } from '../shell/ChapterLayout'
import type { Step } from '../state/useStepper'
import type { ChapterProps } from './index'

interface Nest {
  name: string
  note: string
  /** where this level is covered in the app */
  link?: { id: string; label: string }
}

// outermost first: each level contains the next
const NEST: Nest[] = [
  { name: 'AI（人工知能）', note: '人の知的な振る舞いをコンピュータで実現する技術の総称。ルールを人が書く方法も含む' },
  { name: '機械学習', note: 'ルールを書かず、データから規則（重み）を学ぶ', link: { id: 'ml-overview', label: 'ML 編' } },
  { name: 'ニューラルネットワーク', note: '重み付きの線でつないだノードを層に重ねた計算モデル', link: { id: 'overview', label: 'LLM-1' } },
  { name: '深層学習', note: '層を深くしたニューラルネットワークを学習させる手法。特徴も学ぶ', link: { id: 'ml-compare', label: 'ML-7' } },
  { name: 'Transformer', note: '注意機構と MLP を積んだモデル構造（アーキテクチャ）', link: { id: 'attention', label: 'LLM-4' } },
  { name: 'LLM', note: 'Transformer を膨大な文章で「次の語当て」学習させたもの', link: { id: 'overview', label: 'LLM 編' } },
]

type Cell = { text: string; gen?: boolean; link?: { id: string; label: string } }

const AXES_COLS = ['言語（NLP）', '画像', '音声']
const AXES_ROWS: { method: string; cells: Cell[] }[] = [
  { method: 'ルールを人が書く', cells: [{ text: '辞書・文法ルールによる翻訳や判定' }, { text: '色や輪郭のしきい値で検出' }, { text: '—' }] },
  { method: '古典的な機械学習', cells: [{ text: '話題分類、迷惑メール判定', link: { id: 'ml-overview', label: 'ML 編' } }, { text: '人が設計した特徴＋分類器' }, { text: '人が設計した特徴＋分類器' }] },
  { method: '深層学習（CNN・RNN など）', cells: [{ text: '翻訳、感情分析' }, { text: '物体認識、顔認識' }, { text: '音声認識' }] },
  { method: '深層学習（Transformer）', cells: [{ text: 'LLM：文章の生成・要約・対話', gen: true, link: { id: 'overview', label: 'LLM 編' } }, { text: '画像認識（ViT）、画像の説明文' }, { text: '音声認識、音声合成', gen: true }] },
  { method: '深層学習（拡散モデルなど）', cells: [{ text: '—' }, { text: '画像生成', gen: true }, { text: '音楽・音声の生成', gen: true }] },
]

const HIERARCHY: [string, string, string][] = [
  ['機械学習・自然言語処理・生成 AI を同じ「分野」の段に並べる', '機械学習は「手法」、NLP は「扱う対象（言語）」、生成 AI は「用途（作り出す）」。軸が違うので、入れ子ではなく交差として描く', '入れ子の図 ＋ 手法 × 対象の表'],
  ['深層学習の下にニューラルネットワーク', 'ニューラルネットワークはモデルの種類で、深層学習は「層を深くした NN を学習させる手法」。NN の方が外側', '機械学習 ⊃ NN ⊃ 深層学習'],
  ['Transformer は「アルゴリズム」', '学習の手順（アルゴリズム）は勾配降下で、ほかのモデルと共通。Transformer は計算のつなぎ方＝モデル構造（アーキテクチャ）', 'モデル構造として入れ子の 1 段に'],
  ['LLM は深層学習の具体例', 'そのとおり。加えて、対象の軸では NLP、用途の軸では生成 AI にも属する、3 つの軸の交点', '表の「Transformer × 言語」のマス'],
]

export function MapChapter({ onNavigate }: ChapterProps) {
  const link = (l?: { id: string; label: string }) =>
    l && (
      <button className="gloss-link" onClick={() => onNavigate(l.id)}>
        → {l.label}
      </button>
    )

  const nested = (k: number): ReactNode => {
    const n = NEST[k]
    return (
      <div className={'map-nest' + (k === NEST.length - 1 ? ' innermost' : '')}>
        <div className="map-nest-head">
          <span className="map-nest-name">{n.name}</span>
          {link(n.link)}
        </div>
        <div className="map-nest-note">{n.note}</div>
        {k < NEST.length - 1 && nested(k + 1)}
      </div>
    )
  }

  const axes = (
    <div className="card" style={{ maxWidth: 860 }}>
      <table className="data map-axes">
        <thead>
          <tr>
            <th>手法 ＼ 扱う対象</th>
            {AXES_COLS.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {AXES_ROWS.map((r) => (
            <tr key={r.method}>
              <td>{r.method}</td>
              {r.cells.map((c, i) => (
                <td key={i} className={c.gen ? 'gen' : ''}>
                  {c.text}
                  {c.gen && <span className="gen-mark">生成</span>}
                  {link(c.link)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        縦が手法（下ほど新しい）、横が扱う対象。左の列が<Term id="nlp">自然言語処理</Term>。朱の枠が<Term id="genai">生成 AI</Term>：手法でも対象でもなく「作り出す」という用途でくくった呼び名なので、表を斜めに横切ります。
      </p>
    </div>
  )

  const steps: Step[] = [
    {
      id: 'nest',
      title: '手法の入れ子：AI ⊃ 機械学習 ⊃ … ⊃ LLM',
      body: (
        <>
          <p>
            <Term id="ai">AI</Term>・<Term id="ml">機械学習</Term>・<Term id="deeplearning">深層学習</Term>・<Term id="llm">LLM</Term> は、<strong>手法として</strong>入れ子の関係にあります。左の図の外側ほど広い概念で、内側はその一部です。
          </p>
          <p>
            AI には、人がルールを書く方法と、データから学ぶ機械学習があります。機械学習のうち、<Term id="nn">ニューラルネットワーク</Term>を使うものがあり、その層を深くしたのが深層学習。深層学習のモデル構造の 1 つが <Term id="transformer">Transformer</Term> で、それを膨大な文章で学習させたのが LLM です。
          </p>
          <p>各段の「→」から、その段を扱っている章へ移動できます。</p>
        </>
      ),
    },
    {
      id: 'axes',
      title: 'もう 1 つの軸：何を扱い、何に使うか',
      body: (
        <>
          <p>
            「<Term id="nlp">自然言語処理</Term>」や「<Term id="genai">生成 AI</Term>」は、入れ子の図のどこにも入りません。手法の名前ではなく、<strong>扱う対象</strong>（言語）や<strong>用途</strong>（新しく作り出す）による呼び名だからです。
          </p>
          <p>
            左の表のように、手法と対象を 2 つの軸にすると整理できます。同じ「言語」の問題でも、時代とともにルール → 古典的な機械学習 → 深層学習と手法が変わってきました。ML 編の話題分類と LLM 編の LLM は、同じ列の違う行にいます。
          </p>
        </>
      ),
    },
    {
      id: 'llm',
      title: 'LLM はどこにいるか',
      body: (
        <>
          <p>LLM は 3 つの軸で同時に位置づけられます。</p>
          <p>
            <strong>手法</strong>：深層学習のうち Transformer を使うもの。<strong>対象</strong>：言語（自然言語処理）。<strong>用途</strong>：文章を作り出す（生成 AI）。
          </p>
          <p>左の表は、よくある並べ方（概念 → 手法 → 具体モデルの 3 段）と、この地図での整理の違いです。「どの段か」より「どの軸か」で考えると、言葉どうしの関係が混乱しにくくなります。</p>
        </>
      ),
    },
    {
      id: 'route',
      title: 'このアプリの歩き方',
      body: (
        <>
          <p>
            <strong>機械学習編</strong>（ML-1〜7）では、いちばん素朴な機械学習で「データ → 特徴量 → モデル → 学習 → 評価」の流れを見ます。<strong>LLM 編</strong>（LLM-1〜9）では、同じ文集を使うミニ LLM の中身を、実際の数値で 1 段ずつ見ます。
          </p>
          <p>ML 編を先に読むと、LLM の何が新しいのかが分かりやすくなります（ML-7 で両者を比べます）。LLM 編から読み始めてもかまいません。</p>
          <p>深層学習一般（画像の CNN など）や、画像生成の拡散モデルは、いまのところこのアプリでは扱っていません。</p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="AI・機械学習・LLM の地図"
      lede="よく一緒に出てくる言葉の関係を、手法の入れ子と、扱う対象・用途の 2 つの軸で整理します。"
      purpose={<>「機械学習」「深層学習」「LLM」「生成 AI」は、それぞれ違う切り口の言葉です。関係を先に押さえておくと、各編がどこを扱っているかが分かります。</>}
      terms={['ai', 'ml', 'nn', 'deeplearning', 'transformer', 'llm', 'nlp', 'genai']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'nest') return <div style={{ maxWidth: 760 }}>{nested(0)}</div>
        if (step.id === 'axes') return axes
        if (step.id === 'llm')
          return (
            <div className="card" style={{ maxWidth: 860 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>よくある並べ方</th>
                    <th>実際の関係</th>
                    <th>この地図では</th>
                  </tr>
                </thead>
                <tbody>
                  {HIERARCHY.map(([a, b, c]) => (
                    <tr key={a}>
                      <td>{a}</td>
                      <td className="muted">{b}</td>
                      <td>{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        return (
          <div className="pipeline">
            {[
              ['ml-overview', 'ML 編', '機械学習の基本', 'ML-1〜7'],
              ['ml-compare', 'ML-7', 'LLM との違い', '両者を並べて比べる'],
              ['overview', 'LLM 編', 'LLM の中身', 'LLM-1〜9'],
            ].map(([id, n, name, shape]) => (
              <button key={id} className="pipe-node" onClick={() => onNavigate(id)}>
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
