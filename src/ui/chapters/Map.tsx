import { Fragment, type ReactNode } from 'react'
import { Term } from '../../content/glossary'
import { totalSize } from '../../engine/params'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import type { ChapterProps } from './index'

interface TreeCell {
  name: string
  note: string
  /** where this level is covered in the app */
  link?: { id: string; label: string }
  /** the model this app actually builds */
  here?: boolean
}

/** one row per rank; a row is either shared by both lineages or split into [shallow, deep] */
const TREE: { rank: string; shared?: TreeCell & { other?: string; split?: boolean }; pair?: [TreeCell, TreeCell] }[] = [
  { rank: '総称', shared: { name: 'AI（人工知能）', note: '人の知的な振る舞いをコンピュータで実現する技術' } },
  { rank: '分野', shared: { name: '機械学習', note: 'ルールを書かず、データから規則（重み）を学ぶ', other: 'ほかの枝：ルールを人が書く AI' } },
  {
    rank: 'モデルの種類',
    shared: { name: 'ニューラルネットワーク', note: '入力 × 重みの和を次へ渡す計算を、層にしたもの', link: { id: 'overview', label: 'LLM-1' }, other: 'ほかの枝：決定木・SVM など', split: true },
  },
  {
    rank: '深さ',
    pair: [
      { name: '浅い（中間層なし）', note: '重みを 1 回掛けるだけ' },
      { name: '深い（深層学習）', note: '中間層を重ね、特徴も学ぶ' },
    ],
  },
  {
    rank: 'モデル構造',
    pair: [
      { name: '線形モデル（ロジスティック回帰）', note: '特徴は人が設計した単語の回数', link: { id: 'ml-model', label: 'ML-4' } },
      { name: 'Transformer', note: '注意機構と MLP を積んだ構造', link: { id: 'attention', label: 'LLM-4' } },
    ],
  },
  {
    rank: '何をするモデルか',
    pair: [
      { name: 'テキスト分類器', note: '文 → 3 つの話題のどれか', link: { id: 'ml-overview', label: 'ML 編' }, here: true },
      { name: '言語モデル（LLM）', note: '文 → 次のトークン → 文章を生成', link: { id: 'overview', label: 'LLM 編' }, here: true },
    ],
  },
]

type Cell = { text: string; gen?: boolean; link?: { id: string; label: string } }

const AXES_COLS = ['言語（NLP）', '画像', '音声']
const AXES_ROWS: { method: string; cells: Cell[] }[] = [
  { method: 'ルールを人が書く', cells: [{ text: '辞書・文法ルールによる翻訳や判定' }, { text: '色や輪郭のしきい値で検出' }, { text: '—' }] },
  { method: '古典的な機械学習（ロジスティック回帰など）', cells: [{ text: '話題分類、迷惑メール判定', link: { id: 'ml-overview', label: 'ML 編' } }, { text: '人が設計した特徴＋分類器' }, { text: '人が設計した特徴＋分類器' }] },
  { method: '深層学習（CNN・RNN など）', cells: [{ text: '翻訳、感情分析' }, { text: '物体認識、顔認識' }, { text: '音声認識' }] },
  { method: '深層学習（Transformer）', cells: [{ text: 'LLM：文章の生成・要約・対話', gen: true, link: { id: 'overview', label: 'LLM 編' } }, { text: '画像認識（ViT）、画像の説明文' }, { text: '音声認識、音声合成', gen: true }] },
  { method: '深層学習（拡散モデルなど）', cells: [{ text: '—' }, { text: '画像生成', gen: true }, { text: '音楽・音声の生成', gen: true }] },
]

const HIERARCHY: [string, string, string][] = [
  ['機械学習・自然言語処理・生成 AI を同じ「分野」の段に並べる', '機械学習は「手法」、NLP は「扱う対象（言語）」、生成 AI は「用途（作り出す）」。軸が違うので、入れ子ではなく交差として描く', '系統樹 ＋ 手法 × 対象の表'],
  ['深層学習の下にニューラルネットワーク', 'ニューラルネットワークはモデルの種類で、深層学習は「層を深くした NN を学習させる手法」。NN の方が外側', '機械学習 ⊃ NN ⊃ 深層学習'],
  ['「機械学習」と「LLM」を比べる', '機械学習は LLM を含む広い分野なので、段がそろわない（種と、それを含む門を比べるようなもの）', '同じ段の「浅い NN の分類器」と「深い NN の LLM」を比べる'],
  ['Transformer は「アルゴリズム」', '学習の手順（アルゴリズム）は勾配降下で、ほかのモデルと共通。Transformer は計算のつなぎ方＝モデル構造（アーキテクチャ）', 'モデル構造として入れ子の 1 段に'],
  ['LLM は深層学習の具体例', 'そのとおり。加えて、対象の軸では NLP、用途の軸では生成 AI にも属する、3 つの軸の交点', '表の「Transformer × 言語」のマス'],
]

export function MapChapter({ onNavigate }: ChapterProps) {
  const lab = useLab()
  const ml = useMl()
  const cfg = lab.params.config
  const link = (l?: { id: string; label: string }) =>
    l && (
      <button className="gloss-link" onClick={() => onNavigate(l.id)}>
        → {l.label}
      </button>
    )

  const cell = (c: TreeCell, extra?: ReactNode) => (
    <>
      <div className="map-nest-head">
        <span className="map-nest-name">{c.name}</span>
        {link(c.link)}
      </div>
      <div className="map-nest-note">{c.note}</div>
      {extra}
    </>
  )

  const tree = (
    <div className="map-tree">
      <div className="map-tree-rank" />
      <div className="map-tree-head">ML 編の枝</div>
      <div className="map-tree-head">LLM 編の枝</div>
      {TREE.map((r) => (
        <Fragment key={r.rank}>
          <div className="map-tree-rank">{r.rank}</div>
          {r.shared ? (
            <div className={'map-tree-cell shared' + (r.shared.split ? ' split' : '')}>
              {cell(r.shared, r.shared.other && <div className="map-tree-other">{r.shared.other}</div>)}
            </div>
          ) : (
            r.pair!.map((c) => (
              <div key={c.name} className={'map-tree-cell' + (c.here ? ' here' : '')}>
                {cell(c)}
              </div>
            ))
          )}
        </Fragment>
      ))}
    </div>
  )

  const lineage = (
    <div className="card" style={{ maxWidth: 820 }}>
      <table className="data map-lineage">
        <thead>
          <tr>
            <th>段</th>
            <th>ML 編の分類器</th>
            <th>LLM 編の mini LLM</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>分野</td>
            <td colSpan={2} className="shared">
              機械学習（データから重みを学ぶ）
            </td>
          </tr>
          <tr>
            <td>モデルの種類</td>
            <td colSpan={2} className="shared">
              ニューラルネットワーク ← ここで分かれる
            </td>
          </tr>
          <tr>
            <td>深さ</td>
            <td>浅い（中間層なし）</td>
            <td>深い（深層学習、ブロック {cfg.nLayers} 層）</td>
          </tr>
          <tr>
            <td>モデル構造</td>
            <td>線形モデル（ロジスティック回帰）</td>
            <td>Transformer</td>
          </tr>
          <tr>
            <td>何をするモデルか</td>
            <td>テキスト分類器（3 択）</td>
            <td>言語モデル（次のトークン {cfg.vocabSize} 択 → 生成）</td>
          </tr>
          <tr>
            <td>このアプリの実物</td>
            <td>パラメータ {ml.model.paramCount.toLocaleString()} 個</td>
            <td>パラメータ {totalSize(lab.params.specs).toLocaleString()} 個</td>
          </tr>
        </tbody>
      </table>
      <div className="row" style={{ marginTop: 14 }}>
        <div className="grow">
          <div className="card-title">分かれる前から共通（祖先から受け継いだもの）</div>
          <div className="small">重み × 入力の和 ／ softmax で確率 ／ 交差エントロピーの損失 ／ 勾配で重みを更新</div>
        </div>
        <div className="grow">
          <div className="card-title">分かれた後に違うもの</div>
          <div className="small">中間層と非線形の有無 ／ 特徴を人が作るか学ぶか ／ 語順を使うか ／ 選ぶか生み出すか</div>
        </div>
      </div>
    </div>
  )

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
      title: '手法の系統樹：2 つのモデルはどこで分かれるか',
      body: (
        <>
          <p>
            <Term id="ai">AI</Term>・<Term id="ml">機械学習</Term>・<Term id="nn">ニューラルネットワーク</Term>・<Term id="deeplearning">深層学習</Term>・<Term id="llm">LLM</Term> は、<strong>手法として</strong>入れ子の関係にあります。左の図の上の段ほど広い概念で、下の段はその一部です。
          </p>
          <p>
            このアプリの 2 つのモデルは、どちらもいちばん下の段（朱の枠）にいます。ML 編の分類器（ロジスティック回帰）は、計算としては<strong>中間層のないニューラルネットワーク</strong>と同じものです。そのため 2 つは<strong>ニューラルネットワークの段で枝分かれした親戚</strong>で、浅い枝の先が分類器、深い枝（深層学習 → <Term id="transformer">Transformer</Term>）の先が LLM です。
          </p>
          <p>
            「機械学習と LLM を比べる」と言うと、LLM を含む広い分野と LLM を比べることになり、段がそろいません。この図の同じ段にいる 2 つを比べるのが、このアプリでの対比です。
          </p>
          <p>各段の「→」から、その段を扱っている章へ移動できます。</p>
        </>
      ),
    },
    {
      id: 'lineage',
      title: '同じ段どうしで並べる',
      body: (
        <>
          <p>系統樹の 2 本の枝を、段ごとに横に並べました。上の 2 段（機械学習・ニューラルネットワーク）は共通で、3 段目から分かれます。</p>
          <p>
            分かれる前から持っている性質は、両方のモデルに共通です。重みを掛けて足す、softmax で確率にする、交差エントロピーを勾配で減らす。ML-5 と LLM-7 の学習が同じ式なのはこのためです。
          </p>
          <p>分かれた後の違い（中間層があるか、特徴を学ぶか、語順を使うか、文章を生み出すか）が、ML-7 で比べている「LLM の新しさ」にあたります。</p>
        </>
      ),
    },
    {
      id: 'axes',
      title: 'もう 1 つの軸：何を扱い、何に使うか',
      body: (
        <>
          <p>
            「<Term id="nlp">自然言語処理</Term>」や「<Term id="genai">生成 AI</Term>」は、系統樹のどこにも入りません。手法の名前ではなく、<strong>扱う対象</strong>（言語）や<strong>用途</strong>（新しく作り出す）による呼び名だからです。
          </p>
          <p>
            左の表のように、手法と対象を 2 つの軸にすると整理できます。同じ「言語」の問題でも、時代とともにルール → 古典的な機械学習 → 深層学習と手法が変わってきました。ML 編の話題分類と LLM 編の LLM は、同じ列の違う行にいます。なおロジスティック回帰は、特徴を人が作る点で「古典的な機械学習」の代表であり、計算の形では「中間層のないニューラルネットワーク」でもある、両者の境目にいる手法です。
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
        if (step.id === 'nest') return tree
        if (step.id === 'lineage') return lineage
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
