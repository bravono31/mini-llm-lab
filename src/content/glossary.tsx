import type { ReactNode } from 'react'
import { useNav } from '../ui/state/NavProvider'

export interface GlossaryEntry {
  term: string
  /** one or two sentences, plain language */
  short: string
  /** chapter where it is explained in depth */
  chapter?: string
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  llm: { term: 'LLM（大規模言語モデル）', short: '大量の文章から「次に来る語」を当てるように学習したモデル。ChatGPT などの中身。ここでは同じ構造を極小にしたものを扱う。', chapter: 'overview' },
  transformer: { term: 'Transformer', short: '注意機構と MLP を交互に積んだニューラルネットワークの構造。現在の LLM のほぼすべてがこれ。', chapter: 'overview' },
  gpt: { term: 'GPT', short: 'Transformer のうち「前の語だけを見て次の語を予測する」型（decoder-only）。このアプリのモデルも GPT-2 と同じ構造。', chapter: 'overview' },
  parameter: { term: 'パラメータ（重み）', short: '学習で値が決まる数値。行列やベクトルとして保存され、モデルの「知識」はすべてここにある。初期値は乱数。', chapter: 'params' },
  token: { term: 'トークン', short: 'モデルが扱う最小単位の文字列。文字 1 つのこともあれば「ねこ」「 the」のような語のこともある。', chapter: 'tokenize' },
  tokenizer: { term: 'トークナイザ', short: '文字列をトークンの列に切り分け、各トークンを整数 ID に変換する仕組み。', chapter: 'tokenize' },
  bpe: { term: 'BPE（Byte Pair Encoding）', short: 'よく隣り合う 2 つのトークンを 1 つに結合していく、語彙の作り方。GPT 系のトークナイザはこれの一種。', chapter: 'tokenize' },
  vocab: { term: '語彙（vocabulary）', short: 'モデルが知っているトークンの一覧表。各トークンに 0 から順の ID が付く。このモデルは日本語 96 個 / 英語 64 個。', chapter: 'tokenize' },
  id: { term: 'トークン ID', short: '語彙表での行番号。モデルへの入力は文字ではなくこの整数の列。', chapter: 'tokenize' },
  eos: { term: '⟨eos⟩', short: 'end of sentence。文の終わりを表す特別なトークン（ID 0）。生成でこれが出たら止まる。', chapter: 'tokenize' },
  vector: { term: 'ベクトル / 行列', short: 'ベクトルは数を一列に並べたもの、行列は表の形に並べたもの。モデルの中ではトークン 1 つが 1 本のベクトル、文全体が 1 つの行列になる。', chapter: 'embed' },
  dimension: { term: '次元（d_model）', short: 'トークンを表すベクトルの長さ。このモデルは 16。GPT-2 small は 768。大きいほど表現力が上がるが計算も増える。', chapter: 'embed' },
  embedding: { term: '埋め込み（embedding）', short: 'トークン ID をベクトルに置き換えること、またはその表。似た使われ方の語が近いベクトルになるよう学習される。', chapter: 'embed' },
  positional: { term: '位置埋め込み', short: '「何番目か」を表すベクトル。注意機構は順番を区別できないので、トークンの埋め込みに足して順番の情報を与える。', chapter: 'embed' },
  residual: { term: '残差ストリーム / 残差接続', short: '各層の出力を入力に「足す」つなぎ方。文全体の T×16 行列が層を通るたびに情報を書き足されていく、その流れを残差ストリームと呼ぶ。', chapter: 'mlp' },
  layernorm: { term: 'LayerNorm', short: 'ベクトルを平均 0・分散 1 に揃えてから、学習した γ を掛け β を足す正規化。各層の入口にあり計算を安定させる。', chapter: 'attention' },
  attention: { term: '注意機構（attention）', short: '各トークンが文中の他のトークンとの関係の強さを計算し、関係が強い相手の情報を取り込む仕組み。Transformer の核心。', chapter: 'attention' },
  qkv: { term: 'Query / Key / Value', short: '注意機構で各トークンから作る 3 種のベクトル。Q は「何を探すか」、K は「自分は何か」、V は「相手に渡す中身」。Q と K の内積で関係の強さを測り、V を重み付きで集める。', chapter: 'attention' },
  head: { term: 'ヘッド（multi-head attention）', short: '注意機構を小さな注意機構に分けて並列に走らせる仕組み。1 つの注意は位置ごとに 1 通りの「誰を見るか」しか決められないが、ヘッドを分けると「直前の語を見る」「主語を見る」のように複数の関係を同時に扱える。このモデルは 16 次元を 8 次元 × 2 ヘッドに分割。', chapter: 'attention' },
  projection: { term: '射影（projection）', short: 'ベクトルに行列を掛けて、別の次元数・別の座標系に写すこと。Q・K・V を作る W_qkv、ヘッドの出力を混ぜる W_o、MLP の W₁・W₂ はすべて射影で、違いは「何のために」と「行列の形」だけ。', chapter: 'attention' },
  softmax: { term: 'softmax', short: '数の列を「合計 1 の確率」に変える関数。大きい値ほど大きな確率になる。注意重みと出力確率の両方で使う。', chapter: 'attention' },
  mask: { term: '因果マスク', short: '各位置が自分より後ろのトークンを見られないようにする仕組み。「次を当てる」学習に必要。', chapter: 'attention' },
  layer: { term: '層 / ブロック', short: '注意機構 + MLP のひとまとまり（Transformer ブロック）。一般のニューラルネットワークで中間層を積むのと同じで、ブロック 1 の出力がブロック 2 の入力になる。同じ構造が別の重みで繰り返され、深いほど複雑な関係を扱える。このモデルは 2 層、GPT-2 small は 12 層。', chapter: 'mlp' },
  mlp: { term: 'MLP（多層パーセプトロン）', short: '行列を掛けて広げ、非線形関数を通し、行列を掛けて戻す 2 段の変換。注意機構が集めた情報を位置ごとに加工する。', chapter: 'mlp' },
  gelu: { term: 'GELU', short: 'MLP の途中で使う非線形関数。負の入力をほぼ 0 にし、正はそのまま通す。これがないと何層重ねても 1 つの行列と同じになる。', chapter: 'mlp' },
  logits: { term: 'ロジット', short: 'softmax にかける前の、語彙ごとのスコア。大きいほど「次に来やすい」。', chapter: 'output' },
  probability: { term: '確率分布', short: '語彙の各トークンが次に来る確率。合計 1。ここから 1 つ引くのが生成。', chapter: 'output' },
  temperature: { term: '温度', short: 'ロジットを割る数。小さいと 1 位に集中（決定的）、大きいと平らに（多様）。', chapter: 'output' },
  sampling: { term: 'サンプリング', short: '確率分布から 1 つのトークンを引くこと。常に最大を取る greedy、上位 k 個から引く top-k などがある。', chapter: 'output' },
  autoregressive: { term: '自己回帰', short: '出力したトークンを入力の末尾に足して、また次を予測する繰り返し。LLM の文章生成はすべてこれ。', chapter: 'output' },
  tying: { term: '重み共有（weight tying）', short: '出力層の行列に入力の埋め込み表をそのまま使うこと。パラメータを節約でき、GPT-2 でも採用。', chapter: 'output' },
  loss: { term: '損失（loss）', short: '予測の悪さを 1 つの数にしたもの。ここでは正解トークンに付けた確率の −log（交差エントロピー）。小さいほど良い。', chapter: 'train' },
  gradient: { term: '勾配', short: '各パラメータを少し増やしたとき損失がどれだけ変わるか。符号が更新の方向、大きさが強さを決める。', chapter: 'train' },
  backprop: { term: '逆伝播（backpropagation）', short: '損失の勾配を出力側から入力側へ、連鎖律で順に計算する手続き。', chapter: 'train' },
  adamw: { term: 'AdamW', short: '勾配の移動平均と二乗平均で歩幅を要素ごとに調整する更新法。weight decay で重みをわずかに 0 へ戻す。', chapter: 'train' },
  lr: { term: '学習率', short: '1 回の更新でパラメータをどれだけ動かすかの倍率。大きすぎると発散、小さすぎると遅い。', chapter: 'train' },
  batch: { term: 'バッチ', short: '1 回の更新に使う例のまとまり。ここでは 16 本の長さ 16 の列。', chapter: 'train' },
  step: { term: 'ステップ', short: 'forward → 損失 → 逆伝播 → 更新 の 1 回。同梱の学習済み重みは 3000 ステップ。', chapter: 'train' },
  context: { term: '文脈長（context length）', short: 'モデルが一度に見られるトークン数。このモデルは 16。超えた分は捨てる。', chapter: 'overview' },
  forward: { term: 'forward（順伝播）', short: '入力のトークン列から出力の確率分布まで、第 03〜06 章の計算を一通り流すこと。生成では 1 トークン出すたびに 1 回、学習では 1 ステップに 1 回行う。逆向きに勾配を流すのが backward（逆伝播）。', chapter: 'overview' },
  pca: { term: 'PCA（主成分分析）', short: '高次元のデータを、ばらつきが最も大きい方向に沿って 2 次元に射影する方法。見るためだけの道具でモデルには無関係。', chapter: 'embed' },
}

/** Inline term: dotted underline, hover for a definition, click to jump to the glossary at the page bottom. */
export function Term({ id, children }: { id: string; children?: ReactNode }) {
  const g = GLOSSARY[id]
  if (!g) return <>{children}</>
  return (
    <span
      className="term"
      title={g.short}
      role="link"
      tabIndex={0}
      onClick={() => {
        const el = document.getElementById(`gloss-${id}`)
        if (!el) return
        el.scrollIntoView({ block: 'center', behavior: 'smooth' })
        el.classList.add('flash')
        setTimeout(() => el.classList.remove('flash'), 1600)
      }}
    >
      {children ?? g.term}
    </span>
  )
}

export function GlossaryList({ ids, current }: { ids: string[]; current: string }) {
  const nav = useNav()
  return (
    <dl className="glossary">
      {ids.map((id) => {
        const g = GLOSSARY[id]
        if (!g) return null
        const link = g.chapter && g.chapter !== current
        return (
          <div key={id} id={`gloss-${id}`} className="gloss-item">
            <dt>{g.term}</dt>
            <dd>
              {g.short}
              {link && (
                <button className="gloss-link" onClick={() => nav.goFrom(g.chapter!, `gloss-${id}`)}>
                  → {chapterName(g.chapter!)} で詳しく
                </button>
              )}
            </dd>
          </div>
        )
      })}
    </dl>
  )
}

function chapterName(id: string): string {
  const names: Record<string, string> = { overview: '第 01 章', tokenize: '第 02 章', embed: '第 03 章', attention: '第 04 章', mlp: '第 05 章', output: '第 06 章', train: '第 07 章', params: '第 08 章' }
  return names[id] ?? id
}
