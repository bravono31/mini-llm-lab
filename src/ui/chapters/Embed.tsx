import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import { view } from '../../engine/params'
import { pca2d } from '../../engine/pca'
import { displayToken } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { fmtPct } from '../viz/colors'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import { Scatter2D } from '../viz/Scatter2D'
import { TokenChips } from '../viz/TokenChips'

export function Embed(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, tokenizer, ids, acts } = lab
  const { vocabSize: V, dModel: D, ctxLen } = params.config
  const T = ids.length
  const [tSel, setT] = useState<number | null>(null)
  const t = Math.min(tSel ?? T - 1, T - 1)
  const id = ids[t]
  const tokens = tokenizer.tokensOf(ids)
  const wte = view(params, 'wte')
  const wpe = view(params, 'wpe')
  const tokRow = wte.subarray(id * D, (id + 1) * D)
  const posRow = wpe.subarray(t * D, (t + 1) * D)
  const x0Row = acts.encoded.subarray(t * D, (t + 1) * D)
  const dimLabels = useMemo(() => Array.from({ length: D }, (_, i) => String(i)), [D])
  const vocabLabels = useMemo(() => tokenizer.vocab.map(displayToken), [tokenizer])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const pca = useMemo(() => pca2d(wte, V, D), [wte, V, D, lab.version])
  const idSet = useMemo(() => new Set(ids), [ids])

  const chips = <TokenChips tokens={tokens} ids={ids} positions active={t} onSelect={setT} />

  const steps: Step[] = [
    {
      id: 'table',
      title: '埋め込み表：ID ごとに 1 行',
      body: (
        <>
          <p>
            埋め込み表 W<sub>te</sub> は <strong>{V} 行 × {D} 列</strong>の数表です。行が語彙の各トークン、列が「意味の座標軸」に相当します。数値は学習で決まり、最初は乱数です。列に名前はなく、「列 3 が動物らしさ」のような意味は人が決めたものではありません。学習の結果として、たまたまそういう軸が生まれることがあります。
          </p>
          <p>入力文に含まれるトークンの行を強調しています。トークンをクリックすると選択が変わります。セルに触れると値が読めます。</p>
        </>
      ),
      formula: `W_te ∈ ℝ^{${V}×${D}}   （${V * D} 個のパラメータ）`,
    },
    {
      id: 'lookup',
      title: '行を取り出す = 埋め込み',
      body: (
        <>
          <p>
            「埋め込む」とは、ID {id} 番の<strong>行をそのまま取り出す</strong>ことです。掛け算はありません。one-hot ベクトル（{id} 番目だけ 1）と表の積と考えても同じです。
          </p>
          <p>こうしてトークン「{displayToken(tokens[t])}」は {D} 個の実数になりました。同じトークンは文中のどこにあっても同じ行を引きます。</p>
        </>
      ),
      formula: `e_t = W_te[ ids[t] ]  = W_te[${id}]`,
    },
    {
      id: 'pos',
      title: '位置埋め込み：何番目かを足す',
      body: (
        <>
          <p>
            注意機構は、そのままではトークンの<strong>順番を区別できません</strong>。そこで位置 t 専用のベクトル（W<sub>pe</sub> の t 行目）を用意します。文脈長 {ctxLen} なので {ctxLen} 行あります。
          </p>
          <p>この位置ベクトルも学習で決まります。位置 {t} の行を強調しています。</p>
        </>
      ),
      formula: `p_t = W_pe[t] = W_pe[${t}]`,
    },
    {
      id: 'sum',
      title: '足し合わせて入力ベクトルに',
      body: (
        <>
          <p>
            トークン埋め込みと位置埋め込みを<strong>要素ごとに足す</strong>だけで、Transformer への入力 x₀ が完成します。
          </p>
          <p>「何の語か」と「何番目か」が同じベクトルに混ざるのが不思議に見えますが、後段の層は学習によって両方を読み分けます。</p>
        </>
      ),
      formula: `x₀[t] = e_t + p_t`,
    },
    {
      id: 'x0',
      title: '入力行列 x₀',
      body: (
        <>
          <p>
            全トークン分を並べると <strong>{T} × {D}</strong> の行列になります。これが「残差ストリーム」の初期状態で、以降の層はこの行列を少しずつ書き換えていきます。
          </p>
          <p>行ごとに見ると、同じトークンでも位置が違えば少し違う値になっているはずです。</p>
        </>
      ),
      formula: `x₀ ∈ ℝ^{${T}×${D}}`,
    },
    {
      id: 'pca',
      title: '埋め込みの地図',
      body: (
        <>
          <p>
            {D} 次元は目で見えないので、主成分分析（PCA）で 2 次元に射影しました。分散の {fmtPct(pca.explained[0] + pca.explained[1], 0)} がこの平面に載っています。
          </p>
          <p>
            学習済みモデルでは、文中で同じような使われ方をするトークン（助詞どうし、動詞どうし）が近くに集まります。
            {lab.session.origin === 'random' && <strong>いまはランダム初期化なので、まだ構造はありません。</strong>}
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="03"
      title="埋め込み"
      lede="整数の ID を、意味と位置を持つベクトルに置き換える。ここで初めて「数値」が生まれます。"
      purpose={
        <>
          ID はただの番号で、「ねこ」と「いぬ」が似ているといった性質を持ちません。そこで各 ID を {D} 個の数の組（<Term id="vector">ベクトル</Term>）に置き換えます。この {D} 個の数は<Term id="parameter">学習で決まる</Term>ので、似た使われ方の語は似たベクトルになっていきます。さらに「何番目か」のベクトルを足して順番の情報も持たせます。{D} という<Term id="dimension">次元数</Term>は設計値で、画面に収まる大きさとして選びました（GPT-2 small は 768）。
        </>
      }
      io={`ID の列（${T} 個）→ ${T} × ${D} の行列 x₀`}
      terms={['embedding', 'vector', 'dimension', 'positional', 'parameter', 'residual', 'pca']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'table')
          return (
            <div className="col">
              {chips}
              <Heatmap values={wte} rows={V} cols={D} cell={12} rowLabels={vocabLabels} colLabels={dimLabels} highlightRows={[...idSet]} legend title={`W_te（${V} × ${D}）`} onRowClick={(r) => {
                const k = ids.indexOf(r)
                if (k >= 0) setT(k)
              }} rowName="token" colName="dim" tag="param" rowAxis="語彙のトークン（1 行 = 1 トークン）" colAxis={`意味の座標軸（${D} 本、名前はない）`} />
            </div>
          )
        if (step.id === 'lookup')
          return (
            <div className="col">
              {chips}
              <div className="card">
                <div className="card-title">
                  W_te[{id}]（トークン「{displayToken(tokens[t])}」の行）
                </div>
                <VectorStrip values={tokRow} colLabels={dimLabels} cell={34} legend tag="param" />
              </div>
            </div>
          )
        if (step.id === 'pos')
          return (
            <div className="col">
              {chips}
              <div className="row">
                <Heatmap values={wpe} rows={ctxLen} cols={D} cell={18} rowLabels={Array.from({ length: ctxLen }, (_, i) => `t=${i}`)} colLabels={dimLabels} highlightRows={[t]} legend title={`W_pe（${ctxLen} × ${D}）`} rowName="pos" colName="dim" tag="param" rowAxis="位置 t（何番目か）" colAxis="次元" />
                <div className="card grow">
                  <div className="card-title">W_pe[{t}]</div>
                  <VectorStrip values={posRow} colLabels={dimLabels} cell={34} />
                </div>
              </div>
            </div>
          )
        if (step.id === 'sum')
          return (
            <div className="col">
              {chips}
              <div className="card">
                <div className="card-title">e_t = W_te[{id}]</div>
                <VectorStrip values={tokRow} cell={34} maxAbs={1} />
                <div className="muted" style={{ margin: '6px 0 6px 4px' }}>
                  ＋ p_t = W_pe[{t}]
                </div>
                <VectorStrip values={posRow} cell={34} maxAbs={1} />
                <div className="muted" style={{ margin: '6px 0 6px 4px' }}>
                  ＝ x₀[{t}]
                </div>
                <VectorStrip values={x0Row} cell={34} maxAbs={1} colLabels={dimLabels} />
              </div>
            </div>
          )
        if (step.id === 'x0')
          return (
            <div className="col">
              {chips}
              <Heatmap values={acts.encoded} rows={T} cols={D} cell={22} rowLabels={tokens.map((tk, i) => `${i} ${displayToken(tk)}`)} rowLabelWidth={76} colLabels={dimLabels} highlightRows={[t]} showValues legend title={`x₀（${T} × ${D}）`} rowName="t" colName="dim" onRowClick={setT} tag="computed" rowAxis="入力文の位置 t（1 行 = 1 トークン）" colAxis="次元" />
            </div>
          )
        return (
          <div className="col">
            {chips}
            <Scatter2D points={pca.points} labels={vocabLabels} highlight={idSet} title={`W_te の PCA 射影（第 1 成分 ${fmtPct(pca.explained[0], 0)} ・ 第 2 成分 ${fmtPct(pca.explained[1], 0)}）`} />
          </div>
        )
      }}
    </ChapterLayout>
  )
}
