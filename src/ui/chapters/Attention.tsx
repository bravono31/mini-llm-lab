import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import { view } from '../../engine/params'
import { displayToken } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { AttentionArcs } from '../viz/AttentionArcs'
import { fmt } from '../viz/colors'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import { MatrixMul } from '../viz/MatrixMul'
import { NetworkDiagram } from '../viz/NetworkDiagram'
import { TokenChips } from '../viz/TokenChips'
import { Arrow, Seg, StickyBar } from './controls'

export function Attention(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { params, tokenizer, ids, acts } = lab
  const { dModel: D, nHeads: H, nLayers: L } = params.config
  const hs = D / H
  const T = ids.length
  const [l, setL] = useState(0)
  const [h, setH] = useState(0)
  const [tSel, setT] = useState<number | null>(null)
  const [j, setJ] = useState(0)
  const [jo, setJo] = useState(0)
  const [jk, setJk] = useState<number | null>(null)
  const t = Math.min(tSel ?? T - 1, T - 1)
  const tokens = tokenizer.tokensOf(ids)
  const disp = tokens.map(displayToken)
  const a = acts.layers[l]
  const p = `layer${l}.`
  const dimLabels = useMemo(() => Array.from({ length: D }, (_, i) => String(i)), [D])
  const hsLabels = useMemo(() => Array.from({ length: hs }, (_, i) => String(i)), [hs])

  const xIn = a.xIn.subarray(t * D, (t + 1) * D)
  const ln1 = a.ln1.subarray(t * D, (t + 1) * D)
  const qkvRow = a.qkv.subarray(t * 3 * D, (t + 1) * 3 * D)
  const headRows = (offset: number) => {
    const out = new Float32Array(T * hs)
    for (let i = 0; i < T; i++) for (let d = 0; d < hs; d++) out[i * hs + d] = a.qkv[i * 3 * D + offset + h * hs + d]
    return out
  }
  const Q = useMemo(() => headRows(0), [a, h, T, hs, D]) // eslint-disable-line react-hooks/exhaustive-deps
  const K = useMemo(() => headRows(D), [a, h, T, hs, D]) // eslint-disable-line react-hooks/exhaustive-deps
  const Vh = useMemo(() => headRows(2 * D), [a, h, T, hs, D]) // eslint-disable-line react-hooks/exhaustive-deps
  const preatt = a.preatt.subarray(h * T * T, (h + 1) * T * T)
  const att = a.att.subarray(h * T * T, (h + 1) * T * T)
  const attRow = att.subarray(t * T, (t + 1) * T)
  const scoreRow = preatt.subarray(t * T, (t + 1) * T)
  const headOut = a.attnOut.subarray(t * D + h * hs, t * D + (h + 1) * hs)
  const attnOut = a.attnOut.subarray(t * D, (t + 1) * D)
  const attnProj = a.attnProj.subarray(t * D, (t + 1) * D)
  const x1 = a.x1.subarray(t * D, (t + 1) * D)
  const ln1g = view(params, p + 'ln1.g')
  const ln1b = view(params, p + 'ln1.b')
  const mean = a.ln1Mean[t]
  const rstd = a.ln1Rstd[t]

  let topKey = 0
  for (let i = 1; i <= t; i++) if (attRow[i] > attRow[topKey]) topKey = i
  let qk = 0
  for (let d = 0; d < hs; d++) qk += Q[t * hs + d] * K[topKey * hs + d]
  const kSel = Math.min(jk ?? topKey, T - 1)
  const qRow = Q.subarray(t * hs, (t + 1) * hs)
  const KT = useMemo(() => {
    const o = new Float32Array(hs * T)
    for (let i = 0; i < T; i++) for (let d = 0; d < hs; d++) o[d * T + i] = K[i * hs + d]
    return o
  }, [K, T, hs])
  const dots = useMemo(() => {
    const o = new Float32Array(T)
    for (let i = 0; i < T; i++) {
      let s = 0
      for (let d = 0; d < hs; d++) s += Q[t * hs + d] * K[i * hs + d]
      o[i] = s
    }
    return o
  }, [Q, K, t, T, hs])

  const colGroups = [
    { label: 'Q', from: 0, to: D, color: 'var(--q)' },
    { label: 'K', from: D, to: 2 * D, color: 'var(--k)' },
    { label: 'V', from: 2 * D, to: 3 * D, color: 'var(--v)' },
  ]

  const controls = (
    <StickyBar>
      <div className="row" style={{ alignItems: 'center', gap: 18 }}>
        <Seg label="層" value={l} options={Array.from({ length: L }, (_, i) => ({ value: i, label: `${i + 1}` }))} onChange={setL} />
        <Seg label="ヘッド" value={h} options={Array.from({ length: H }, (_, i) => ({ value: i, label: `${i + 1}` }))} onChange={setH} />
        <span className="field">注目トークン（クエリ）</span>
        <TokenChips tokens={tokens} active={t} onSelect={setT} positions />
      </div>
    </StickyBar>
  )
  const controlsHelp = (
    <p className="controls-help">
      <strong>層</strong>：注意機構 + MLP のひとまとまり（ブロック）を、ニューラルネットワークの中間層のように {L} 段積んでいます（最後のステップに図があります）。層 1 の出力が層 2 の入力です。どの段を見るかの切替。
      <strong>ヘッド</strong>：注意機構を {H} 個の小さな注意機構に分けたもの。1 つの注意は位置ごとに 1 通りの「誰を見るか」しか決められないので、複数に分けて別々の関係を同時に見ます。どの 1 つを表示するかの切替（softmax のステップに {H} つの比較図があります）。
      <strong>注目トークン</strong>：どの位置を「見る側（クエリ）」として表示するか。
    </p>
  )

  const steps: Step[] = [
    {
      id: 'ln1',
      title: 'LayerNorm：まず整える',
      body: (
        <>
          <p>
            各ブロックの入口で、トークンごとのベクトルを<strong>平均 0・分散 1 に正規化</strong>し、学習可能な γ（拡大）と β（平行移動）を掛け足します。値のスケールが揃うので、後の計算が安定します。
          </p>
          <p>
            位置 {t} のベクトルの平均は {fmt(mean, 3)}、標準偏差の逆数は {fmt(rstd, 3)} でした。
          </p>
        </>
      ),
      formula: `LN(x) = (x − μ) / √(σ² + ε) · γ + β`,
    },
    {
      id: 'qkv',
      title: 'Q・K・V を作る',
      body: (
        <>
          <p>
            正規化した {D} 次元のベクトルに行列 W<sub>qkv</sub> を掛けて、<strong>3 本のベクトル</strong> <Term id="qkv">Q・K・V</Term>（各 {D} 次元）を作ります。3 本を別々の行列で作っても同じですが、1 つの {D} × {3 * D} 行列にまとめて一度に計算するのが慣例です。だから出力は {3 * D} 列で、前から {D} 列ずつが <strong style={{ color: 'var(--q)' }}>Q</strong>・<strong style={{ color: 'var(--k)' }}>K</strong>・<strong style={{ color: 'var(--v)' }}>V</strong> です。
          </p>
          <p>
            <strong>行列の読み方</strong>：行 i は「入力ベクトルの i 番目の数にいくら掛けるか」、列 j は「出力の j 番目の数を作る配合表」です。つまり 1 列 = 1 つの出力を作るレシピで、{3 * D} 個のレシピが並んでいます。セルの色は重みの値（藍が負、朱が正）で、上の帯がその列の役割（Q / K / V）です。
          </p>
          <p>
            Q は「何を探しているか」、K は「自分は何か」、V は「相手に渡す中身」。各 {D} 次元はさらに {H} つの<Term id="head">ヘッド</Term>に {hs} 次元ずつ分けます。下の 3 枚の図はヘッド {h + 1} の分だけを取り出したもので、行 = 位置（トークン）、列 = その {hs} 次元です。
          </p>
          <p>
            <strong>なぜヘッドに分けるのか</strong>：1 つの注意は、位置ごとに 1 通りの「誰をどれだけ見るか」しか決められません。しかし「直前の語」「文の主語」「同じ語の前回の出現」のように、同時に見たい関係は複数あります。そこで {D} 次元を {H} 組に分け、組ごとに独立した Q・K・V で別々の関係を測ります。計算量は分けても増えず、表現できる関係の種類が増えるのが利点です。
          </p>
          <p>
            <strong>式の b はバイアス</strong>（偏り）です。直線 y = ax + b の b と同じで、入力がすべて 0 でも出力を b だけずらせる定数。掛け算 x·W だけでは出力の基準点を動かせないので、出力 {3 * D} 個それぞれに 1 個ずつ持たせ、W と一緒に学習で決めます。
          </p>
        </>
      ),
      formula: `[q | k | v] = LN(x) · W_qkv + b_qkv\nW_qkv ∈ ℝ^{${D}×${3 * D}}   （${D} 次元 → Q,K,V の ${D} 次元 × 3）\nhead h は各ベクトルの [${hs}h, ${hs}h+${hs}) 次元`,
    },
    {
      id: 'scores',
      title: 'スコア = q · k / √d',
      body: (
        <>
          <p>
            クエリ位置 t の q と、すべての位置 t′ の k の<strong>内積</strong>を取ります。似た方向を向いていれば大きな値。√{hs} で割るのは、次元数が増えても値が暴れないためです。
          </p>
          <p>
            位置 {t}（{disp[t]}）といちばん強く結びついたのは位置 {topKey}（{disp[topKey]}）で、q·k = {fmt(qk, 3)}、÷√{hs} で {fmt(scoreRow[topKey], 3)} です。
          </p>
        </>
      ),
      formula: `score[t, t′] = (q_t · k_t′) / √${hs}`,
    },
    {
      id: 'mask',
      title: '因果マスク：未来は見ない',
      body: (
        <>
          <p>
            言語モデルは「次」を当てるのが仕事なので、位置 t は<strong>自分より後ろの位置を参照できません</strong>。右上三角のスコアを −∞ に置き換え（実装では無視し）、softmax で重み 0 にします。
          </p>
          <p>
            このおかげで、1 回の <Term id="forward">forward</Term>（順伝播：入力のトークン列から出力の確率分布まで計算を一通り流すこと）で、全位置の「次トークン予測」を同時に計算でき、学習でも全位置の誤差を一度に使えます。
          </p>
        </>
      ),
      formula: `score[t, t′] = −∞  (t′ > t)`,
    },
    {
      id: 'softmax',
      title: 'softmax で注意重みに',
      body: (
        <>
          <p>
            スコア行列の<strong>各行</strong>（行 t = 見る側の位置 t が、各位置 t′ に付けた点数の並び）を <Term id="softmax">softmax</Term> にかけると、行ごとに<strong>合計 1 の重み</strong>になります。行 t の重みは「位置 t が各位置 t′ をどれだけ見るか」の配分です。これが「注意」です。弧の太さが重みの大きさ。位置 {t} は {disp[topKey]} に {fmt(attRow[topKey], 2)} の重みを置いています。
          </p>
          <p>
            下の「ヘッドの比較」を見ると、同じ層でもヘッドごとに見ている相手が違います。これがヘッドを分ける理由で、1 つの注意では表せない複数の関係を同時に扱えます。
          </p>
        </>
      ),
      formula: `att[t, t′] = exp(score[t,t′]) / Σ_u exp(score[t,u])`,
    },
    {
      id: 'weighted',
      title: 'Value の重み付き和',
      body: (
        <>
          <p>
            注意重みで各位置の v を<strong>混ぜ合わせ</strong>ます。重みが大きい位置の情報がたくさん取り込まれる。これがヘッド {h + 1} の出力（{hs} 次元）です。
          </p>
          <p>「関係のある語の情報を、自分のベクトルに集める」操作を、全位置・全ヘッドについて同時に行っています。</p>
        </>
      ),
      formula: `head_h[t] = Σ_t′ att[t, t′] · v_t′`,
    },
    {
      id: 'concat',
      title: 'ヘッドを結合し、W_o で混ぜる（出力射影）',
      body: (
        <>
          <p>
            {H} ヘッドの出力（{hs} 次元 × {H}）を横に<strong>並べて {D} 次元</strong>に戻します。並べただけでは「前半 {hs} 個はヘッド 1 の結果、後半はヘッド 2 の結果」と分かれたままなので、行列 W<sub>o</sub>（{D} × {D}）を掛けて混ぜ合わせ、1 本の {D} 次元ベクトルにします。
          </p>
          <p>
            この「行列を掛けて別の座標に写す」操作を<Term id="projection">射影</Term>と呼び、W<sub>o</sub> は注意機構の出力側にあるので「出力射影」と呼ばれます。Q・K・V を作った W<sub>qkv</sub> も射影（入力側）で、やっていることは同じ行列の掛け算です。
          </p>
        </>
      ),
      formula: `attn[t] = concat(head_1, …, head_${H}) · W_o + b_o`,
    },
    {
      id: 'residual',
      title: '残差接続：元に足し戻す',
      body: (
        <>
          <p>
            注意機構の出力は、入力を置き換えるのではなく<strong>入力に足されます</strong>。だから各層は「差分」だけを学べばよく、深いネットワークでも情報が失われません。
          </p>
          <p>この足し算の結果が、次の MLP への入力になります。</p>
        </>
      ),
      formula: `x₁[t] = x[t] + attn[t]`,
    },
  ]

  const scoresHeat = (masked: boolean) => (
    <Heatmap values={preatt} rows={T} cols={T} cell={26} rowLabels={disp} colLabels={disp} rowLabelWidth={64} highlightRows={[t]} masked={masked ? (r, c) => c > r : undefined} showValues legend title={`スコア（層 ${l + 1} ヘッド ${h + 1}）`} rowName="t" colName="t′" onRowClick={setT} tag="computed" rowAxis="見る側の位置 t（クエリ）" colAxis="見られる側の位置 t′（キー）" />
  )

  return (
    <ChapterLayout
      num="04"
      title="注意機構"
      lede="各トークンが文中の他のトークンを見渡し、関係の深い相手から情報を集める。Transformer の核心です。"
      purpose={
        <>
          埋め込みを終えた時点で、各位置のベクトルは「自分のトークンが何か」しか知りません。「ねこ が すき」の「すき」は、誰が好きなのかを知る必要があります。<Term id="attention">注意機構</Term>は、各位置が他の位置との関係の強さを計算し、強い相手の情報を自分のベクトルに足し込みます。これが「文脈を読む」操作です。関係の強さを測るために <Term id="qkv">Q・K・V</Term> という 3 本のベクトルを作り、<Term id="softmax">softmax</Term> で重みにします。
        </>
      }
      io={`${T} × ${D} の行列 → ${T} × ${D} の行列（各行に他の行の情報が混ざる）`}
      terms={['attention', 'qkv', 'head', 'layer', 'forward', 'projection', 'layernorm', 'softmax', 'mask', 'residual', 'dimension']}
      steps={steps}
    >
      {(step) => (
        <div className="col">
          {controls}
          {controlsHelp}
          {step.id === 'ln1' && (
            <div className="card">
              <div className="card-title">位置 {t}「{disp[t]}」</div>
              <div className="viz-title">x[{t}]（ブロックへの入力）</div>
              <VectorStrip values={xIn} cell={34} colLabels={dimLabels} tag="computed" />
              <Arrow>↓ 平均 μ = {fmt(mean, 3)}、1/σ = {fmt(rstd, 3)} で正規化し、γ を掛けて β を足す</Arrow>
              <div className="row">
                <div>
                  <VectorStrip values={ln1g} cell={34} maxAbs={2} title="γ" tag="param" />
                </div>
                <div>
                  <VectorStrip values={ln1b} cell={34} maxAbs={2} title="β" tag="param" />
                </div>
              </div>
              <Arrow>＝ LN(x[{t}])</Arrow>
              <VectorStrip values={ln1} cell={34} colLabels={dimLabels} />
            </div>
          )}
          {step.id === 'qkv' && (
            <>
              <div className="card">
                <MatrixMul x={ln1} W={view(params, p + 'attn.qkv.w')} C={D} OC={3 * D} bias={view(params, p + 'attn.qkv.b')} out={qkvRow} j={j} onSelect={setJ} xLabel={`LN(x[${t}])　位置 ${t} の正規化済みベクトル`} wLabel={`W_qkv（${D} 行 × ${3 * D} 列）`} outLabel="qkv" colGroups={colGroups} rowAxis={`入力の次元 i（LN(x) の ${D} 個の数のどれに掛けるか）`} colAxis={`出力の次元 j（0–${D - 1} が Q、${D}–${2 * D - 1} が K、${2 * D}–${3 * D - 1} が V）`} />
              </div>
              <div className="row">
                {[
                  ['q', Q, 'var(--q)'],
                  ['k', K, 'var(--k)'],
                  ['v', Vh, 'var(--v)'],
                ].map(([name, data, color]) => (
                  <div key={name as string}>
                    <div className="viz-title" style={{ color: color as string }}>
                      {name as string}（ヘッド {h + 1}、全位置 × {hs}）
                    </div>
                    <Heatmap values={data as Float32Array} rows={T} cols={hs} cell={18} rowLabels={disp} colLabels={hsLabels} highlightRows={[t]} rowLabelWidth={60} rowName="t" colName="d" tag="computed" rowAxis="位置 t" colAxis={`ヘッド ${h + 1} の次元`} />
                  </div>
                ))}
              </div>
            </>
          )}
          {step.id === 'scores' && (
            <>
              <div className="card">
                <div className="card-title">
                  スコアの作り方：位置 {t}「{disp[t]}」の q と、各位置の k との内積（ヘッド {h + 1}）
                </div>
                <div className="row" style={{ gap: 28 }}>
                  <Heatmap values={Q} rows={T} cols={hs} cell={22} rowLabels={disp} colLabels={hsLabels} rowLabelWidth={60} highlightRows={[t]} showValues title="Q（全位置の q）" tag="computed" rowAxis="位置 t（見る側）" colAxis={`ヘッド ${h + 1} の次元 d`} />
                  <Heatmap values={K} rows={T} cols={hs} cell={22} rowLabels={disp} colLabels={hsLabels} rowLabelWidth={60} highlightRows={[kSel]} showValues title="K（全位置の k）" tag="computed" rowAxis="位置 t′（見られる側）" colAxis={`ヘッド ${h + 1} の次元 d`} onRowClick={setJk} />
                </div>
                <Arrow>
                  ↓ Q の行 {t} を取り出し、K を転置した行列（{hs} 行 × {T} 列。列 t′ が k<sub>t′</sub>）に掛けると、全位置との内積が一度に出ます
                </Arrow>
                <MatrixMul x={qRow} W={KT} C={hs} OC={T} out={dots} j={kSel} onSelect={setJk} xLabel={`q_${t}（Q の行 ${t}）`} wLabel={`Kᵀ（${hs} 行 × ${T} 列）`} outLabel="q·k" cell={26} rowAxis={`ヘッド ${h + 1} の次元 d`} colAxis="見られる側の位置 t′（列をクリックで切替）" wTag="computed" />
                <Arrow>↓ ÷ √{hs} = {fmt(Math.sqrt(hs), 3)}</Arrow>
                <VectorStrip values={scoreRow} cell={44} colLabels={disp} highlight={[kSel]} title={`score[${t}, ·]`} tag="computed" />
                <p className="muted small" style={{ marginTop: 8 }}>
                  これがスコア行列の行 {t} です。同じことを全ての行 t について行うと、下の {T} × {T} の表になります（行 = 見る側の q、列 = 見られる側の k）。
                </p>
              </div>
              {scoresHeat(false)}
            </>
          )}
          {step.id === 'mask' && (
            <>
              {scoresHeat(true)}
              <div className="card">
                <div className="card-title">1 回の forward で、全位置の「次トークン」の確率が同時に出る</div>
                <Heatmap values={acts.probs} rows={T} cols={params.config.vocabSize} cell={Math.max(4, Math.min(8, Math.floor(560 / params.config.vocabSize)))} cellH={18} gap={0} mode="sequential" max={1} rowLabels={disp.map((d, i) => `${i} ${d}`)} rowLabelWidth={70} tag="computed" rowAxis="位置 t（その位置までを見て「次」を予測）" colAxis="語彙の各トークン（濃いほど高確率）" legend />
                <p className="muted small" style={{ marginTop: 8 }}>
                  マスクのおかげで行 t は位置 t までしか見ていないので、{T} 行すべてが「そこまで読んだ時点の予測」として同時に使えます。第 06 章の最後のステップに、行ごとの予測の表があります。
                </p>
              </div>
            </>
          )}
          {step.id === 'softmax' && (
            <>
              <div className="card">
                <div className="card-title">「各行を softmax」とは：スコア行列の行 t（= 見る側の位置 {t}「{disp[t]}」）を取り出して、合計 1 にする</div>
                <div className="viz-title">スコア行列の行 {t}（列 = 見られる側 t′。灰色は未来でマスク済み）</div>
                <VectorStrip values={scoreRow} cell={44} colLabels={disp} masked={(_, c) => c > t} tag="computed" />
                <Arrow>↓ softmax：exp を取って合計で割る（マスクされた列は 0）</Arrow>
                <div className="viz-title">注意重みの行 {t}（合計 = {fmt(Array.from(attRow).reduce((s, v) => s + v, 0), 2)}）</div>
                <VectorStrip values={attRow} cell={44} colLabels={disp} mode="sequential" max={1} tag="computed" />
                <p className="muted small" style={{ marginTop: 8 }}>
                  下のヒートマップでは、この行が朱の枠で囲まれています。他の行も同じ操作を独立に受けます。
                </p>
              </div>
              <AttentionArcs tokens={tokens} weights={attRow} query={t} onSelect={setT} />
              <div className="card">
                <div className="card-title">ヘッドの比較（層 {l + 1}）：同じ入力でも「誰を見るか」が違う</div>
                <div className="row" style={{ gap: 28 }}>
                  {Array.from({ length: H }, (_, hh) => (
                    <div key={hh} style={{ opacity: hh === h ? 1 : 0.75, cursor: 'pointer' }} onClick={() => setH(hh)}>
                      <Heatmap values={a.att.subarray(hh * T * T, (hh + 1) * T * T)} rows={T} cols={T} cell={22} mode="sequential" max={1} rowLabels={disp} colLabels={disp} rowLabelWidth={60} highlightRows={[t]} showValues title={`ヘッド ${hh + 1}${hh === h ? '（表示中）' : ''}`} tag="computed" rowAxis="見る側 t" colAxis="見られる側 t′" />
                    </div>
                  ))}
                </div>
              </div>
              <Heatmap values={att} rows={T} cols={T} cell={26} mode="sequential" max={1} rowLabels={disp} colLabels={disp} rowLabelWidth={64} highlightRows={[t]} showValues legend title={`注意重み（層 ${l + 1} ヘッド ${h + 1}）　各行の合計 = 1`} rowName="t" colName="t′" onRowClick={setT} tag="computed" rowAxis="見る側の位置 t" colAxis="見られる側の位置 t′" />
            </>
          )}
          {step.id === 'weighted' && (
            <div className="card">
              <div className="viz-title">att[{t}, ·]（重み）</div>
              <VectorStrip values={attRow} cell={34} mode="sequential" max={1} colLabels={disp} />
              <Arrow>× 各位置の v（ヘッド {h + 1}）</Arrow>
              <Heatmap values={Vh} rows={T} cols={hs} cell={22} rowLabels={disp} colLabels={hsLabels} rowLabelWidth={60} highlightRows={[topKey]} showValues rowName="t′" colName="d" />
              <Arrow>＝ Σ を取って head_{h + 1}[{t}]</Arrow>
              <VectorStrip values={headOut} cell={34} colLabels={hsLabels} />
            </div>
          )}
          {step.id === 'concat' && (
            <div className="card">
              <div className="viz-title">concat(head_1 … head_{H})[{t}]　（{hs} 次元 × {H} ヘッド）</div>
              <VectorStrip values={attnOut} cell={30} colLabels={Array.from({ length: D }, (_, i) => `h${Math.floor(i / hs) + 1}·${i % hs}`)} />
              <Arrow>↓ W_o を掛ける</Arrow>
              <MatrixMul x={attnOut} W={view(params, p + 'attn.proj.w')} C={D} OC={D} bias={view(params, p + 'attn.proj.b')} out={attnProj} j={jo} onSelect={setJo} xLabel="concat" wLabel={`W_o（${D} × ${D}）`} outLabel="attn" cell={16} />
            </div>
          )}
          {step.id === 'residual' && (
            <div className="card">
              <NetworkDiagram panel="stack" />
              <div className="viz-title" style={{ marginTop: 18 }}>x[{t}]（ブロック入力）</div>
              <VectorStrip values={xIn} cell={34} maxAbs={Math.max(...x1.map(Math.abs))} />
              <Arrow>＋ attn[{t}]（注意機構の出力）</Arrow>
              <VectorStrip values={attnProj} cell={34} maxAbs={Math.max(...x1.map(Math.abs))} />
              <Arrow>＝ x₁[{t}]</Arrow>
              <VectorStrip values={x1} cell={34} maxAbs={Math.max(...x1.map(Math.abs))} colLabels={dimLabels} />
            </div>
          )}
        </div>
      )}
    </ChapterLayout>
  )
}
