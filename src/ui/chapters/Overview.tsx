import { useMemo } from 'react'
import { Term } from '../../content/glossary'
import { totalSize } from '../../engine/params'
import { displayToken } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmtPct } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import { NetworkDiagram } from '../viz/NetworkDiagram'
import { Tag } from '../viz/Tag'
import { TokenChips } from '../viz/TokenChips'
import type { ChapterProps } from './index'

export function Overview({ onNavigate }: ChapterProps) {
  const lab = useLab()
  const cfg = lab.params.config
  const T = lab.ids.length
  const total = totalSize(lab.params.specs)
  const tokens = lab.tokenizer.tokensOf(lab.ids)
  const disp = tokens.map(displayToken)
  const V = cfg.vocabSize

  const breakdown = useMemo(() => {
    const g = { 埋め込み: 0, 注意機構: 0, MLP: 0, LayerNorm: 0 }
    for (const s of lab.params.specs) {
      if (s.group === 'embedding') g['埋め込み'] += s.size
      else if (s.name.includes('attn')) g['注意機構'] += s.size
      else if (s.name.includes('mlp')) g['MLP'] += s.size
      else g['LayerNorm'] += s.size
    }
    return Object.entries(g).map(([label, value]) => ({ label, value, sub: fmtPct(value / total, 0) }))
  }, [lab.params.specs, total])

  const top3 = useMemo(() => {
    const probs = lab.acts.probs.subarray((T - 1) * V, T * V)
    return [...probs.keys()]
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, 3)
      .map((i) => ({ label: displayToken(lab.tokenizer.vocab[i]), value: probs[i] }))
  }, [lab.acts, lab.tokenizer, V, T])

  const att0 = lab.acts.layers[0].att.subarray(0, T * T)
  const lastX = lab.acts.layers[cfg.nLayers - 1].x2

  const flow = [
    {
      id: null,
      name: '入力文',
      why: '人が書いた文字列。コンピュータはこのままでは計算できない。',
      shape: `"${lab.input}"`,
      viz: <div className="mono" style={{ fontSize: 18 }}>{lab.input || lab.corpus.defaultInput}</div>,
    },
    {
      id: 'tokenize',
      name: 'トークン化',
      why: '文字列を語彙表の番号（整数）の列にする。数にして初めて計算の対象になる。',
      shape: `文字列 → ${T} 個の整数`,
      viz: <TokenChips tokens={tokens} ids={lab.ids} />,
    },
    {
      id: 'embed',
      name: '埋め込み',
      why: `番号には意味がない。学習で決まった ${cfg.dModel} 個の数（ベクトル）に置き換え、「何番目か」も足して、意味と位置を持たせる。`,
      shape: `${T} 個の整数 → ${T} × ${cfg.dModel} の行列`,
      viz: <Heatmap values={lab.acts.encoded} rows={T} cols={cfg.dModel} cell={11} rowLabels={disp} rowLabelWidth={60} tag="computed" />,
    },
    {
      id: 'attention',
      name: `注意機構 × ${cfg.nLayers}`,
      why: '各トークンが文中の他のトークンを見渡し、関係の深い相手の情報を自分のベクトルに取り込む。ここで「文脈」が生まれる。',
      shape: `${T} × ${cfg.dModel} → ${T} × ${cfg.dModel}（各行に文脈が混ざる）`,
      viz: <Heatmap values={att0} rows={T} cols={T} cell={16} mode="sequential" max={1} rowLabels={disp} colLabels={disp} rowLabelWidth={60} tag="computed" title="誰が誰を見ているか（層 1 ヘッド 1 の注意重み）" />,
    },
    {
      id: 'mlp',
      name: `MLP × ${cfg.nLayers}`,
      why: '集めた情報を位置ごとに変換して書き足す。「猫は魚を食べる」のような知識はここに蓄えられると考えられている。',
      shape: `${T} × ${cfg.dModel} → ${T} × ${cfg.dModel}`,
      viz: <Heatmap values={lastX} rows={T} cols={cfg.dModel} cell={11} rowLabels={disp} rowLabelWidth={60} tag="computed" title="最終ブロックを出た行列" />,
    },
    {
      id: 'output',
      name: '出力 → 確率',
      why: `最後の行のベクトルを語彙 ${V} 個それぞれの「次に来やすさ」に変換し、確率にする。そこから 1 つ引いて文末に足す。`,
      shape: `${cfg.dModel} 個の数 → ${V} 個の確率 → 1 トークン`,
      viz: <BarChart items={top3.map((t, i) => ({ ...t, highlight: i === 0 }))} max={1} format={(v) => fmtPct(v)} width={320} barHeight={14} />,
    },
    {
      id: 'train',
      name: '学習',
      why: '予測と正解のずれ（損失）を測り、ずれを減らす方向に全パラメータを少し動かす。数千回繰り返すと乱数だった重みが言語の規則を写す。',
      shape: '損失 → 勾配 → 更新された重み',
      viz: <div className="muted small">推論（上の 6 段階）を毎回やり直しながら、重みだけが少しずつ変わっていきます。</div>,
    },
  ]

  const numbers: [string, string | number, string][] = [
    ['埋め込みの次元 d_model', cfg.dModel, '設計値。画面に 1 行が収まる大きさとして 16 にした。GPT-2 small は 768、大型モデルは 1 万前後'],
    ['ヘッド数', cfg.nHeads, `設計値。${cfg.dModel} 次元を ${cfg.nHeads} 等分し、1 ヘッド ${cfg.dModel / cfg.nHeads} 次元。GPT-2 small は 12 ヘッド`],
    ['Q・K・V 射影の出力', 3 * cfg.dModel, `導出値。Q・K・V それぞれ ${cfg.dModel} 次元が必要なので 3 × ${cfg.dModel}。3 つを 1 つの行列で一度に計算している`],
    ['MLP の中間次元 d_ff', cfg.dFF, `慣例。GPT-2 以来「d_model の 4 倍」が標準で、${cfg.dModel} × 4 = ${cfg.dFF}`],
    ['層の数', cfg.nLayers, '設計値。仕組みが見える最小の深さ。GPT-2 small は 12 層'],
    ['語彙数', V, `トークナイザの目標値。コーパスの規模から ${V} に設定した。GPT-2 は 50,257`],
    ['文脈長', cfg.ctxLen, '設計値。注意の行列が読める大きさとして 16。GPT-2 は 1,024、最近のモデルは数十万'],
    ['パラメータ総数', total.toLocaleString(), '上の値から計算で決まる。第 08 章に内訳'],
  ]

  const steps: Step[] = [
    {
      id: 'flow',
      title: '全体の流れ：一文が次の一語になるまで',
      body: (
        <>
          <p>
            <Term id="llm" /> は「これまでの<Term id="token">トークン</Term>列から、次のトークンの<Term id="probability">確率分布</Term>を出す」関数です。左の図は、いま上に入っている文が各段階でどんな形のデータになるかを、実際の値で並べたものです。
          </p>
          <p>
            大事なのは<strong>形</strong>です。文字列 → 整数の列 → 行列（T × 16）→ …と形が変わり、行列になってからは <Term id="attention">注意機構</Term>と <Term id="mlp">MLP</Term> が同じ形のまま中身を書き換えていきます。
          </p>
          <p>各段の「なぜ」を読めば、次の章以降で見る計算が何のためのものか分かります。段をクリックすると章へ移動します。</p>
        </>
      ),
    },
    {
      id: 'network',
      title: 'ニューラルネットワークとしての構造',
      body: (
        <>
          <p>
            <Term id="transformer">Transformer</Term> も、ノード（数値）を線（重み）でつないだ<strong>ニューラルネットワーク</strong>です。左の図は <Term id="mlp">MLP</Term> をそのまま描いたもので、入力 {cfg.dModel} → 中間層 {cfg.dFF} → 出力 {cfg.dModel} の 3 層ネットワークそのものです。線 1 本が重み 1 個に対応します。
          </p>
          <p>
            右の図は全体です。注意機構と MLP を組にした「ブロック」を、一般のネットワークで中間層を積むのと同じように {cfg.nLayers} 段重ねています。各章の「層」の切替は、この段のどれを見るかの選択です。
          </p>
          <p>一般のネットワークと違うのは 2 点。ノード 1 つが数値ではなく {cfg.dModel} 次元のベクトル（行列の 1 行）であること、そして注意機構が行どうし（トークンどうし）をつなぐことです。</p>
        </>
      ),
    },
    {
      id: 'model',
      title: 'モデルカードと、数の決め方',
      body: (
        <>
          <p>
            これは <Term id="gpt">GPT</Term>-2 と同じ構造（<Term id="transformer">Transformer</Term>）を極小化したモデルで、<strong>{total.toLocaleString()} 個</strong>の<Term id="parameter">パラメータ</Term>を持ちます。
          </p>
          <p>
            「16 次元」「48 列」といった数は自然に決まるものではなく、<strong>設計者が決めた値</strong>と、そこから<strong>計算で導かれる値</strong>があります。左の表に、それぞれの数がどう決まったかを書きました。
          </p>
        </>
      ),
    },
    {
      id: 'values',
      title: '画面の数値は 3 種類',
      body: (
        <>
          <p>以降の章で出てくる数値には、出どころが 3 種類あります。各図のタイトル横のタグで区別しています。</p>
          <p>
            <Tag kind="param" /> は最初は乱数（平均 0・標準偏差 0.02 の正規分布）で、<Term id="step">学習ステップ</Term>ごとに変わる値。
            <Tag kind="computed" /> は入力文とパラメータから一意に計算される値で、入力を変えると変わります。
            <Tag kind="design" /> は人が決めた定数です。
          </p>
        </>
      ),
    },
    {
      id: 'howto',
      title: 'このアプリの使い方',
      body: (
        <>
          <p>
            <strong>入力文</strong>（上部）を書き換えると、すべての章の数値がその文で再計算されます。使える文字はコーパスに含まれるものだけです。
          </p>
          <p>各章は右のパネルの ▶ でステップを進めます。キーボードの ← → でも移動でき、スペースで自動再生します。</p>
          <p>本文中の点線付きの語は用語で、クリックするとページ下の用語集に飛びます。用語集から解説の章へ移動した場合、上部のバーで元の章に戻れます。</p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="01"
      title="概要"
      lede="入力文が次の一語に変わるまでの道のりを、実際の値で並べました。"
      steps={steps}
      terms={['llm', 'transformer', 'gpt', 'parameter', 'layer', 'token', 'vector', 'attention', 'mlp', 'residual', 'probability', 'forward', 'context', 'step']}
    >
      {(step) => {
        if (step.id === 'flow')
          return (
            <div className="flow">
              {flow.map((f, i) => (
                <div key={f.name} className={'flow-row' + (f.id ? ' clickable' : '')} onClick={() => f.id && onNavigate(f.id)} role={f.id ? 'button' : undefined}>
                  <div className="flow-marker">
                    <div className="dot">{i === 0 ? '·' : String(i + 1).padStart(2, '0')}</div>
                    {i < flow.length - 1 && <div className="line" />}
                  </div>
                  <div className="flow-text">
                    <div className="name">{f.name}</div>
                    <div className="why">{f.why}</div>
                    <div className="shape">{f.shape}</div>
                  </div>
                  <div className="flow-viz">{f.viz}</div>
                </div>
              ))}
            </div>
          )
        if (step.id === 'network') return <NetworkDiagram panel="both" />
        if (step.id === 'model')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">それぞれの数はどう決まったか</div>
                <table className="data">
                  <thead>
                    <tr>
                      <th>項目</th>
                      <th>値</th>
                      <th>決まり方</th>
                    </tr>
                  </thead>
                  <tbody>
                    {numbers.map(([k, v, how]) => (
                      <tr key={k}>
                        <td>{k}</td>
                        <td className="num">{v}</td>
                        <td className="muted">{how}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="card" style={{ maxWidth: 520 }}>
                <BarChart title="パラメータの内訳" items={breakdown} width={440} labelWidth={84} format={(v) => v.toLocaleString()} />
              </div>
            </div>
          )
        if (step.id === 'values')
          return (
            <div className="card" style={{ maxWidth: 640 }}>
              <table className="data">
                <tbody>
                  <tr>
                    <td>
                      <Tag kind="param" />
                    </td>
                    <td>埋め込み表、W_qkv、W_o、W₁、W₂、LayerNorm の γ・β。初期値は乱数で、第 07 章の学習で動く。全 {total.toLocaleString()} 個</td>
                  </tr>
                  <tr>
                    <td>
                      <Tag kind="computed" />
                    </td>
                    <td>x₀、Q・K・V、注意スコア、注意重み、MLP の中間値、ロジット、確率。入力文とパラメータが決まれば一意に決まる</td>
                  </tr>
                  <tr>
                    <td>
                      <Tag kind="design" />
                    </td>
                    <td>次元数 16、ヘッド数 2、層数 2、d_ff 64、文脈長 16、語彙数。設計時に人が選んだ定数</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )
        return (
          <div className="card" style={{ maxWidth: 560 }}>
            <table className="data">
              <tbody>
                <tr>
                  <td>入力文</td>
                  <td>上部の欄。語彙外の文字は朱で警告</td>
                </tr>
                <tr>
                  <td>ステップ</td>
                  <td>右パネルの ◀ ▶、または ← → キー</td>
                </tr>
                <tr>
                  <td>自動再生</td>
                  <td>「自動再生」ボタン、またはスペースキー</td>
                </tr>
                <tr>
                  <td>数値の確認</td>
                  <td>ヒートマップのセルにマウスを乗せると値が出ます</td>
                </tr>
                <tr>
                  <td>用語</td>
                  <td>点線付きの語をクリック → ページ下の用語集。そこから解説の章へ</td>
                </tr>
                <tr>
                  <td>コーパス</td>
                  <td>日本語 / English を切替（モデルも切り替わります）</td>
                </tr>
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
