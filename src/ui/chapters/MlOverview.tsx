import { Term } from '../../content/glossary'
import { segment } from '../../engine/classifier'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { Heatmap, VectorStrip } from '../viz/Heatmap'
import type { ChapterProps } from './index'
import { ClassBadge, InputVerdict, ProbBars, WordChips } from './MlParts'

export function MlOverview({ onNavigate }: ChapterProps) {
  const ml = useMl()
  const { ds } = ml
  const F = ds.vocab.length
  const C = ds.classes.length
  const used = ds.vocab.flatMap((_, f) => (ml.x[f] ? [f] : []))
  const prefix = segment(ds.texts[6], ds.data)
    .slice(0, 2)
    .map((w) => w.text)
    .join(ds.data.lang === 'en' ? ' ' : '')
  const examples = ds.classes.map((_, c) => ds.texts.filter((_, i) => ds.y[i] === c).slice(0, 3))

  const flow = [
    {
      id: 'ml-data',
      name: 'データとラベル',
      why: `人が文を読んで「何の話か」の正解（ラベル）を付ける。${ds.texts.length} 文のうち ${ds.train.length} 文で学習し、残り ${ds.test.length} 文は最後の試験用に取っておく。`,
      shape: `${ds.texts.length} 文 × ラベル 1 つ`,
      viz: (
        <div className="muted small">
          {ds.texts.slice(0, 3).map((t, i) => (
            <div key={t}>
              <span className="mono">{t}</span> → <ClassBadge c={ds.y[i]} />
            </div>
          ))}
        </div>
      ),
    },
    {
      id: 'ml-features',
      name: '特徴量',
      why: `文のままでは計算できないので、人が決めた辞書の ${F} 語それぞれが何回出たかを数えて、${F} 個の数の並び（ベクトル）にする。`,
      shape: `文字列 → ${F} 個の整数（ほとんど 0）`,
      viz: (
        <div className="col" style={{ gap: 8 }}>
          <WordChips words={ml.words} />
          {used.length > 0 && <VectorStrip values={used.map((f) => ml.x[f])} colLabels={used.map((f) => ds.vocab[f])} mode="sequential" max={2} digits={0} cell={34} />}
        </div>
      ),
    },
    {
      id: 'ml-model',
      name: 'モデル',
      why: `語ごと・クラスごとの「点数」（重み）の表を持ち、出た語の点数を足し合わせてクラスごとの合計点にし、確率に直す。`,
      shape: `${F} 個の数 → ${C} 個の確率`,
      viz: <ProbBars probs={ml.probs} width={320} />,
    },
    {
      id: 'ml-train',
      name: '学習',
      why: '予測とラベルのずれ（損失）を測り、ずれが減る方向に重みを少し動かす。これを数百回繰り返す。',
      shape: '損失 → 勾配 → 更新された重み',
      viz: <div className="muted small">この編の分類器は最初から {ml.steps} ステップ学習済みです。ML-5 で 0 から学習し直せます。</div>,
    },
    {
      id: 'ml-eval',
      name: '評価',
      why: '学習に使わなかった文で正解率を測る。見たことのない文に通用するかどうかが、本当の実力。',
      shape: 'テスト用の文 → 正解率',
      viz: <div className="muted small">学習用の文だけで測ると、丸暗記でも満点が取れてしまいます。</div>,
    },
  ]

  const steps: Step[] = [
    {
      id: 'what',
      title: '機械学習とは：ルールを書く代わりに、例から学ぶ',
      body: (
        <>
          <p>
            「この文は何の話か」を判定するプログラムを作るとします。昔ながらの方法は、人がルールを書くことです。「ねこ・いぬ・とり… が入っていたら『いきもの』」。でも語が増えるたび、例外が見つかるたびに、人がルールを書き足し続けることになります。
          </p>
          <p>
            <Term id="ml">機械学習</Term>は逆の発想です。人は<strong>正解付きの例</strong>をたくさん用意するだけ。どの語がどのくらい「いきもの」らしいかという<strong>規則（重み）は、例からコンピュータが自動で決めます</strong>。
          </p>
          <p>この編では、LLM 編と同じひらがなの短文を使って、文を 3 つの話題に分ける分類器を 0 から作ります。上の入力文を書き換えると、この編のすべての章で分類し直します。</p>
        </>
      ),
    },
    {
      id: 'flow',
      title: '全体の流れ：データから予測まで',
      body: (
        <>
          <p>機械学習は、だいたいどれも左の 5 段でできています。各段をクリックすると、その章へ移動します。</p>
          <p>
            LLM 編と見比べると、<strong>「特徴量」を人が設計する</strong>ところと、<strong>正解（ラベル）を人が付ける</strong>ところが大きな違いです。LLM は特徴を自分で学び（<Term id="embedding">埋め込み</Term>）、正解は文章そのものから自動で作ります（次の語が正解）。違いは ML-7 でまとめます。
          </p>
        </>
      ),
    },
    {
      id: 'task',
      title: '同じ文、違う問題',
      body: (
        <>
          <p>LLM 編のミニモデルは、この文集で「次に来る語」を当てるよう学習しました。この編では同じ文集で「この文は何の話か」を当てます。</p>
          <p>
            クラス（答えの候補）は <ClassBadge c={0} /> <ClassBadge c={1} /> <ClassBadge c={2} /> の 3 つです。出力は<strong>この 3 つのどれか</strong>だけで、LLM のように文章を生み出すことはできません。決められた選択肢から選ぶ問題を<Term id="supervised">分類</Term>と呼びます。
          </p>
        </>
      ),
    },
    {
      id: 'kinds',
      title: '機械学習の種類と、LLM の位置',
      body: (
        <>
          <p>機械学習は「何を手がかりに学ぶか」で大きく分かれます。この編の分類器は、人が付けた正解から学ぶ<Term id="supervised">教師あり学習</Term>です。</p>
          <p>
            LLM の<Term id="pretrain">事前学習</Term>は<Term id="selfsupervised">自己教師あり学習</Term>と呼ばれます。正解（次の語）が文章の中にすでに入っているので、人がラベルを付けなくても、ネット上の膨大な文章をそのまま学習に使えます。LLM が「大規模」になれた一番の理由はここにあります。
          </p>
          <p>
            どちらも「例から重みを決める」点では同じ機械学習です。各分野の関係は
            <button className="gloss-link" onClick={() => onNavigate('map')}>
              → 全体像の地図
            </button>
            を参照してください。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="機械学習の概要"
      lede="ルールを人が書く代わりに、正解付きの例から規則を学ばせる。その最小の実例を、LLM 編と同じ文集で作ります。"
      purpose={<>LLM も機械学習の一種です。いちばん素朴な機械学習を先に見ておくと、LLM の何が新しく、何が同じなのかがはっきりします。</>}
      io={`文 → ${C} つの話題のどれか（いきもの / しぜん / ひと）`}
      terms={['ml', 'supervised', 'selfsupervised', 'label', 'feature', 'pretrain', 'embedding']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'what')
          return (
            <div className="col">
              <div className="row">
                <div className="card grow">
                  <div className="card-title">ルールを人が書く</div>
                  <div className="mono small" style={{ lineHeight: 1.9 }}>
                    もし「ねこ」「いぬ」…を含む → いきもの
                    <br />
                    もし「そら」「あめ」…を含む → しぜん
                    <br />
                    もし「わたし」「あなた」を含む → ひと
                    <br />
                    <span className="muted">「わたしはねこがすき」は？ → ルールを足す…</span>
                  </div>
                </div>
                <div className="card grow">
                  <div className="card-title">例から学ぶ（機械学習）</div>
                  <div className="small" style={{ lineHeight: 1.9 }}>
                    {ds.train.slice(0, 4).map((i) => (
                      <div key={i}>
                        <span className="mono">{ds.texts[i]}</span> → <ClassBadge c={ds.y[i]} />
                      </div>
                    ))}
                    <div className="muted">…など {ds.train.length} 文を見せると、規則（重み）が自動で決まる</div>
                  </div>
                </div>
              </div>
              <InputVerdict />
            </div>
          )
        if (step.id === 'flow')
          return (
            <div className="flow">
              {flow.map((f, i) => (
                <div key={f.name} className="flow-row clickable" onClick={() => onNavigate(f.id)} role="button">
                  <div className="flow-marker">
                    <div className="dot">{String(i + 1).padStart(2, '0')}</div>
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
        if (step.id === 'task')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">クラスごとの例</div>
                <table className="data">
                  <tbody>
                    {examples.map((ex, c) => (
                      <tr key={c}>
                        <td>
                          <ClassBadge c={c} />
                        </td>
                        <td className="mono">{ex.join('　')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="card">
                <div className="card-title">同じ文で、2 つのモデルが答えるもの</div>
                <table className="data">
                  <thead>
                    <tr>
                      <th></th>
                      <th>入力</th>
                      <th>出力</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>LLM 編</td>
                      <td className="mono">{prefix}</td>
                      <td>次の語の確率（語彙の数だけの選択肢）→ 文章が続く</td>
                    </tr>
                    <tr>
                      <td>この編</td>
                      <td className="mono">{ds.texts[6]}</td>
                      <td>
                        話題の確率（3 択）→ <ClassBadge c={ds.y[6]} />
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )
        return (
          <div className="col">
            <div className="card">
              <table className="data">
                <thead>
                  <tr>
                    <th>種類</th>
                    <th>正解はどこから</th>
                    <th>例</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>教師あり学習</td>
                    <td>人が付けたラベル</td>
                    <td>この編の話題分類、迷惑メール判定、画像の物体認識</td>
                  </tr>
                  <tr>
                    <td>自己教師あり学習</td>
                    <td>データ自身（文の続き、隠した部分）</td>
                    <td>LLM の事前学習（LLM-7）</td>
                  </tr>
                  <tr>
                    <td>教師なし学習</td>
                    <td>正解なし。データのまとまりを見つける</td>
                    <td>顧客のグループ分け、LLM-3 で使った PCA</td>
                  </tr>
                  <tr>
                    <td>強化学習</td>
                    <td>行動の結果の良し悪し（報酬）</td>
                    <td>ゲーム AI、対話モデルの事後学習（RLHF）</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <Heatmap
              values={Float64Array.from(ds.texts.slice(0, 8).flatMap((_, i) => ds.classes.map((_, c) => (ds.y[i] === c ? 1 : 0))))}
              rows={8}
              cols={C}
              mode="sequential"
              max={1}
              cell={64}
              cellH={20}
              rowLabels={ds.texts.slice(0, 8)}
              rowLabelWidth={160}
              colLabels={ds.classes.map((c) => ds.data.classNames[c])}
              title="教師あり学習の「正解」：人が付けたラベル（先頭 8 文）"
              tag="design"
            />
          </div>
        )
      }}
    </ChapterLayout>
  )
}
