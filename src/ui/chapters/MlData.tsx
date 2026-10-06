import { Term } from '../../content/glossary'
import { displayToken } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import { useMl } from '../state/MlProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import type { ChapterProps } from './index'
import { CLASS_COLORS, ClassBadge, useClassNames } from './MlParts'

export function MlData(_: ChapterProps) {
  const lab = useLab()
  const { ds } = useMl()
  const names = useClassNames()
  const testSet = new Set(ds.test)
  const count = (idx: number[]) => ds.classes.map((_, c) => idx.filter((i) => ds.y[i] === c).length)
  const trainCount = count(ds.train)
  const testCount = count(ds.test)
  const ja = ds.data.lang === 'ja'

  // how the LLM part turns one sentence into training examples by itself
  const sample = ds.texts[6]
  const toks = lab.tokenizer.tokensOf(lab.tokenizer.encode(sample))
  const pairs = toks.map((_, i) => [toks.slice(0, i + 1).join(''), toks[i + 1] === undefined ? '⟨eos⟩' : displayToken(toks[i + 1])])

  const ambiguous: [string, string, string][] = ja
    ? [
        ['わたしのねこはしろい', 'いきもの', '「わたし」が出てくるが、白いのはねこ。話の主役はねこなので いきもの にした'],
        ['わたしはねこがすき', 'ひと', '「ねこ」も出てくるが、好きなのは わたし。主役は わたし なので ひと にした'],
        ['ねこがすき', 'いきもの', '主語（だれが好きか）が書かれていない。ねこの話として いきもの にした'],
        ['ねこはつきをみる', 'いきもの', '「つき」は しぜん の語だが、見ているのはねこ'],
      ]
    : [
        ['my cat is white', 'animals', '「my」は人を指すが、白いのは猫。主役は猫なので animals にした'],
        ['i like cats', 'people', '「cats」も出てくるが、好きなのは i。主役は i なので people にした'],
        ['your dog is black', 'animals', '「your」は人を指すが、黒いのは犬'],
        ['i see the moon', 'people', '「moon」は nature の語だが、見ているのは i'],
      ]

  const steps: Step[] = [
    {
      id: 'table',
      title: 'データ：文と正解の組',
      body: (
        <>
          <p>
            <Term id="supervised">教師あり学習</Term>のデータは「入力」と「正解」の組の集まりです。ここでは入力が文、正解がその文の話題（<Term id="label">ラベル</Term>）です。左の表は {ds.texts.length} 文すべてと、そのラベルです。
          </p>
          <p>
            文は LLM 編のコーパスと同じものです。ラベルは<strong>このアプリの作者が 1 文ずつ読んで手で付けました</strong>。基準は「その文が何について述べているか」。
          </p>
          <p>クラスの件数に偏りがあることにも注意してください（{names.map((n, c) => `${n} ${trainCount[c] + testCount[c]}`).join('・')}）。多いクラスほどモデルは「迷ったらそれ」と答えやすくなります。</p>
        </>
      ),
    },
    {
      id: 'labels',
      title: 'ラベルは人の判断',
      body: (
        <>
          <p>ラベル付けは機械的な作業に見えて、実は判断の連続です。左の文はどれも、別の人なら別のラベルを付けてもおかしくありません。</p>
          <p>
            モデルは<strong>ラベルを付けた人の基準をそのまま真似る</strong>だけです。基準がぶれていればモデルもぶれ、偏っていればモデルも偏ります。モデルの性能の上限は、ラベルの質で決まると言ってもかまいません。
          </p>
          <p>実務では、何千・何万もの例にラベルを付ける作業が、機械学習でいちばん時間とお金のかかる部分になることがよくあります。</p>
        </>
      ),
    },
    {
      id: 'split',
      title: '学習用とテスト用に分ける',
      body: (
        <>
          <p>
            全部の文で学習してしまうと、モデルが「本当に規則を覚えたのか、答えを丸暗記しただけなのか」を確かめられません。そこで 4 文に 1 文（{ds.test.length} 文）を<Term id="testdata">テストデータ</Term>として取りのけ、学習には残りの {ds.train.length} 文だけを使います。
          </p>
          <p>試験問題を事前に見せないのと同じです。テストデータの成績は ML-6 で測ります。</p>
        </>
      ),
    },
    {
      id: 'vs-llm',
      title: 'LLM の学習にはラベルが要らない',
      body: (
        <>
          <p>
            LLM 編の学習（LLM-7）では、人はラベルを 1 つも付けていません。左の表のように、<strong>文の前半が入力、次の語が正解</strong>になるので、文章さえあれば正解付きの例がいくらでも自動で作れます。これを<Term id="selfsupervised">自己教師あり学習</Term>と呼びます。
          </p>
          <p>
            1 文から、この分類器は例を <strong>1 つ</strong>しか作れませんが、LLM は<strong>トークンの数だけ</strong>作れます。しかも人手が要らないので、インターネット規模の文章をそのまま使えます。LLM が大規模になれたのは、この「ラベル不要」の性質のおかげです。
          </p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      title="データとラベル"
      lede="機械学習の出発点は、正解付きの例です。正解は人が付けます。"
      purpose={<>モデルは例から学ぶことしかできません。どんな例を、どんな正解とともに見せるかで、モデルの能力と癖が決まります。</>}
      io={`${ds.texts.length} 文 → （文, ラベル）の組 ${ds.texts.length} 個 ＝ 学習用 ${ds.train.length} ＋ テスト用 ${ds.test.length}`}
      terms={['supervised', 'label', 'testdata', 'selfsupervised']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'table')
          return (
            <div className="card" style={{ maxWidth: 560 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>文</th>
                    <th>ラベル</th>
                  </tr>
                </thead>
                <tbody>
                  {ds.texts.map((t, i) => (
                    <tr key={t}>
                      <td className="num muted">{i + 1}</td>
                      <td className="mono">{t}</td>
                      <td>
                        <ClassBadge c={ds.y[i]} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        if (step.id === 'labels')
          return (
            <div className="card" style={{ maxWidth: 720 }}>
              <div className="card-title">迷う文と、付けたラベル</div>
              <table className="data">
                <thead>
                  <tr>
                    <th>文</th>
                    <th>ラベル</th>
                    <th>判断</th>
                  </tr>
                </thead>
                <tbody>
                  {ambiguous.map(([t, l, why]) => (
                    <tr key={t}>
                      <td className="mono">{t}</td>
                      <td>
                        <ClassBadge c={names.indexOf(l)} />
                      </td>
                      <td className="muted">{why}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        if (step.id === 'split')
          return (
            <div className="row">
              <div className="card">
                <BarChart title={`学習用 ${ds.train.length} 文の内訳`} items={names.map((label, c) => ({ label, value: trainCount[c], color: CLASS_COLORS[c] }))} min={0} width={320} format={(v) => String(v)} />
                <BarChart title={`テスト用 ${ds.test.length} 文の内訳`} items={names.map((label, c) => ({ label, value: testCount[c], color: CLASS_COLORS[c] }))} min={0} max={Math.max(...trainCount)} width={320} format={(v) => String(v)} />
              </div>
              <div className="card" style={{ maxWidth: 360 }}>
                <div className="card-title">テスト用の文（学習では見せない）</div>
                <div className="small" style={{ lineHeight: 1.9 }}>
                  {ds.texts.map((t, i) =>
                    testSet.has(i) ? (
                      <div key={t}>
                        <span className="mono">{t}</span> <ClassBadge c={ds.y[i]} />
                      </div>
                    ) : null,
                  )}
                </div>
              </div>
            </div>
          )
        return (
          <div className="row">
            <div className="card">
              <div className="card-title">この編：1 文 → 例 1 つ（ラベルは人が付ける）</div>
              <table className="data">
                <thead>
                  <tr>
                    <th>入力</th>
                    <th>正解</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="mono">{sample}</td>
                    <td>
                      <ClassBadge c={ds.y[6]} />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="card">
              <div className="card-title">LLM 編：1 文 → 例 {pairs.length} 個（正解は文の中にある）</div>
              <table className="data">
                <thead>
                  <tr>
                    <th>入力（ここまで）</th>
                    <th>正解（次のトークン）</th>
                  </tr>
                </thead>
                <tbody>
                  {pairs.map(([inp, next], i) => (
                    <tr key={i}>
                      <td className="mono">{inp}</td>
                      <td className="mono">{next}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
