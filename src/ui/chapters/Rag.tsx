import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import ragJa from '../../data/rag-ja.json'
import { RAG_SETS } from '../../data/rag-docs'
import { checkpointParams, compare, retrieve, type RagCheckpoints } from '../../engine/rag'
import { displayToken, EOS, EOS_ID, type Lang } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { BarChart } from '../viz/BarChart'
import { fmt } from '../viz/colors'
import { Heatmap } from '../viz/Heatmap'
import { TokenChips } from '../viz/TokenChips'
import type { ChapterProps } from './index'
import { Arrow, Seg, StickyBar } from './controls'
import { AnswerView } from './RagAnswer'

const RAG_MODELS: Partial<Record<Lang, RagCheckpoints>> = { ja: ragJa as RagCheckpoints }

export function Rag(_: ChapterProps) {
  const lab = useLab()
  const { tokenizer, lang } = lab
  const ctxLen = lab.params.config.ctxLen
  const set = RAG_SETS[lang]
  const [qIdx, setQ] = useState(0)
  const [k, setK] = useState(2)
  const checkpoints = RAG_MODELS[lang]?.checkpoints
  const [ckSel, setCk] = useState<number | null>(null)
  const ckIdx = checkpoints ? Math.min(ckSel ?? checkpoints.length - 1, checkpoints.length - 1) : 0
  const question = set.questions[qIdx]

  const docIds = useMemo(() => set.docs.map((d) => tokenizer.encode(d)), [set, tokenizer])
  const qIds = useMemo(() => tokenizer.encode(question.text), [tokenizer, question])
  const r = useMemo(() => retrieve(qIds, docIds, tokenizer.vocabSize), [qIds, docIds, tokenizer])

  // only the vocabulary columns that occur somewhere, so the matrices stay readable
  const cols = useMemo(() => [...new Set([...docIds.flat(), ...qIds])].sort((a, b) => a - b), [docIds, qIds])
  const colLabels = cols.map((id) => displayToken(tokenizer.vocab[id]))
  const qRows = cols.flatMap((id, c) => (qIds.includes(id) ? [c] : []))
  // matrices are token × document (one column per document) so they stay narrow
  const countMat = Float64Array.from(cols.flatMap((id) => docIds.map((d) => d.filter((x) => x === id).length)))
  const vecs = [r.queryVec, ...r.docVecs]
  const tfidfMat = Float64Array.from(cols.flatMap((id) => vecs.map((v) => v[id])))
  const matches = qRows.flatMap((row) => set.docs.flatMap((_, d) => (r.docVecs[d][cols[row]] > 0 ? [[row, d + 1] as [number, number]] : [])))
  const top = r.order.slice(0, k)
  const topSet = new Set(top)
  const docLabels = set.docs.map((_, i) => `文書 ${i + 1}`)
  const docNums = set.docs.map((_, i) => String(i + 1))
  const N = set.docs.length
  const docBand = { label: `文書 1〜${N}`, color: 'var(--ink-3)' }
  const rowLabelWidth = lang === 'ja' ? 64 : 72

  const promptIds = [...top.flatMap((i) => [...docIds[i], EOS_ID]), ...qIds]
  const promptTokens = tokenizer.tokensOf(promptIds)
  const retrievedPos = new Set(promptTokens.flatMap((_, i) => (i < promptIds.length - qIds.length && promptTokens[i] !== EOS ? [i] : [])))

  const pretrained = lab.session.pretrainedParams
  const comparisons = useMemo(() => {
    if (!checkpoints) return null
    const params = checkpointParams(pretrained, checkpoints[ckIdx])
    return set.questions.map((q) => compare(params, tokenizer, q, set.docs[retrieve(tokenizer.encode(q.text), docIds, tokenizer.vocabSize).order[0]]))
  }, [pretrained, checkpoints, ckIdx, tokenizer, set, docIds])
  const current = comparisons?.[qIdx]
  const hits = comparisons?.filter((c) => c.correct).length ?? 0

  const controls = (showK: boolean, showCk = false) => (
    <>
      <StickyBar>
        <div className="row" style={{ alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
          <Seg label="質問" value={qIdx} options={set.questions.map((q, i) => ({ value: i, label: q.text }))} onChange={setQ} />
          {showK && <Seg label="取り出す数 k" value={k} options={[1, 2, 3].map((v) => ({ value: v, label: String(v) }))} onChange={setK} />}
          {showCk && checkpoints && <Seg label="追加学習" value={ckIdx} options={checkpoints.map((c, i) => ({ value: i, label: c.step ? c.step.toLocaleString() : '0（元）' }))} onChange={setCk} />}
        </div>
      </StickyBar>
      {showK && (
        <p className="controls-help" style={{ marginTop: 6 }}>
          <strong>k</strong>：類似度の高い順に、上から何件の文書を取り出してプロンプトに入れるか。
        </p>
      )}
      {showCk && checkpoints && (
        <p className="controls-help" style={{ marginTop: 6 }}>
          <strong>追加学習</strong>：文書を読んで答える練習を、元のモデルに何ステップ追加したか。0 は LLM-2〜8 と同じ元のモデル。
        </p>
      )}
    </>
  )

  const docList = (
    <div className="col" style={{ gap: 8 }}>
      {set.docs.map((_, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="mono muted small" style={{ width: 52, flexShrink: 0 }}>
            文書 {i + 1}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <TokenChips tokens={tokenizer.tokensOf(docIds[i])} />
          </div>
        </div>
      ))}
    </div>
  )

  const steps: Step[] = [
    {
      id: 'why',
      title: 'モデルが知らないことは答えられない',
      body: (
        <>
          <p>
            LLM の知識は、学習で決まった<Term id="parameter">重み</Term>の中にしかありません。学習データに無かったこと（昨日の出来事、社内の資料、個人のメモ）は、どれだけ上手に質問しても答えられません。
          </p>
          <p>
            そこで、質問に関係する文書を<strong>検索して取り出し（Retrieval）</strong>、<strong>プロンプトに書き足して（Augmented）</strong>から<strong>生成させる（Generation）</strong>。これが RAG です。重みは一切変えません。
          </p>
          <p>右の {set.docs.length} つの文は、このアプリの学習コーパスに<strong>含まれていない</strong>「メモ」です。これを検索の対象にします。</p>
        </>
      ),
    },
    {
      id: 'count',
      title: '文書をベクトルにする：トークンを数える',
      body: (
        <>
          <p>
            検索するには「似ているか」を数で測れる必要があります。いちばん素朴な方法は、文書に<strong>どのトークンが何回出たか</strong>を数えることです。<Term id="tokenizer">トークナイザ</Term>（文字列をトークンの列に切り分ける仕組み。LLM-2）で区切り、語彙の列ごとに回数を書き込むと、1 文書 = 1 本の<Term id="vector">ベクトル</Term>になります。
          </p>
          <p>右の表は 1 列が 1 文書のベクトルです。語順は捨てています（「ねこがいぬを」と「いぬがねこを」は同じベクトル）。それでも検索にはかなり使えます。行は、文書か質問に 1 回以上出るトークンだけに絞って表示しています。</p>
        </>
      ),
      formula: `count[d][w] = 文書 d にトークン w が出た回数`,
    },
    {
      id: 'idf',
      title: 'よく出るトークンは軽くする（IDF）',
      body: (
        <>
          <p>
            「の」「は」のように<strong>どの文書にも出るトークン</strong>は、一致しても手がかりになりません。そこで、出現する文書の数が多いトークンほど小さい重みを掛けます。これが IDF（逆文書頻度）で、回数 × IDF を <strong>TF-IDF</strong> と呼びます。
          </p>
          <p>棒グラフは各トークンの IDF です。{set.docs.length} 文書のうち 1 つにしか出ないトークンが最も重く、多くの文書に出るトークンほど軽くなっています。</p>
        </>
      ),
      formula: `idf[w] = log((N + 1) / (df[w] + 1)) + 1   （N = ${set.docs.length} 文書、df = w を含む文書数）`,
    },
    {
      id: 'query',
      title: '質問も同じ方法でベクトルにする',
      body: (
        <>
          <p>
            質問「{question.text}」も、<strong>文書と同じトークナイザ・同じ IDF</strong> でベクトルにします。同じ座標系に置かないと比べられないからです。
          </p>
          <p>左端の列（q）が質問ベクトル、1〜{set.docs.length} の列がそれぞれ文書 1〜{set.docs.length} のベクトルです。質問に出てこないトークンの行は薄くしてあり、朱枠は「質問と同じトークンを文書も持っている」セルです。朱枠が多く、しかも IDF の大きいトークンで一致している文書ほど似ていることになります。上のバーで質問を切り替えられます。</p>
        </>
      ),
      formula: `q[w] = count_q[w] · idf[w]`,
    },
    {
      id: 'score',
      title: '似ている順に並べて、上位 k 件を取り出す',
      body: (
        <>
          <p>
            質問ベクトルと各文書ベクトルの<strong>コサイン類似度</strong>を計算します。2 本の矢印のなす角の cos で、向きが同じなら 1、共通のトークンが 1 つも無ければ 0 です。文書の長さ（トークン数）に左右されないのが利点です。
          </p>
          <p>
            1 位は文書 {r.order[0] + 1}（類似度 {fmt(r.scores[r.order[0]], 2)}）でした。上位 {k} 件（朱）を取り出します。
            {r.order[0] === question.answer ? '答えが書いてある文書を正しく見つけています。' : ''}
          </p>
        </>
      ),
      formula: `cos(q, d) = (q · d) / (‖q‖ ‖d‖)`,
    },
    {
      id: 'prompt',
      title: 'プロンプトに書き足す',
      body: (
        <>
          <p>
            取り出した文書を、質問の<strong>前に</strong>並べてからモデルに渡します。モデルから見ると、答えがすでに<Term id="context">文脈</Term>の中に書いてある状態です。実際のシステムでは「次の資料だけを根拠に答えて」のような指示文も添えます。
          </p>
          <p>
            このミニモデルのトークンで数えると {promptIds.length} トークンになり、文脈長 {ctxLen} {promptIds.length > ctxLen ? 'を超えてしまいます' : 'にぎりぎり収まります'}。実際の LLM の文脈長は数万〜数十万トークンあり、何件もの文書を差し込めます。
          </p>
        </>
      ),
    },
    {
      id: 'answer',
      title: '実際に答えは変わるか：追加学習の量で比べる',
      body: !comparisons || !current || !checkpoints ? (
        <>
          <p>
            日本語版では、文書を読む練習を<strong>追加学習</strong>させたミニモデルで、文書を前に置くと答えが変わるかを実際に試せます。
          </p>
          <p>
            英語版では用意できませんでした。英語のトークナイザは語彙にない単語（mountain、tomorrow など）を 1 文字ずつに分けるため、文書と書き出しを合わせると文脈長 {ctxLen} を超えてしまい、文書を読む練習そのものができないからです。上部で日本語に切り替えると試せます。
          </p>
        </>
      ) : (
        <>
          <p>
            元のミニモデル（追加学習 0）は、1 文ずつの続きを当てる<Term id="pretrain">事前学習</Term>しかしていないので、前に文書を置いても答えはほとんど変わりません。そこで「⟨eos⟩ 文書 ⟨eos⟩ 同じ文」という形の例で<strong>追加学習</strong>させました。2 回目の文の後半は 1 回目を読まないと当てられないので、モデルは「文書から写す」ことを覚えます。半分は答えがランダムな文字列の例で、丸暗記では解けません。この章の 8 文書は学習に含めていません。実際の LLM が事前学習のあとに行う事後学習の、ごく小さな版です。
          </p>
          <p>
            このミニモデルは疑問文には答えられないので、「{current.q.cloze}」のような<strong>書き出しの続き</strong>を当てさせます。文書は第 5 ステップの検索で 1 位になった 1 件です（文脈長 {ctxLen} に収まるのが 1 件だけのため）。
          </p>
          <p>
            上のバーで追加学習の量を切り替えてみてください。いまは {checkpoints[ckIdx].step.toLocaleString()} ステップで、{comparisons.length} 問中 <strong>{hits} 問</strong>で文書どおりの答えを出しています。下のグラフの「写す力」は、学習に一度も出てこなかった組み合わせの文をどれだけ写せるかで、追加学習の量とともに上がっていきます。
          </p>
          <p>
            最後まで直らない問題もあります。「あしたは」の続きは、学習コーパスに「あしたははれる」がそのまま入っているため、覚えた知識が文書に勝ってしまいます（大きな LLM でも起きる、知識と文書の食い違いです）。「たま」のような名前を写すのも、この大きさのモデルには難しいようです。
          </p>
        </>
      ),
    },
    {
      id: 'flow',
      title: '全体の流れと、このミニモデルの限界',
      body: (
        <>
          <p>
            まとめると、RAG は<strong>検索</strong>（ここまでの計算）と<strong>生成</strong>（LLM-2〜6 の計算）をつなぐだけの仕組みです。実際のシステムでは、トークンの回数の代わりに、意味の近さを学習した専用の<Term id="embedding">埋め込み</Term>モデルでベクトルを作ることが多く、言い換え（「くろい」と「黒色」）にも強くなります。
          </p>
          <p>
            <strong>限界：</strong>日本語版の前のステップで見たとおり、このミニモデルは追加学習をしても、文書を正しく読めるのは一部の問題だけです。大きな LLM は、桁違いの大きさと学習量、それに指示に従わせる追加の学習によって、長い文書から必要な部分を探して読み取れるようになっています。
          </p>
        </>
      ),
    },
  ]

  const pipeline: [string, string, string][] = [
    ['1', '質問', question.text],
    ['2', 'ベクトル化', 'TF-IDF（実際は埋め込みモデル）'],
    ['3', '検索', `コサイン類似度で上位 ${k} 件`],
    ['4', 'プロンプト', '文書 ＋ 質問'],
    ['5', 'LLM で生成', 'LLM-2〜6 の計算'],
  ]

  return (
    <ChapterLayout
      title="RAG（検索拡張生成）"
      lede="学習し直さずに、モデルが知らない知識を使わせる。質問に合う文書を探して、プロンプトに書き足します。"
      purpose={
        <>
          モデルの知識は学習時点の<Term id="parameter">重み</Term>で固定されています。新しい情報を覚えさせるには学習し直す必要がありますが、それは重くて時間がかかります。RAG は重みに触れず、<strong>入力の側に答えの材料を入れてしまう</strong>ことで、この問題を回避します。
        </>
      }
      io="質問文 ＋ 文書集 → 関係する文書を前に付けたプロンプト"
      terms={['rag', 'tokenizer', 'tfidf', 'cosine', 'vector', 'embedding', 'context', 'pretrain']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'why')
          return (
            <div className="col">
              <div className="card">
                <div className="card-title">検索の対象にする文書（学習コーパスには無い）</div>
                {docList}
              </div>
            </div>
          )
        if (step.id === 'count')
          return (
            <div className="row" style={{ flexWrap: 'wrap', alignItems: 'flex-start' }}>
              <Heatmap values={countMat} rows={cols.length} cols={set.docs.length} mode="sequential" cell={30} cellH={20} rowLabels={colLabels} rowLabelWidth={rowLabelWidth} colLabels={docNums} colGroups={[{ ...docBand, from: 0, to: N }]} showValues digits={0} legend title={`トークンの出現回数（${cols.length} トークン × ${set.docs.length} 文書）`} rowName="トークン" colName="文書" rowAxis="トークン（文書か質問に出るものだけ）" colAxis="文書（1 列 = 1 文書）" tag="computed" />
              <div className="card grow">{docList}</div>
            </div>
          )
        if (step.id === 'idf')
          return (
            <div className="col">
              <BarChart
                items={cols.flatMap((id, c) => (docIds.some((d) => d.includes(id)) ? [{ label: colLabels[c], value: r.idf[id] }] : [])).sort((a, b) => b.value - a.value)}
                min={0}
                width={420}
                barHeight={12}
                gap={4}
                title="トークンごとの IDF（大きいほど手がかりになる）"
              />
            </div>
          )
        if (step.id === 'query')
          return (
            <div className="col">
              {controls(false)}
              <div className="card">
                <div className="card-title">質問のトークン</div>
                <TokenChips tokens={tokenizer.tokensOf(qIds)} />
              </div>
              <Heatmap values={tfidfMat} rows={cols.length} cols={set.docs.length + 1} mode="sequential" cell={30} cellH={20} rowLabels={colLabels} rowLabelWidth={rowLabelWidth} colLabels={['q', ...docNums]} colGroups={[{ label: '質問', from: 0, to: 1, color: 'var(--accent)' }, { ...docBand, from: 1, to: N + 1 }]} highlightRows={qRows} highlightCells={matches} dimOthers showValues digits={1} legend title="質問ベクトル q と文書ベクトル（TF-IDF）" rowName="トークン" rowAxis="トークン（薄い行 = 質問に出ない）" colAxis={`左端 q = 質問、1〜${N} = 文書の番号（列番号ではない）`} tag="computed" />
            </div>
          )
        if (step.id === 'score')
          return (
            <div className="col">
              {controls(true)}
              <BarChart
                items={r.order.map((i) => ({ label: docLabels[i], value: r.scores[i], sub: set.docs[i], highlight: topSet.has(i), muted: !topSet.has(i) }))}
                min={0}
                max={1}
                width={470}
                labelWidth={56}
                valueWidth={200}
                title={`質問「${question.text}」とのコサイン類似度`}
              />
            </div>
          )
        if (step.id === 'answer')
          return (
            <div className="col">
              {controls(false, true)}
              {comparisons && current && checkpoints ? (
                <AnswerView current={current} all={comparisons} checkpoints={checkpoints} ckIdx={ckIdx} tok={tokenizer} />
              ) : (
                <div className="card muted">英語版ではこのステップの実験はありません（理由は説明を参照）。</div>
              )}
            </div>
          )
        if (step.id === 'prompt')
          return (
            <div className="col">
              {controls(true)}
              <div className="card">
                <div className="card-title">モデルに渡す文章</div>
                <div className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.9 }}>
                  {'【資料】\n'}
                  {top.map((i) => `・${set.docs[i]}\n`).join('')}
                  {`【質問】${question.text}`}
                </div>
              </div>
              <Arrow>↓ このミニモデルのトークンにすると（朱 = 検索で足した部分、⟨eos⟩ = 区切り）</Arrow>
              <div className="card">
                <div className="card-title">
                  {promptIds.length} トークン ／ 文脈長 {ctxLen}
                </div>
                <TokenChips tokens={promptTokens} newSet={retrievedPos} positions dimSet={new Set(promptTokens.flatMap((_, i) => (i >= ctxLen ? [i] : [])))} />
                {promptIds.length > ctxLen && <p className="muted small">薄い部分は文脈長 {ctxLen} を超えていて、このミニモデルには入りきりません。</p>}
              </div>
            </div>
          )
        return (
          <div className="col">
            <div className="pipeline">
              {pipeline.map(([n, name, shape]) => (
                <div key={n} className="pipe-node">
                  <span className="num">{n}</span>
                  <span className="name">{name}</span>
                  <span className="shape">{shape}</span>
                </div>
              ))}
            </div>
            <div className="card">
              <div className="card-title">学習し直す方法（fine-tuning）との違い</div>
              <table className="data">
                <thead>
                  <tr>
                    <th></th>
                    <th>RAG</th>
                    <th>学習し直す</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>知識の置き場所</td>
                    <td>外部の文書（プロンプトに入れる）</td>
                    <td>重みの中</td>
                  </tr>
                  <tr>
                    <td>情報の更新</td>
                    <td>文書を差し替えるだけ</td>
                    <td>もう一度学習が必要</td>
                  </tr>
                  <tr>
                    <td>根拠の提示</td>
                    <td>取り出した文書を見せられる</td>
                    <td>難しい</td>
                  </tr>
                  <tr>
                    <td>弱点</td>
                    <td>検索を外すと答えも外す。文脈長を使う</td>
                    <td>計算が重い。古い知識が残る</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
