import { useMemo, useState } from 'react'
import { Term } from '../../content/glossary'
import { displayToken, mergeSyms, preTokenize, type Pair } from '../../engine/tokenizer'
import { ChapterLayout } from '../shell/ChapterLayout'
import { useLab } from '../state/LabProvider'
import type { Step } from '../state/useStepper'
import { TokenChips } from '../viz/TokenChips'

/** Tokenise with only the first k merges (to replay BPE training). */
function applyMerges(chunks: string[][], merges: Pair[]): string[][] {
  const ranks = new Map(merges.map((m, i) => [m[0] + '∷' + m[1], i]))
  let cur = chunks.map((c) => [...c])
  for (;;) {
    let best: { pair: Pair; rank: number } | null = null
    for (const ch of cur)
      for (let i = 0; i + 1 < ch.length; i++) {
        const r = ranks.get(ch[i] + '∷' + ch[i + 1])
        if (r !== undefined && (best === null || r < best.rank)) best = { pair: [ch[i], ch[i + 1]], rank: r }
      }
    if (!best) return cur
    const [a, b] = best.pair
    cur = cur.map((c) => mergeSyms(c, a, b, a + b))
  }
}

function SubStepper({ index, count, setIndex, label }: { index: number; count: number; setIndex: (i: number) => void; label: string }) {
  return (
    <div className="row" style={{ alignItems: 'center', gap: 10 }}>
      <button className="ctl" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
        ◀
      </button>
      <button className="ctl primary" onClick={() => setIndex(Math.min(count, index + 1))} disabled={index >= count}>
        ▶
      </button>
      <input type="range" min={0} max={count} value={index} onChange={(e) => setIndex(Number(e.target.value))} style={{ width: 200, accentColor: 'var(--accent)' }} />
      <span className="mono muted small">
        {label} {index} / {count}
      </span>
      <button className="ctl" onClick={() => setIndex(count)} disabled={index >= count}>
        最後まで
      </button>
    </div>
  )
}

export function Tokenize(_: { onNavigate: (id: string) => void }) {
  const lab = useLab()
  const { tokenizer, corpus, trace } = lab
  const merges = tokenizer.merges
  const baseCount = tokenizer.vocab.length - merges.length
  const [k, setK] = useState(0)
  const [s, setS] = useState(0)

  const stats = useMemo(() => {
    const chars = corpus.sentences.join('')
    return { sentences: corpus.sentences.length, chars: chars.length, unique: new Set(chars.replace(/\s/g, '')).size }
  }, [corpus])

  const sampleText = lab.input.trim() ? lab.input : corpus.defaultInput
  const sampleChunks = useMemo(() => preTokenize(sampleText, tokenizer.lang).map((c) => [...c]), [sampleText, tokenizer.lang])
  const replay = useMemo(() => applyMerges(sampleChunks, merges.slice(0, k)), [sampleChunks, merges, k])
  const record = k > 0 ? tokenizer.mergeLog[k - 1] : null
  const nextRecord = k < merges.length ? tokenizer.mergeLog[k] : null

  const encState = s === 0 ? trace.chunks : trace.steps[s - 1].chunks
  const encMerged = s === 0 ? null : trace.steps[s - 1].merged
  const encFlat = encState.flat()
  const encNew = new Set<number>()
  if (encMerged) encFlat.forEach((t, i) => t === encMerged[0] + encMerged[1] && encNew.add(i))

  const origin = (id: number) => {
    if (id < baseCount) return id < 2 ? '特殊' : '文字'
    const m = merges[id - baseCount]
    return `#${id - baseCount + 1}: ${displayToken(m[0])} + ${displayToken(m[1])}`
  }

  const steps: Step[] = [
    {
      id: 'corpus',
      title: 'コーパス（学習テキスト）',
      body: (
        <>
          <p>
            モデルが学ぶのは、この <strong>{stats.sentences} 文</strong>だけです。トークナイザもモデルも、ここに出てくる文字と語の並びから作られます。
          </p>
          <p>本物の LLM はこれが数兆文字になりますが、やることは同じ。「文字の並びの規則性」を数値に写します。</p>
        </>
      ),
    },
    {
      id: 'base',
      title: '初期語彙 = 文字',
      body: (
        <>
          <p>
            出発点は、コーパスに現れる<strong>すべての文字</strong>（{baseCount - 2} 種）です。これに文の終わり ⟨eos⟩ と未知文字 ⟨unk⟩ を加えて初期語彙とします。
          </p>
          <p>文字だけでも文は表せますが、1 文字ずつでは列が長く、意味の単位にもなりません。そこで、よく隣り合う文字を結合して「語彙」を育てます。</p>
        </>
      ),
    },
    {
      id: 'bpe',
      title: 'BPE：頻出ペアを結合していく',
      body: (
        <>
          <p>
            <Term id="bpe">Byte Pair Encoding（BPE）</Term>は単純な手続きです。コーパス中で<strong>いちばん多く隣り合うトークンの組</strong>を数え、それを 1 つの新トークンに結合する。これを語彙が目標サイズ（{tokenizer.vocabSize}
            ）になるまで繰り返します。
          </p>
          <p>▶ で 1 マージずつ進めると、頻度表の 1 位が結合されて語彙に加わり、下の例文の分かち方が変わっていきます。</p>
        </>
      ),
      formula: `merge k: (a, b) = argmax count(a·b)\nvocab ← vocab ∪ { ab }`,
    },
    {
      id: 'encode',
      title: '入力文をトークンにする',
      body: (
        <>
          <p>
            新しい文は、まず文字に分け、学習したマージを<strong>学習時と同じ順番</strong>（ランクの小さい順）で適用します。順番が固定されているので、同じ文は常に同じトークン列になります。
          </p>
          <p>
            ▶ で 1 マージずつ適用していきます。{tokenizer.lang === 'en' ? '英語では単語（␣付き）の内側でだけ結合します。' : '日本語は分かち書きがないため、文全体の中で結合します。'}
          </p>
        </>
      ),
    },
    {
      id: 'ids',
      title: 'トークン → 整数 ID',
      body: (
        <>
          <p>
            モデルが受け取るのは文字ではなく、<strong>語彙表の行番号（ID）</strong>の列です。ここから先、文は {lab.ids.length} 個の整数として扱われます。
          </p>
          <p>ID そのものに意味はありません。次章の埋め込み表で、各 ID が {lab.params.config.dModel} 次元のベクトルに置き換えられます。</p>
        </>
      ),
    },
    {
      id: 'vocab',
      title: '語彙表',
      body: (
        <>
          <p>
            完成した語彙表です。前半は文字、後半は BPE で結合された語です。「頻度」はコーパス中の出現回数で、頻出する語ほど早く結合されて短い ID になっています。
          </p>
          <p>語彙外の文字（コーパスにない文字）は ⟨unk⟩（ID 1）になり、モデルは区別できません。本物の LLM がバイト単位の語彙を持つのは、この「未知」をなくすためです。</p>
        </>
      ),
    },
  ]

  return (
    <ChapterLayout
      num="02"
      title="トークン化"
      lede="文をモデルが扱える整数の列に変える。語彙は、コーパスから BPE で育てます。"
      purpose={
        <>
          コンピュータは文字を直接計算できないので、文を<Term id="vocab">語彙表</Term>の番号（<Term id="id">ID</Term>）の列に変えます。1 文字ずつ番号にしてもよいのですが、列が長くなり「ねこ」のような意味の単位も壊れます。そこで <Term id="bpe">BPE</Term> で頻出する文字列を 1 つの<Term id="token">トークン</Term>にまとめ、短く・意味の単位に近い列にします。
        </>
      }
      io={`文字列 → トークン ID の列（整数、最大 ${lab.params.config.ctxLen} 個）`}
      terms={['token', 'tokenizer', 'bpe', 'vocab', 'id', 'eos', 'context']}
      steps={steps}
    >
      {(step) => {
        if (step.id === 'corpus')
          return (
            <div className="col">
              <div className="row">
                <div className="stat">
                  <span className="label">文</span>
                  <span className="value">{stats.sentences}</span>
                </div>
                <div className="stat">
                  <span className="label">文字数</span>
                  <span className="value">{stats.chars}</span>
                </div>
                <div className="stat">
                  <span className="label">文字の種類</span>
                  <span className="value">{stats.unique}</span>
                </div>
              </div>
              <div className="card" style={{ lineHeight: 2.1 }}>
                {corpus.sentences.map((s, i) => (
                  <span key={i} className="mono" style={{ display: 'inline-block', marginRight: 18 }}>
                    {s}
                    <span className="muted"> ⟨eos⟩</span>
                  </span>
                ))}
              </div>
            </div>
          )
        if (step.id === 'base')
          return (
            <div className="col">
              <div className="viz-title">初期語彙 {baseCount} 個（特殊 2 + 文字 {baseCount - 2}）</div>
              <TokenChips tokens={tokenizer.vocab.slice(0, baseCount)} ids={[...Array(baseCount).keys()]} />
            </div>
          )
        if (step.id === 'bpe')
          return (
            <div className="col">
              <SubStepper index={k} count={merges.length} setIndex={setK} label="マージ" />
              <div className="row">
                <div className="card" style={{ minWidth: 300 }}>
                  <div className="card-title">{nextRecord ? `隣接ペアの頻度（マージ ${k + 1} の前）` : '目標語彙数に到達'}</div>
                  {nextRecord ? (
                    <table className="data">
                      <thead>
                        <tr>
                          <th>ペア</th>
                          <th>回数</th>
                        </tr>
                      </thead>
                      <tbody>
                        {nextRecord.topPairs.map((pc, i) => (
                          <tr key={i} className={i === 0 ? 'hi' : ''}>
                            <td className="mono">
                              {displayToken(pc.pair[0])} + {displayToken(pc.pair[1])}
                            </td>
                            <td className="num">{pc.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="muted">これ以上のマージは行いません。</p>
                  )}
                </div>
                <div className="card grow">
                  <div className="card-title">{record ? `マージ ${k}：新しいトークン` : 'まだ結合していません'}</div>
                  {record && (
                    <div className="row" style={{ alignItems: 'center' }}>
                      <TokenChips tokens={[record.pair[0], record.pair[1]]} />
                      <span className="muted">→</span>
                      <TokenChips tokens={[record.token]} ids={[baseCount + k - 1]} newSet={new Set([0])} />
                      <span className="muted small">（{record.count} 回）</span>
                    </div>
                  )}
                  <div className="muted small" style={{ marginTop: 12 }}>
                    語彙サイズ {baseCount + k} / {tokenizer.vocabSize}
                  </div>
                </div>
              </div>
              <div className="card">
                <div className="card-title">例文の分かち方（先頭 {k} マージを適用）</div>
                <TokenChips tokens={replay.flat()} newSet={record ? new Set(replay.flat().map((t, i) => (t === record.token ? i : -1)).filter((i) => i >= 0)) : undefined} />
              </div>
            </div>
          )
        if (step.id === 'encode')
          return (
            <div className="col">
              <SubStepper index={s} count={trace.steps.length} setIndex={setS} label="適用" />
              <div className="card">
                <div className="card-title">
                  {s === 0 ? '文字に分割' : `マージ #${(trace.steps[s - 1].rank ?? 0) + 1} を適用：${displayToken(encMerged![0])} + ${displayToken(encMerged![1])} → ${displayToken(encMerged![0] + encMerged![1])}`}
                </div>
                <TokenChips tokens={encFlat} newSet={encNew} />
              </div>
              {trace.unknownChars.length > 0 && <p className="input-note">語彙にない文字 {trace.unknownChars.join(' ')} は ⟨unk⟩ になります。</p>}
            </div>
          )
        if (step.id === 'ids')
          return (
            <div className="col">
              <TokenChips tokens={trace.tokens} ids={trace.ids} positions />
              <div className="formula">ids = [{trace.ids.join(', ')}]</div>
              {lab.truncated && <p className="input-note">文脈長 {lab.params.config.ctxLen} を超えたため、モデルには先頭 {lab.params.config.ctxLen} トークンだけを渡します。</p>}
            </div>
          )
        return (
          <div className="card" style={{ maxHeight: 640, overflow: 'auto' }}>
            <table className="data">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>トークン</th>
                  <th>頻度</th>
                  <th>由来</th>
                </tr>
              </thead>
              <tbody>
                {tokenizer.vocab.map((t, id) => (
                  <tr key={id} className={lab.ids.includes(id) ? 'hi' : ''}>
                    <td className="num">{id}</td>
                    <td className="mono">{displayToken(t)}</td>
                    <td className="num">{tokenizer.tokenFreq[id]}</td>
                    <td className="mono muted">{origin(id)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      }}
    </ChapterLayout>
  )
}
