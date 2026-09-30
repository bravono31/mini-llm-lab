import { useMemo } from 'react'
import type { Checkpoint, Comparison, Probe } from '../../engine/rag'
import { displayToken, EOS, type Tokenizer } from '../../engine/tokenizer'
import { BarChart } from '../viz/BarChart'
import { fmtPct } from '../viz/colors'
import { TokenChips } from '../viz/TokenChips'

function ProbeCard({ title, probe: pr, answerId, tok, docLen }: { title: string; probe: Probe; answerId: number | null; tok: Tokenizer; docLen: number }) {
  const tokens = tok.tokensOf(pr.prompt)
  const docPos = new Set(tokens.flatMap((t, i) => (i > 0 && i <= docLen && t !== EOS ? [i] : [])))
  return (
    <div className="card grow" style={{ minWidth: 260 }}>
      <div className="card-title">{title}</div>
      <TokenChips tokens={tokens} newSet={docPos} />
      <BarChart
        items={pr.top.map(({ id, p }) => ({ label: displayToken(tok.vocab[id]), value: p, highlight: id === answerId, muted: id !== answerId }))}
        min={0}
        max={1}
        width={300}
        labelWidth={70}
        valueWidth={50}
        format={(v) => fmtPct(v, 0)}
        title="次のトークンの確率（上位 5、朱 = 文書にある答え）"
      />
      <div style={{ marginTop: 8 }}>
        続き：<strong className="mono">{pr.text || '（すぐに ⟨eos⟩）'}</strong>
      </div>
    </div>
  )
}

export function AnswerView({ current, all, checkpoints, ckIdx, tok }: { current: Comparison; all: Comparison[]; checkpoints: Checkpoint[]; ckIdx: number; tok: Tokenizer }) {
  const docLen = current.withDoc.prompt.length - tok.encode(current.q.cloze).length - 2
  const accItems = useMemo(
    () => checkpoints.map((c, i) => ({ label: `${c.step.toLocaleString()}`, value: c.copyAcc / 100, highlight: i === ckIdx, muted: i !== ckIdx })),
    [checkpoints, ckIdx],
  )
  return (
    <div className="col">
      <div className="row">
        <ProbeCard title="文書なし" probe={current.noDoc} answerId={current.answerId} tok={tok} docLen={0} />
        <ProbeCard title="文書あり（検索で取り出した 1 件を前に置く）" probe={current.withDoc} answerId={current.answerId} tok={tok} docLen={docLen} />
      </div>
      <div className="card">
        <div className="card-title">5 問すべての結果（追加学習 {checkpoints[ckIdx].step.toLocaleString()} ステップ）</div>
        <table className="data">
          <thead>
            <tr>
              <th>書き出し</th>
              <th>文書なし</th>
              <th>文書あり</th>
              <th>文書にある答え</th>
            </tr>
          </thead>
          <tbody>
            {all.map((c) => (
              <tr key={c.q.cloze} className={c.q === current.q ? 'hi' : ''}>
                <td>{c.q.cloze}</td>
                <td className="mono">{c.noDoc.text || '—'}</td>
                <td className="mono">
                  {c.withDoc.text || '—'} {c.correct ? '✓' : '✗'}
                </td>
                <td className="mono">{c.answer}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <BarChart items={accItems} min={0} max={1} width={420} labelWidth={70} valueWidth={50} format={(v) => fmtPct(v, 0)} title="写す力：学習に使っていない合成の文 200 個のうち、文書から正しく写せた割合（行 = 追加学習のステップ数）" />
    </div>
  )
}
