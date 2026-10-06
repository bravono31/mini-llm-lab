import type { Word } from '../../engine/classifier'
import { useMl } from '../state/MlProvider'
import { BarChart } from '../viz/BarChart'
import { fmtPct } from '../viz/colors'

/** Shared bits of the machine-learning chapters. */

export const CLASS_COLORS = ['var(--accent)', 'var(--indigo)', 'var(--ink-2)']

export function useClassNames(): string[] {
  const { ds } = useMl()
  return ds.classes.map((c) => ds.data.classNames[c])
}

/** feature words as normal chips, particles / articles dimmed, words outside the dictionary marked */
export function WordChips({ words }: { words: Word[] }) {
  if (!words.length) return <span className="muted small">（空）</span>
  return (
    <div className="chips">
      {words.map((w, i) => (
        <span key={i} className={'chip' + (w.kind === 'stop' ? ' dim' : w.kind === 'unknown' ? ' special' : '')} title={w.kind === 'feature' ? '特徴に使う語' : w.kind === 'stop' ? '助詞など：特徴に使わない' : '辞書にない語：無視される'}>
          <span>{w.text}</span>
          <span className="id">{w.kind === 'feature' ? '特徴' : w.kind === 'stop' ? '捨てる' : '辞書外'}</span>
        </span>
      ))}
    </div>
  )
}

export function ClassBadge({ c }: { c: number }) {
  const names = useClassNames()
  return (
    <span className="class-badge" style={{ color: CLASS_COLORS[c], borderColor: CLASS_COLORS[c] }}>
      {names[c]}
    </span>
  )
}

export function ProbBars({ probs, title, width = 360 }: { probs: ArrayLike<number>; title?: string; width?: number }) {
  const names = useClassNames()
  let best = 0
  for (let c = 1; c < probs.length; c++) if (probs[c] > probs[best]) best = c
  return <BarChart title={title} items={names.map((label, c) => ({ label, value: probs[c], color: CLASS_COLORS[c], highlight: c === best }))} min={0} max={1} width={width} format={(v) => fmtPct(v)} />
}

/** "the classifier says …" card for the top-bar input */
export function InputVerdict() {
  const ml = useMl()
  const nFeat = ml.words.filter((w) => w.kind === 'feature').length
  return (
    <div className="card">
      <div className="card-title">上部の入力文を分類すると</div>
      <WordChips words={ml.words} />
      {nFeat === 0 && <p className="muted small">辞書にある語が 1 つもないので手がかりがなく、切片 b（学習データに多いクラスほど大きい）だけで確率が決まります。</p>}
      <div style={{ marginTop: 10 }}>
        <ProbBars probs={ml.probs} />
      </div>
    </div>
  )
}
