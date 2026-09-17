import type { ReactNode } from 'react'
import { GlossaryList } from '../../content/glossary'
import { CHAPTER_META, chapterLabel } from '../chapterMeta'
import { useNav } from '../state/NavProvider'
import { useStepper, type Step } from '../state/useStepper'

interface Props {
  num: string
  title: string
  lede: ReactNode
  steps: Step[]
  /** extra controls rendered above the stepper (selectors, sliders) */
  aside?: ReactNode
  /** why this stage exists (shown in a box under the lede) */
  purpose?: ReactNode
  /** "input → output" of this stage */
  io?: string
  /** glossary ids for the terms used in this chapter */
  terms?: string[]
  children: (step: Step, index: number) => ReactNode
}

export function ChapterLayout({ num, title, lede, steps, aside, purpose, io, terms, children }: Props) {
  const nav = useNav()
  const idx = CHAPTER_META.findIndex((c) => c.id === nav.current)
  const nextChapter = idx >= 0 && idx < CHAPTER_META.length - 1 ? CHAPTER_META[idx + 1] : null
  const s = useStepper(steps.length, true, nextChapter ? () => nav.go(nextChapter.id) : undefined)
  const atEnd = s.index >= steps.length - 1
  const step = steps[s.index]
  return (
    <>
      <section className="stage">
        {nav.returnTo && (
          <button className="return-bar" onClick={nav.back}>
            ← {chapterLabel(nav.returnTo.id)}に戻る
          </button>
        )}
        <header className="stage-head">
          <div className="chapter-num">CHAPTER {num}</div>
          <h2>{title}</h2>
          <p className="lede">{lede}</p>
          {(purpose || io) && (
            <div className="purpose">
              {purpose && (
                <div>
                  <span className="purpose-label">なぜやるのか</span>
                  <div>{purpose}</div>
                </div>
              )}
              {io && (
                <div>
                  <span className="purpose-label">入力 → 出力</span>
                  <div className="mono">{io}</div>
                </div>
              )}
            </div>
          )}
        </header>
        <div key={step.id} className="fade-in">
          {children(step, s.index)}
        </div>
        {terms && terms.length > 0 && (
          <section className="glossary-section">
            <h4>この章の用語</h4>
            <p className="muted small">本文の点線付きの語をクリックするとここに飛びます。「詳しく」を押すと解説の章へ移動し、上部のバーで戻れます。</p>
            <GlossaryList ids={terms} current={nav.current} />
          </section>
        )}
      </section>
      <aside className="explain">
        <div className="explain-body">
          <div className="step-label">
            STEP {s.index + 1} / {steps.length}
          </div>
          <h3>{step.title}</h3>
          <div key={step.id} className="fade-in">
            {step.body}
            {step.formula && <div className="formula">{step.formula}</div>}
          </div>
          {aside && <div style={{ marginTop: 20 }}>{aside}</div>}
        </div>
        <div className="controls">
          <div className="controls-row">
            <button className="ctl" onClick={s.prev} disabled={s.index === 0} aria-label="前のステップ">
              ◀
            </button>
            <button
              className={'ctl ' + (atEnd && nextChapter ? 'accent' : 'primary')}
              onClick={s.next}
              disabled={atEnd && !nextChapter}
              aria-label={atEnd && nextChapter ? `次の章へ：${nextChapter.title}` : '次のステップ'}
              title={atEnd && nextChapter ? `次の章「${nextChapter.title}」へ` : '次のステップ'}
            >
              {atEnd && nextChapter ? <span style={{ letterSpacing: '-0.25em', marginRight: '0.25em' }}>▶▶</span> : '▶'}
            </button>
            <div className="dots" role="tablist">
              {steps.map((st, i) => (
                <button key={st.id} role="tab" aria-current={i === s.index} className={i < s.index ? 'done' : ''} onClick={() => s.setIndex(i)} title={st.title} />
              ))}
            </div>
            <span className="step-count">
              {s.index + 1} / {steps.length}
            </span>
          </div>
          <div className="controls-row" style={{ marginTop: 10 }}>
            <button className="ctl" onClick={s.togglePlay}>
              {s.playing ? '⏸ 一時停止' : '自動再生'}
            </button>
            <label className="speed">
              <span>速さ</span>
              <input type="range" min={600} max={4000} step={200} value={5000 - s.interval} onChange={(e) => s.setInterval(5000 - Number(e.target.value))} />
            </label>
            <span className="muted small" style={{ marginLeft: 'auto' }}>
              {atEnd && nextChapter ? `▶▶ で第 ${nextChapter.num} 章「${nextChapter.title}」へ` : '← → キーでも移動'}
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}
