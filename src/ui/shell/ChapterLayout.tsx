import { useEffect, useLayoutEffect, useRef, type ReactNode } from 'react'
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
  const prevChapter = idx > 0 ? CHAPTER_META[idx - 1] : null
  const s = useStepper(
    steps.length,
    true,
    nextChapter ? () => nav.go(nextChapter.id) : undefined,
    prevChapter ? () => nav.go(prevChapter.id, 'end') : undefined,
    nav.entry === 'end' ? steps.length - 1 : 0,
  )
  const atEnd = s.index >= steps.length - 1
  const atStart = s.index === 0
  const step = steps[s.index]
  const bodyRef = useRef<HTMLDivElement>(null)
  const controlsRef = useRef<HTMLDivElement>(null)
  const firstRender = useRef(true)

  // sticky offsets on narrow screens depend on the (wrapping) top bar and the step bar heights
  useLayoutEffect(() => {
    const topbar = document.querySelector<HTMLElement>('.topbar')
    const controls = controlsRef.current
    if (!topbar || !controls) return
    const root = document.documentElement.style
    const update = () => {
      root.setProperty('--topbar-h', `${topbar.offsetHeight}px`)
      root.setProperty('--stepbar-h', `${controls.offsetHeight}px`)
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(topbar)
    ro.observe(controls)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    const body = bodyRef.current
    if (!body) return
    body.scrollTop = 0
    if (!matchMedia('(max-width: 1100px)').matches) {
      window.scrollTo({ top: 0 })
      return
    }
    const pinned = parseFloat(getComputedStyle(body).scrollMarginTop) || 0
    if (body.getBoundingClientRect().top < pinned) body.scrollIntoView({ block: 'start' })
  }, [s.index])
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
        <div className="explain-body" ref={bodyRef}>
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
        <div className="controls" ref={controlsRef}>
          <div className="controls-row">
            <button
              className="ctl"
              onClick={s.prev}
              disabled={atStart && !prevChapter}
              aria-label={atStart && prevChapter ? `前の章の最後へ：${prevChapter.title}` : '前のステップ'}
              title={atStart && prevChapter ? `前の章「${prevChapter.title}」の最後へ` : '前のステップ'}
            >
              {atStart && prevChapter ? <span style={{ letterSpacing: '-0.25em', marginRight: '0.25em' }}>◀◀</span> : '◀'}
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
          <div className="controls-row controls-sub">
            <span className="muted small">
              {atEnd && nextChapter ? `▶▶ で第 ${nextChapter.num} 章「${nextChapter.title}」へ` : atStart && prevChapter ? `◀◀ で第 ${prevChapter.num} 章「${prevChapter.title}」の最後へ` : '← → キーでも移動'}
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}
