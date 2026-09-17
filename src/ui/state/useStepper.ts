import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

export interface Step {
  id: string
  title: string
  body: ReactNode
  formula?: string
}

export interface Stepper {
  index: number
  setIndex: (i: number) => void
  next: () => void
  prev: () => void
  playing: boolean
  togglePlay: () => void
  /** milliseconds per step while playing */
  interval: number
  setInterval: (ms: number) => void
  count: number
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

export function useStepper(count: number, keyboard = true, onOverflow?: () => void, onUnderflow?: () => void, initial = 0): Stepper {
  const [index, setIndexState] = useState(() => Math.max(0, Math.min(count - 1, initial)))
  const [playing, setPlaying] = useState(false)
  const [interval, setIntervalMs] = useState(1800)
  const indexRef = useRef(index)
  indexRef.current = index

  const setIndex = useCallback((i: number) => setIndexState(Math.max(0, Math.min(count - 1, i))), [count])
  const next = useCallback(() => {
    const i = indexRef.current
    if (i >= count - 1) {
      onOverflow?.()
      return
    }
    indexRef.current = i + 1
    setIndexState(i + 1)
  }, [count, onOverflow])
  const prev = useCallback(() => {
    const i = indexRef.current
    if (i <= 0) {
      onUnderflow?.()
      return
    }
    indexRef.current = i - 1
    setIndexState(i - 1)
  }, [onUnderflow])
  const togglePlay = useCallback(() => {
    setPlaying((p) => {
      if (!p && index >= count - 1) setIndexState(0)
      return !p
    })
  }, [index, count])

  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      setIndexState((i) => {
        if (i >= count - 1) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, interval)
    return () => window.clearInterval(id)
  }, [playing, interval, count])

  useEffect(() => {
    if (!keyboard) return
    const on = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        next()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        prev()
      } else if (e.key === ' ') {
        e.preventDefault()
        togglePlay()
      }
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [keyboard, next, prev, togglePlay])

  return { index, setIndex, next, prev, playing, togglePlay, interval, setInterval: setIntervalMs, count }
}
