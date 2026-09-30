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
  count: number
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

export function useStepper(count: number, keyboard = true, onOverflow?: () => void, onUnderflow?: () => void, initial = 0): Stepper {
  const [index, setIndexState] = useState(() => Math.max(0, Math.min(count - 1, initial)))
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
      }
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [keyboard, next, prev])

  return { index, setIndex, next, prev, count }
}
