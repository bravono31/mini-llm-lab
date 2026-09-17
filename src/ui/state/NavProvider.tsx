import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { CHAPTER_META } from '../chapterMeta'

export interface ReturnTo {
  id: string
  /** where on the page to scroll back to (element id), optional */
  anchor?: string
}

export type Entry = 'start' | 'end'

interface Nav {
  current: string
  /** which step the current chapter should open on */
  entry: Entry
  /** plain navigation (clears any "return to" state) */
  go: (id: string, entry?: Entry) => void
  /** navigate from a glossary link: remembers where we came from */
  goFrom: (to: string, anchor?: string) => void
  returnTo: ReturnTo | null
  back: () => void
}

const Ctx = createContext<Nav | null>(null)

function readHash(): string {
  const h = location.hash.replace('#', '')
  return CHAPTER_META.some((c) => c.id === h) ? h : CHAPTER_META[0].id
}

export function NavProvider({ children }: { children: ReactNode }) {
  const [current, setCurrent] = useState(readHash)
  const [returnTo, setReturnTo] = useState<ReturnTo | null>(null)
  const [entry, setEntry] = useState<Entry>('start')

  useEffect(() => {
    const on = () => {
      setCurrent(readHash())
      setEntry('start')
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const jump = useCallback((id: string, e: Entry = 'start') => {
    setEntry(e)
    location.hash = id
    setCurrent(id)
    window.scrollTo({ top: 0 })
  }, [])

  const go = useCallback(
    (id: string, e?: Entry) => {
      setReturnTo(null)
      jump(id, e)
    },
    [jump],
  )

  const goFrom = useCallback(
    (to: string, anchor?: string) => {
      if (to === current) return
      setReturnTo({ id: current, anchor })
      jump(to)
    },
    [current, jump],
  )

  const back = useCallback(() => {
    if (!returnTo) return
    const { id, anchor } = returnTo
    setReturnTo(null)
    jump(id)
    if (anchor) requestAnimationFrame(() => document.getElementById(anchor)?.scrollIntoView({ block: 'center' }))
  }, [returnTo, jump])

  const value = useMemo(() => ({ current, entry, go, goFrom, returnTo, back }), [current, entry, go, goFrom, returnTo, back])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useNav(): Nav {
  const v = useContext(Ctx)
  if (!v) throw new Error('NavProvider missing')
  return v
}
