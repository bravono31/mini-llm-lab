import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type ThemeChoice = 'auto' | 'light' | 'dark'
export type Resolved = 'light' | 'dark'

interface ThemeCtx {
  theme: ThemeChoice
  setTheme: (t: ThemeChoice) => void
  resolved: Resolved
}

const Ctx = createContext<ThemeCtx | null>(null)
const KEY = 'mini-llm-lab.theme'

function readStored(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark' || v === 'auto') return v
  } catch {
    /* storage unavailable */
  }
  return 'auto'
}

function applyAttr(t: ThemeChoice): void {
  const root = document.documentElement
  if (t === 'auto') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', t)
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  // apply the stored choice before children render, so components that read
  // CSS tokens (heat-map palettes) see the right theme on the first pass
  const [theme, setThemeState] = useState<ThemeChoice>(() => {
    const t = readStored()
    applyAttr(t)
    return t
  })
  const [system, setSystem] = useState<Resolved>(() =>
    typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  )
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const on = () => setSystem(mq.matches ? 'dark' : 'light')
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  useEffect(() => applyAttr(theme), [theme])
  const setTheme = (t: ThemeChoice) => {
    setThemeState(t)
    try {
      localStorage.setItem(KEY, t)
    } catch {
      /* ignore */
    }
  }
  const resolved: Resolved = theme === 'auto' ? system : theme
  const value = useMemo(() => ({ theme, setTheme, resolved }), [theme, resolved])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useTheme(): ThemeCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('ThemeProvider missing')
  return v
}
