import { useMemo } from 'react'
import { useTheme } from '../state/ThemeProvider'

type Rgb = [number, number, number]

function hexToRgb(hex: string): Rgb {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

function mix(a: Rgb, b: Rgb, t: number): string {
  const r = Math.round(a[0] + (b[0] - a[0]) * t)
  const g = Math.round(a[1] + (b[1] - a[1]) * t)
  const bl = Math.round(a[2] + (b[2] - a[2]) * t)
  return `rgb(${r},${g},${bl})`
}

export function readToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export interface HeatPalette {
  /** t in [0,1] */
  seq: (t: number) => string
  /** t in [-1,1] */
  div: (t: number) => string
}

/** Heat-map colour scales resolved from the current theme's CSS tokens. */
export function useHeatPalette(): HeatPalette {
  const { resolved, theme } = useTheme()
  return useMemo(() => {
    const lo = hexToRgb(readToken('--seq-lo') || '#efe9db')
    const hi = hexToRgb(readToken('--seq-hi') || '#1c4f86')
    const neg = hexToRgb(readToken('--div-neg') || '#2f6fb5')
    const mid = hexToRgb(readToken('--div-mid') || '#ddd7ca')
    const pos = hexToRgb(readToken('--div-pos') || '#c23a2a')
    const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, Number.isFinite(x) ? x : 0))
    return {
      seq: (t) => mix(lo, hi, clamp(t, 0, 1)),
      div: (t) => {
        const c = clamp(t, -1, 1)
        return c < 0 ? mix(mid, neg, -c) : mix(mid, pos, c)
      },
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved, theme])
}

export function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return v > 0 ? '∞' : v < 0 ? '−∞' : 'NaN'
  const s = v.toFixed(digits)
  return s.startsWith('-') ? '−' + s.slice(1) : s
}

export function fmtPct(p: number, digits = 1): string {
  return (p * 100).toFixed(digits) + '%'
}
