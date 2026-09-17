import { expect } from 'vitest'
import { alloc, type Vec } from '../src/engine/alloc'
import { mulberry32, randn, type Rng } from '../src/engine/rng'

export const rng: Rng = mulberry32(42)

export function randVec(n: number, scale = 1): Vec {
  const v = alloc(n)
  for (let i = 0; i < n; i++) v[i] = randn(rng) * scale
  return v
}

/** Central finite differences of f() with respect to every entry of x. */
export function numericGrad(f: () => number, x: Vec, eps = 1e-6): Float64Array {
  const g = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) {
    const o = x[i]
    x[i] = o + eps
    const fp = f()
    x[i] = o - eps
    const fm = f()
    x[i] = o
    g[i] = (fp - fm) / (2 * eps)
  }
  return g
}

export function maxRelError(analytic: ArrayLike<number>, numeric: ArrayLike<number>): number {
  let worst = 0
  for (let i = 0; i < analytic.length; i++) {
    const err = Math.abs(analytic[i] - numeric[i]) / (Math.abs(numeric[i]) + 1)
    if (err > worst) worst = err
  }
  return worst
}

export function expectGradsClose(analytic: ArrayLike<number>, numeric: ArrayLike<number>, tol = 1e-6): void {
  expect(analytic.length).toBe(numeric.length)
  expect(maxRelError(analytic, numeric)).toBeLessThan(tol)
}

export function weightedSum(out: ArrayLike<number>, w: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < out.length; i++) s += out[i] * w[i]
  return s
}
