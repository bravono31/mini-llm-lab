import type { Vec } from './alloc'

export interface Pca2D {
  points: [number, number][]
  explained: [number, number]
}

/** 2-D PCA of n row vectors of dimension d (power iteration with deflation). */
export function pca2d(rows: Vec, n: number, d: number): Pca2D {
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += rows[i * d + j] / n
  const X = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = rows[i * d + j] - mean[j]
  const cov = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a * d + b] += (X[i * d + a] * X[i * d + b]) / n
  let total = 0
  for (let a = 0; a < d; a++) total += cov[a * d + a]

  const comps: Float64Array[] = []
  const lambdas: number[] = []
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(d).map((_, i) => 1 / Math.sqrt(d) + i * 1e-3)
    let lambda = 0
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(d)
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) w[a] += cov[a * d + b] * v[b]
      lambda = Math.sqrt(w.reduce((s, x) => s + x * x, 0))
      if (lambda === 0) break
      v = w.map((x) => x / lambda)
    }
    comps.push(v)
    lambdas.push(lambda)
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) cov[a * d + b] -= lambda * v[a] * v[b]
  }
  const points: [number, number][] = []
  for (let i = 0; i < n; i++) {
    let x = 0
    let y = 0
    for (let j = 0; j < d; j++) {
      x += X[i * d + j] * comps[0][j]
      y += X[i * d + j] * comps[1][j]
    }
    points.push([x, y])
  }
  const safe = total || 1
  return { points, explained: [lambdas[0] / safe, lambdas[1] / safe] }
}
