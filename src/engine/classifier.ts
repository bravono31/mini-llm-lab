import { CLASS_IDS, type ClassId, type MlData } from '../data/ml-labels'
import { softmax } from './generate'
import { countVector } from './rag'

/** The machine-learning part: hand-made word features (bag of words) + multinomial logistic regression trained by gradient descent. */

export type WordKind = 'feature' | 'stop' | 'unknown'

export interface Word {
  text: string
  kind: WordKind
}

/**
 * ja: split with the hand-made dictionary. Dynamic programming picks the split with the fewest unknown characters,
 * then the fewest words, so 「ねこはなく」 becomes ねこ|は|なく rather than ねこ|はな|く.
 * en: split on spaces.
 */
export function segment(text: string, data: MlData): Word[] {
  const stop = new Set(data.stop)
  if (data.lang === 'en') {
    const known = new Set(featureVocab(data))
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => ({ text: w, kind: stop.has(w) ? 'stop' : known.has(w) ? 'feature' : 'unknown' }))
  }
  const words = [...data.dict, ...data.stop]
  const n = text.length
  // best[i] = [unknown chars, word count] for text[0, i)
  const best: [number, number][] = [[0, 0]]
  const from: { start: number; known: boolean }[] = []
  for (let i = 1; i <= n; i++) {
    let b: [number, number] = [best[i - 1][0] + 1, best[i - 1][1] + 1]
    let f = { start: i - 1, known: false }
    for (const w of words) {
      const s = i - w.length
      if (s < 0 || text.slice(s, i) !== w) continue
      const c: [number, number] = [best[s][0], best[s][1] + 1]
      if (c[0] < b[0] || (c[0] === b[0] && c[1] < b[1])) {
        b = c
        f = { start: s, known: true }
      }
    }
    best[i] = b
    from[i] = f
  }
  const out: Word[] = []
  for (let i = n; i > 0; ) {
    const { start, known } = from[i]
    const w = text.slice(start, i)
    const kind: WordKind = !known ? 'unknown' : stop.has(w) ? 'stop' : 'feature'
    // glue runs of unknown characters back together
    if (kind === 'unknown' && out[0]?.kind === 'unknown') out[0].text = w + out[0].text
    else out.unshift({ text: w, kind })
    i = start
  }
  return out
}

/** Feature words in order of first appearance in the data (ja: dictionary order). */
export function featureVocab(data: MlData): string[] {
  if (data.lang === 'ja') return [...data.dict]
  const stop = new Set(data.stop)
  const seen = new Set<string>()
  for (const s of data.sentences) for (const w of s.split(' ')) if (!stop.has(w)) seen.add(w)
  return [...seen]
}

export interface Dataset {
  data: MlData
  vocab: string[]
  classes: ClassId[]
  texts: string[]
  /** class index per example */
  y: number[]
  /** bag-of-words count vector per example */
  X: Float64Array[]
  train: number[]
  test: number[]
}

export function featurize(text: string, data: MlData, vocab: string[]): Float64Array {
  const index = new Map(vocab.map((w, i) => [w, i]))
  const ids = segment(text, data).flatMap((w) => (w.kind === 'feature' ? [index.get(w.text)!] : []))
  return countVector(ids, vocab.length)
}

/** Every 4th sentence is held out as test data. */
export function buildDataset(data: MlData): Dataset {
  const vocab = featureVocab(data)
  const texts = data.sentences
  const y = texts.map((t) => CLASS_IDS.indexOf(data.labels[t]))
  const X = texts.map((t) => featurize(t, data, vocab))
  const idx = texts.map((_, i) => i)
  return { data, vocab, classes: CLASS_IDS, texts, y, X, train: idx.filter((i) => i % 4 !== 3), test: idx.filter((i) => i % 4 === 3) }
}

export class SoftmaxRegression {
  /** C × F, row c = weights of class c */
  W: Float64Array
  b: Float64Array

  constructor(
    readonly C: number,
    readonly F: number,
  ) {
    this.W = new Float64Array(C * F)
    this.b = new Float64Array(C)
  }

  get paramCount(): number {
    return this.W.length + this.b.length
  }

  logits(x: ArrayLike<number>): Float64Array {
    const z = new Float64Array(this.C)
    for (let c = 0; c < this.C; c++) {
      let s = this.b[c]
      for (let f = 0; f < this.F; f++) s += this.W[c * this.F + f] * x[f]
      z[c] = s
    }
    return z
  }

  probs(x: ArrayLike<number>): Float64Array {
    return softmax(this.logits(x))
  }

  predict(x: ArrayLike<number>): number {
    const p = this.probs(x)
    return p.indexOf(Math.max(...p))
  }

  /** mean cross-entropy over the given examples */
  loss(X: Float64Array[], y: number[], idx: number[]): number {
    let s = 0
    for (const i of idx) s -= Math.log(this.probs(X[i])[y[i]])
    return idx.length ? s / idx.length : 0
  }

  /** gradient of the mean loss: dW = mean (p − onehot(y)) xᵀ, db = mean (p − onehot(y)) */
  gradient(X: Float64Array[], y: number[], idx: number[]): { dW: Float64Array; db: Float64Array } {
    const dW = new Float64Array(this.W.length)
    const db = new Float64Array(this.C)
    for (const i of idx) {
      const p = this.probs(X[i])
      for (let c = 0; c < this.C; c++) {
        const e = (p[c] - (c === y[i] ? 1 : 0)) / idx.length
        db[c] += e
        for (let f = 0; f < this.F; f++) dW[c * this.F + f] += e * X[i][f]
      }
    }
    return { dW, db }
  }

  /** one full-batch gradient-descent step; returns the loss before the update */
  step(X: Float64Array[], y: number[], idx: number[], lr: number): number {
    const before = this.loss(X, y, idx)
    const { dW, db } = this.gradient(X, y, idx)
    for (let i = 0; i < this.W.length; i++) this.W[i] -= lr * dW[i]
    for (let c = 0; c < this.C; c++) this.b[c] -= lr * db[c]
    return before
  }

  clone(): SoftmaxRegression {
    const m = new SoftmaxRegression(this.C, this.F)
    m.W.set(this.W)
    m.b.set(this.b)
    return m
  }
}

export function accuracy(model: SoftmaxRegression, ds: Dataset, idx: number[]): number {
  let ok = 0
  for (const i of idx) if (model.predict(ds.X[i]) === ds.y[i]) ok++
  return idx.length ? ok / idx.length : 0
}

/** C × C counts, row = true class, column = predicted class */
export function confusion(model: SoftmaxRegression, ds: Dataset, idx: number[]): Float64Array {
  const C = ds.classes.length
  const m = new Float64Array(C * C)
  for (const i of idx) m[ds.y[i] * C + model.predict(ds.X[i])]++
  return m
}
