/** Retrieval for the RAG chapter: TF-IDF bag-of-tokens vectors ranked by cosine similarity. */

/** Smoothed IDF (same form as scikit-learn): tokens found in many documents get a small weight. */
export function idfWeights(docs: number[][], V: number): Float64Array {
  const df = new Float64Array(V)
  for (const d of docs) for (const id of new Set(d)) df[id]++
  const idf = new Float64Array(V)
  for (let i = 0; i < V; i++) idf[i] = Math.log((docs.length + 1) / (df[i] + 1)) + 1
  return idf
}

export function countVector(ids: number[], V: number): Float64Array {
  const v = new Float64Array(V)
  for (const id of ids) v[id]++
  return v
}

export function tfidfVector(ids: number[], idf: Float64Array): Float64Array {
  const v = countVector(ids, idf.length)
  for (let i = 0; i < v.length; i++) v[i] *= idf[i]
  return v
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0
}

export interface Retrieval {
  idf: Float64Array
  docVecs: Float64Array[]
  queryVec: Float64Array
  scores: number[]
  /** document indices, most similar first */
  order: number[]
}

export function retrieve(queryIds: number[], docIds: number[][], V: number): Retrieval {
  const idf = idfWeights(docIds, V)
  const docVecs = docIds.map((d) => tfidfVector(d, idf))
  const queryVec = tfidfVector(queryIds, idf)
  const scores = docVecs.map((d) => cosine(queryVec, d))
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b)
  return { idf, docVecs, queryVec, scores, order }
}
