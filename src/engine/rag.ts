import type { RagQuestion } from '../data/rag-docs'
import { forward } from './forward'
import { generate, lastLogits, softmax } from './generate'
import { createParams, type Params } from './params'
import { mulberry32 } from './rng'
import { EOS_ID, type Tokenizer } from './tokenizer'

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

/** Snapshots of the RAG fine-tune (scripts/rag-finetune.ts), all sharing the pretrained model's config and tokenizer. */
export interface RagCheckpoints {
  lang: string
  /** step 0 has no weights: it is the pretrained model itself. copyAcc = % of fresh synthetic facts copied correctly */
  checkpoints: { step: number; copyAcc: number; weights: number[] }[]
  /** mean training loss per 50 steps */
  loss: number[]
}

export type Checkpoint = RagCheckpoints['checkpoints'][number]

/** Step 0 is the untouched pretrained model; later checkpoints carry their own weights. */
export function checkpointParams(pretrained: Params, ck: Checkpoint): Params {
  if (!ck.weights.length) return pretrained
  const p = createParams(pretrained.config)
  p.data.set(ck.weights)
  return p
}

export interface Probe {
  prompt: number[]
  top: { id: number; p: number }[]
  /** greedy continuation until ⟨eos⟩ */
  text: string
}

function probe(params: Params, tok: Tokenizer, prompt: number[]): Probe {
  const probs = softmax(lastLogits(forward(params, prompt, 1, prompt.length)))
  const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5).map((id) => ({ id, p: probs[id] }))
  const steps = generate(params, prompt, 6, { mode: 'greedy', temperature: 1, k: 5 }, mulberry32(1))
  return { prompt, top, text: tok.decode(steps.map((s) => s.chosen)).split('\n')[0] }
}

export interface Comparison {
  q: RagQuestion
  answer: string
  /** first token of the answer as it appears inside the document, if the split falls on a token boundary */
  answerId: number | null
  noDoc: Probe
  withDoc: Probe
  correct: boolean
}

export function compare(params: Params, tok: Tokenizer, q: RagQuestion, doc: string): Comparison {
  const cloze = tok.encode(q.cloze)
  const docIds = tok.encode(doc)
  const aligned = cloze.every((id, i) => docIds[i] === id)
  const answer = doc.slice(q.cloze.length).trim()
  const noDoc = probe(params, tok, [EOS_ID, ...cloze])
  const withDoc = probe(params, tok, [EOS_ID, ...docIds, EOS_ID, ...cloze])
  return { q, answer, answerId: aligned ? (docIds[cloze.length] ?? null) : null, noDoc, withDoc, correct: withDoc.text.trim().startsWith(answer) }
}
