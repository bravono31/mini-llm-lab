import type { Vec } from './alloc'
import { forward, type Activations } from './forward'
import type { Params } from './params'
import type { Rng } from './rng'
import { EOS_ID } from './tokenizer'

export type SampleMode = 'greedy' | 'topk' | 'random'

export interface SampleOptions {
  temperature: number
  mode: SampleMode
  k: number
}

export const DEFAULT_SAMPLE: SampleOptions = { temperature: 1, mode: 'greedy', k: 5 }

export interface Distribution {
  /** logits / temperature */
  scaled: Float64Array
  /** softmax of scaled logits */
  probs: Float64Array
  /** final sampling distribution after top-k filtering (renormalised) */
  sampling: Float64Array
  kept: boolean[]
}

export function lastLogits(acts: Activations): Vec {
  const V = acts.logits.length / (acts.B * acts.T)
  return acts.logits.subarray((acts.T - 1) * V, acts.T * V)
}

export function makeDistribution(logits: ArrayLike<number>, opts: SampleOptions): Distribution {
  const V = logits.length
  const temp = Math.max(opts.temperature, 1e-3)
  const scaled = new Float64Array(V)
  for (let i = 0; i < V; i++) scaled[i] = logits[i] / temp
  const probs = softmax(scaled)
  const kept = new Array<boolean>(V).fill(true)
  if (opts.mode === 'greedy') {
    kept.fill(false)
    kept[argmax(probs)] = true
  } else if (opts.mode === 'topk') {
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a])
    kept.fill(false)
    for (let i = 0; i < Math.min(opts.k, V); i++) kept[order[i]] = true
  }
  const sampling = new Float64Array(V)
  let sum = 0
  for (let i = 0; i < V; i++) if (kept[i]) sum += probs[i]
  for (let i = 0; i < V; i++) sampling[i] = kept[i] ? probs[i] / sum : 0
  return { scaled, probs, sampling, kept }
}

export function sampleIndex(sampling: ArrayLike<number>, rng: Rng): number {
  const r = rng()
  let acc = 0
  for (let i = 0; i < sampling.length; i++) {
    acc += sampling[i]
    if (r < acc) return i
  }
  let last = sampling.length - 1
  while (last > 0 && sampling[last] === 0) last--
  return last
}

export interface GenStep {
  /** context fed to the model (last ctxLen tokens) */
  context: number[]
  acts: Activations
  dist: Distribution
  chosen: number
}

export function contextWindow(ids: number[], ctxLen: number): number[] {
  return ids.length > ctxLen ? ids.slice(ids.length - ctxLen) : ids
}

export function nextStep(params: Params, ids: number[], opts: SampleOptions, rng: Rng): GenStep {
  const context = contextWindow(ids, params.config.ctxLen)
  const acts = forward(params, context, 1, context.length)
  const dist = makeDistribution(lastLogits(acts), opts)
  const chosen = sampleIndex(dist.sampling, rng)
  return { context, acts, dist, chosen }
}

export function generate(params: Params, promptIds: number[], maxNew: number, opts: SampleOptions, rng: Rng, stopAtEos = true): GenStep[] {
  const ids = [...promptIds]
  const steps: GenStep[] = []
  for (let i = 0; i < maxNew; i++) {
    const s = nextStep(params, ids, opts, rng)
    steps.push(s)
    ids.push(s.chosen)
    if (stopAtEos && s.chosen === EOS_ID) break
  }
  return steps
}

export function softmax(x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(x.length)
  let max = -Infinity
  for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i]
  let sum = 0
  for (let i = 0; i < x.length; i++) {
    out[i] = Math.exp(x[i] - max)
    sum += out[i]
  }
  for (let i = 0; i < x.length; i++) out[i] /= sum
  return out
}

export function argmax(x: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i
  return best
}
