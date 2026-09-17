import { alloc, type Vec } from './alloc'
import type { Params } from './params'

export interface AdamWOptions {
  lr: number
  beta1: number
  beta2: number
  eps: number
  weightDecay: number
}

export const DEFAULT_ADAMW: AdamWOptions = { lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 }

export interface StepStats {
  gradNorm: number
  updateNorm: number
}

/** AdamW. Weight decay applies only to 2-D tensors (weights / embeddings), as in GPT-2 / nanoGPT. */
export class AdamW {
  readonly m: Vec
  readonly v: Vec
  readonly decay: Uint8Array
  opts: AdamWOptions
  t = 0

  constructor(params: Params, opts: Partial<AdamWOptions> = {}) {
    this.opts = { ...DEFAULT_ADAMW, ...opts }
    this.m = alloc(params.data.length)
    this.v = alloc(params.data.length)
    this.decay = new Uint8Array(params.data.length)
    for (const s of params.specs) if (s.shape.length === 2) this.decay.fill(1, s.offset, s.offset + s.size)
  }

  step(params: Params, grads: Vec): StepStats {
    const { lr, beta1, beta2, eps, weightDecay } = this.opts
    this.t += 1
    const c1 = 1 - Math.pow(beta1, this.t)
    const c2 = 1 - Math.pow(beta2, this.t)
    const p = params.data
    let gradNorm = 0
    let updateNorm = 0
    for (let i = 0; i < p.length; i++) {
      const g = grads[i]
      gradNorm += g * g
      const m = (this.m[i] = beta1 * this.m[i] + (1 - beta1) * g)
      const v = (this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g)
      const mhat = m / c1
      const vhat = v / c2
      const upd = lr * (mhat / (Math.sqrt(vhat) + eps) + (this.decay[i] ? weightDecay * p[i] : 0))
      p[i] -= upd
      updateNorm += upd * upd
    }
    return { gradNorm: Math.sqrt(gradNorm), updateNorm: Math.sqrt(updateNorm) }
  }

  reset(): void {
    this.m.fill(0)
    this.v.fill(0)
    this.t = 0
  }
}
