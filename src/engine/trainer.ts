import { alloc, type Vec } from './alloc'
import { backward, type ActGrads } from './backward'
import { forward, type Activations } from './forward'
import { AdamW, type AdamWOptions, type StepStats } from './optimizer'
import type { Params } from './params'
import { mulberry32, type Rng } from './rng'

export interface Batch {
  inputs: Int32Array
  targets: Int32Array
  starts: number[]
}

export interface TrainStepResult extends StepStats {
  step: number
  loss: number
  batch: Batch
  acts: Activations
  grads: Vec
  actGrads: ActGrads
  /** parameter values before the optimizer update */
  before: Vec
}

export interface TrainerOptions extends Partial<AdamWOptions> {
  batchSize?: number
  seed?: number
}

export class Trainer {
  readonly params: Params
  readonly stream: Int32Array
  readonly opt: AdamW
  readonly batchSize: number
  readonly T: number
  readonly lossHistory: number[] = []
  rng: Rng
  step = 0

  constructor(params: Params, stream: Int32Array, opts: TrainerOptions = {}) {
    const { batchSize = 16, seed = 1, ...adam } = opts
    this.params = params
    this.stream = stream
    this.batchSize = batchSize
    this.T = params.config.ctxLen
    this.opt = new AdamW(params, adam)
    this.rng = mulberry32(seed)
  }

  get lr(): number {
    return this.opt.opts.lr
  }

  setLr(lr: number): void {
    this.opt.opts.lr = lr
  }

  sampleBatch(batchSize = this.batchSize): Batch {
    const { T, stream } = this
    const inputs = new Int32Array(batchSize * T)
    const targets = new Int32Array(batchSize * T)
    const starts: number[] = []
    const maxStart = stream.length - T - 1
    for (let b = 0; b < batchSize; b++) {
      const s = Math.floor(this.rng() * (maxStart + 1))
      starts.push(s)
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = stream[s + t]
        targets[b * T + t] = stream[s + t + 1]
      }
    }
    return { inputs, targets, starts }
  }

  trainStep(batch: Batch = this.sampleBatch()): TrainStepResult {
    const B = batch.inputs.length / this.T
    const acts = forward(this.params, batch.inputs, B, this.T, batch.targets)
    const { grads, actGrads } = backward(this.params, acts)
    const before = alloc(this.params.data.length)
    before.set(this.params.data)
    const stats = this.opt.step(this.params, grads)
    this.step += 1
    const loss = acts.meanLoss!
    this.lossHistory.push(loss)
    return { step: this.step, loss, batch, acts, grads, actGrads, before, ...stats }
  }
}
