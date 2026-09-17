import { beforeAll, describe, expect, it } from 'vitest'
import { usePrecision } from '../src/engine/alloc'
import { backward } from '../src/engine/backward'
import { makeConfig } from '../src/engine/config'
import { forward } from '../src/engine/forward'
import { createParams, view } from '../src/engine/params'
import { mulberry32, randn } from '../src/engine/rng'
import { expectGradsClose, numericGrad } from './helpers'

beforeAll(() => usePrecision('f64'))

describe('full model', () => {
  it('backward matches finite differences for every parameter', () => {
    const cfg = makeConfig(8, { ctxLen: 4, dModel: 4, nHeads: 2, nLayers: 2, dFF: 8 })
    const rng = mulberry32(7)
    const params = createParams(cfg, rng)
    // make the network clearly non-linear: larger weights, random biases / norms
    for (const s of params.specs) {
      const v = view(params, s.name)
      if (s.group === 'weight' || s.group === 'embedding') for (let i = 0; i < v.length; i++) v[i] *= 25
      else if (s.group === 'bias') for (let i = 0; i < v.length; i++) v[i] = randn(rng) * 0.3
      else for (let i = 0; i < v.length; i++) v[i] += randn(rng) * 0.3
    }
    const B = 2, T = 4
    const tokens = [0, 3, 5, 7, 2, 2, 1, 6]
    const targets = [3, 5, 7, 1, 2, 1, 6, 4]
    const acts = forward(params, tokens, B, T, targets)
    expect(acts.meanLoss).toBeGreaterThan(0)
    const { grads, actGrads } = backward(params, acts)
    expect(actGrads.layers).toHaveLength(2)
    const numeric = numericGrad(() => forward(params, tokens, B, T, targets).meanLoss!, params.data)
    expectGradsClose(grads, numeric, 1e-6)
    // every parameter tensor receives a non-trivial gradient
    for (const s of params.specs) {
      let norm = 0
      for (let i = s.offset; i < s.offset + s.size; i++) norm += grads[i] * grads[i]
      expect(norm, s.name).toBeGreaterThan(0)
    }
  })

  it('forward without targets returns probabilities that sum to one', () => {
    const cfg = makeConfig(8, { ctxLen: 4, dModel: 4, nHeads: 2, nLayers: 1, dFF: 8 })
    const params = createParams(cfg, mulberry32(1))
    const acts = forward(params, [1, 2, 3], 1, 3)
    expect(acts.losses).toBeNull()
    for (let t = 0; t < 3; t++) {
      let s = 0
      for (let v = 0; v < 8; v++) s += acts.probs[t * 8 + v]
      expect(s).toBeCloseTo(1, 10)
    }
  })
})
