import { beforeAll, describe, expect, it } from 'vitest'
import { alloc, usePrecision } from '../src/engine/alloc'
import * as K from '../src/engine/kernels'
import { expectGradsClose, numericGrad, randVec, weightedSum } from './helpers'

beforeAll(() => usePrecision('f64'))

describe('matmul', () => {
  for (const wT of [false, true]) {
    it(`gradients (wT=${wT})`, () => {
      const N = 3, C = 4, OC = 5
      const inp = randVec(N * C), W = randVec(C * OC), bias = randVec(OC), w = randVec(N * OC)
      const out = alloc(N * OC)
      const f = () => {
        K.matmulForward(out, inp, W, bias, N, C, OC, wT)
        return weightedSum(out, w)
      }
      const dinp = alloc(N * C), dW = alloc(C * OC), dbias = alloc(OC)
      f()
      K.matmulBackward(dinp, dW, dbias, w, inp, W, N, C, OC, wT)
      expectGradsClose(dinp, numericGrad(f, inp))
      expectGradsClose(dW, numericGrad(f, W))
      expectGradsClose(dbias, numericGrad(f, bias))
    })
  }
})

describe('layernorm', () => {
  it('gradients', () => {
    const N = 3, C = 5
    const inp = randVec(N * C), g = randVec(C), b = randVec(C), w = randVec(N * C)
    const out = alloc(N * C), mean = alloc(N), rstd = alloc(N)
    const f = () => {
      K.layernormForward(out, mean, rstd, inp, g, b, N, C)
      return weightedSum(out, w)
    }
    const dinp = alloc(N * C), dg = alloc(C), db = alloc(C)
    f()
    K.layernormBackward(dinp, dg, db, w, inp, g, mean, rstd, N, C)
    expectGradsClose(dinp, numericGrad(f, inp))
    expectGradsClose(dg, numericGrad(f, g))
    expectGradsClose(db, numericGrad(f, b))
  })
})

describe('attention', () => {
  it('forward masks the future and rows sum to one', () => {
    const B = 1, T = 4, C = 6, H = 2
    const qkv = randVec(B * T * 3 * C)
    const out = alloc(B * T * C), preatt = alloc(B * H * T * T), att = alloc(B * H * T * T)
    K.attentionForward(out, preatt, att, qkv, B, T, C, H)
    for (let h = 0; h < H; h++)
      for (let t = 0; t < T; t++) {
        let s = 0
        for (let t2 = 0; t2 < T; t2++) {
          const a = att[(h * T + t) * T + t2]
          if (t2 > t) expect(a).toBe(0)
          s += a
        }
        expect(s).toBeCloseTo(1, 10)
      }
  })
  it('gradients', () => {
    const B = 2, T = 4, C = 6, H = 2
    const qkv = randVec(B * T * 3 * C), w = randVec(B * T * C)
    const out = alloc(B * T * C), preatt = alloc(B * H * T * T), att = alloc(B * H * T * T)
    const f = () => {
      K.attentionForward(out, preatt, att, qkv, B, T, C, H)
      return weightedSum(out, w)
    }
    f()
    const dqkv = alloc(qkv.length), dpreatt = alloc(preatt.length), datt = alloc(att.length)
    K.attentionBackward(dqkv, dpreatt, datt, w, qkv, att, B, T, C, H)
    expectGradsClose(dqkv, numericGrad(f, qkv))
  })
})

describe('gelu', () => {
  it('gradients', () => {
    const N = 9
    const inp = randVec(N, 2), w = randVec(N)
    const out = alloc(N)
    const f = () => {
      K.geluForward(out, inp, N)
      return weightedSum(out, w)
    }
    f()
    const dinp = alloc(N)
    K.geluBackward(dinp, inp, w, N)
    expectGradsClose(dinp, numericGrad(f, inp))
  })
})

describe('softmax + cross entropy', () => {
  it('gradients', () => {
    const N = 3, V = 5
    const logits = randVec(N * V, 2), w = randVec(N)
    const targets = [1, 4, 0]
    const probs = alloc(N * V), losses = alloc(N)
    const f = () => {
      K.softmaxForward(probs, logits, N, V)
      K.crossentropyForward(losses, probs, targets, N, V)
      return weightedSum(losses, w)
    }
    f()
    const dlogits = alloc(N * V)
    K.crossentropySoftmaxBackward(dlogits, w, probs, targets, N, V)
    expectGradsClose(dlogits, numericGrad(f, logits))
  })
})

describe('encoder', () => {
  it('gradients', () => {
    const B = 2, T = 3, C = 4, V = 5
    const wte = randVec(V * C), wpe = randVec(T * C), w = randVec(B * T * C)
    const tokens = [0, 3, 3, 1, 4, 0]
    const out = alloc(B * T * C)
    const f = () => {
      K.encoderForward(out, tokens, wte, wpe, B, T, C)
      return weightedSum(out, w)
    }
    f()
    const dwte = alloc(V * C), dwpe = alloc(T * C)
    K.encoderBackward(dwte, dwpe, w, tokens, B, T, C)
    expectGradsClose(dwte, numericGrad(f, wte))
    expectGradsClose(dwpe, numericGrad(f, wpe))
  })
})
