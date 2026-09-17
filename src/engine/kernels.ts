// Explicit forward / backward kernels (llm.c style). No autograd: every
// intermediate lives in a named buffer so the UI can show it.
// Layouts: activations [B,T,C] index (b*T+t)*C+c, attention [B,H,T,T].
// All backward kernels ACCUMULATE (+=) into their d-buffers.
import type { Vec } from './alloc'

const GELU_SCALE = Math.sqrt(2 / Math.PI)

export function encoderForward(out: Vec, inp: ArrayLike<number>, wte: Vec, wpe: Vec, B: number, T: number, C: number): void {
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const o = (b * T + t) * C
      const ix = inp[b * T + t] * C
      const p = t * C
      for (let i = 0; i < C; i++) out[o + i] = wte[ix + i] + wpe[p + i]
    }
  }
}

export function encoderBackward(dwte: Vec, dwpe: Vec, dout: Vec, inp: ArrayLike<number>, B: number, T: number, C: number): void {
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const o = (b * T + t) * C
      const ix = inp[b * T + t] * C
      const p = t * C
      for (let i = 0; i < C; i++) {
        const d = dout[o + i]
        dwte[ix + i] += d
        dwpe[p + i] += d
      }
    }
  }
}

export const LN_EPS = 1e-5

export function layernormForward(out: Vec, mean: Vec, rstd: Vec, inp: Vec, g: Vec, b: Vec, N: number, C: number): void {
  for (let n = 0; n < N; n++) {
    const x = n * C
    let m = 0
    for (let i = 0; i < C; i++) m += inp[x + i]
    m /= C
    let v = 0
    for (let i = 0; i < C; i++) {
      const d = inp[x + i] - m
      v += d * d
    }
    v /= C
    const s = 1 / Math.sqrt(v + LN_EPS)
    for (let i = 0; i < C; i++) out[x + i] = s * (inp[x + i] - m) * g[i] + b[i]
    mean[n] = m
    rstd[n] = s
  }
}

export function layernormBackward(dinp: Vec, dg: Vec, db: Vec, dout: Vec, inp: Vec, g: Vec, mean: Vec, rstd: Vec, N: number, C: number): void {
  for (let n = 0; n < N; n++) {
    const x = n * C
    const m = mean[n]
    const s = rstd[n]
    let dnormMean = 0
    let dnormNormMean = 0
    for (let i = 0; i < C; i++) {
      const norm = (inp[x + i] - m) * s
      const dnorm = g[i] * dout[x + i]
      dnormMean += dnorm
      dnormNormMean += dnorm * norm
    }
    dnormMean /= C
    dnormNormMean /= C
    for (let i = 0; i < C; i++) {
      const norm = (inp[x + i] - m) * s
      const dnorm = g[i] * dout[x + i]
      db[i] += dout[x + i]
      dg[i] += norm * dout[x + i]
      dinp[x + i] += (dnorm - dnormMean - norm * dnormNormMean) * s
    }
  }
}

/**
 * out[N,OC] = inp[N,C] · W + bias.  W is [C,OC], or [OC,C] when wT is true
 * (used for the tied output layer, where W is the embedding table).
 */
export function matmulForward(out: Vec, inp: Vec, W: Vec, bias: Vec | null, N: number, C: number, OC: number, wT = false): void {
  for (let n = 0; n < N; n++) {
    const x = n * C
    const o = n * OC
    for (let j = 0; j < OC; j++) {
      let acc = bias ? bias[j] : 0
      if (wT) {
        const w = j * C
        for (let i = 0; i < C; i++) acc += inp[x + i] * W[w + i]
      } else {
        for (let i = 0; i < C; i++) acc += inp[x + i] * W[i * OC + j]
      }
      out[o + j] = acc
    }
  }
}

export function matmulBackward(dinp: Vec, dW: Vec, dbias: Vec | null, dout: Vec, inp: Vec, W: Vec, N: number, C: number, OC: number, wT = false): void {
  for (let n = 0; n < N; n++) {
    const x = n * C
    const o = n * OC
    for (let j = 0; j < OC; j++) {
      const d = dout[o + j]
      if (dbias) dbias[j] += d
      if (wT) {
        const w = j * C
        for (let i = 0; i < C; i++) {
          dinp[x + i] += W[w + i] * d
          dW[w + i] += inp[x + i] * d
        }
      } else {
        for (let i = 0; i < C; i++) {
          dinp[x + i] += W[i * OC + j] * d
          dW[i * OC + j] += inp[x + i] * d
        }
      }
    }
  }
}

/**
 * Causal multi-head attention.
 * qkv [B,T,3C] (q | k | v, each C wide, head h occupies [h*hs, (h+1)*hs)).
 * preatt [B,H,T,T]: scaled scores q·k/sqrt(hs), computed for ALL key
 * positions (future ones too) so the UI can show what the mask removes.
 * att [B,H,T,T]: softmax over keys t2 <= t; zero for t2 > t.
 * out [B,T,C]: concatenated head outputs.
 */
export function attentionForward(out: Vec, preatt: Vec, att: Vec, qkv: Vec, B: number, T: number, C: number, H: number): void {
  const C3 = 3 * C
  const hs = C / H
  const scale = 1 / Math.sqrt(hs)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const q = (b * T + t) * C3 + h * hs
        const row = ((b * H + h) * T + t) * T
        let maxval = -Infinity
        for (let t2 = 0; t2 < T; t2++) {
          const k = (b * T + t2) * C3 + h * hs + C
          let dot = 0
          for (let i = 0; i < hs; i++) dot += qkv[q + i] * qkv[k + i]
          const val = dot * scale
          preatt[row + t2] = val
          if (t2 <= t && val > maxval) maxval = val
        }
        let expsum = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const e = Math.exp(preatt[row + t2] - maxval)
          att[row + t2] = e
          expsum += e
        }
        const inv = expsum === 0 ? 0 : 1 / expsum
        for (let t2 = 0; t2 < T; t2++) att[row + t2] = t2 <= t ? att[row + t2] * inv : 0
        const o = (b * T + t) * C + h * hs
        for (let i = 0; i < hs; i++) out[o + i] = 0
        for (let t2 = 0; t2 <= t; t2++) {
          const v = (b * T + t2) * C3 + h * hs + 2 * C
          const a = att[row + t2]
          for (let i = 0; i < hs; i++) out[o + i] += a * qkv[v + i]
        }
      }
    }
  }
}

export function attentionBackward(dqkv: Vec, dpreatt: Vec, datt: Vec, dout: Vec, qkv: Vec, att: Vec, B: number, T: number, C: number, H: number): void {
  const C3 = 3 * C
  const hs = C / H
  const scale = 1 / Math.sqrt(hs)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const row = ((b * H + h) * T + t) * T
        const q = (b * T + t) * C3 + h * hs
        const o = (b * T + t) * C + h * hs
        // 1) out = att · v
        for (let t2 = 0; t2 <= t; t2++) {
          const v = (b * T + t2) * C3 + h * hs + 2 * C
          for (let i = 0; i < hs; i++) {
            datt[row + t2] += qkv[v + i] * dout[o + i]
            dqkv[v + i] += att[row + t2] * dout[o + i]
          }
        }
        // 2) att = softmax(preatt)
        for (let t2 = 0; t2 <= t; t2++) {
          for (let t3 = 0; t3 <= t; t3++) {
            const ind = t2 === t3 ? 1 : 0
            dpreatt[row + t3] += att[row + t2] * (ind - att[row + t3]) * datt[row + t2]
          }
        }
        // 3) preatt = (q · k) * scale
        for (let t2 = 0; t2 <= t; t2++) {
          const k = (b * T + t2) * C3 + h * hs + C
          const d = dpreatt[row + t2] * scale
          for (let i = 0; i < hs; i++) {
            dqkv[q + i] += qkv[k + i] * d
            dqkv[k + i] += qkv[q + i] * d
          }
        }
      }
    }
  }
}

export function gelu(x: number): number {
  const cube = 0.044715 * x * x * x
  return 0.5 * x * (1 + Math.tanh(GELU_SCALE * (x + cube)))
}

export function geluForward(out: Vec, inp: Vec, N: number): void {
  for (let i = 0; i < N; i++) out[i] = gelu(inp[i])
}

export function geluBackward(dinp: Vec, inp: Vec, dout: Vec, N: number): void {
  for (let i = 0; i < N; i++) {
    const x = inp[i]
    const cube = 0.044715 * x * x * x
    const arg = GELU_SCALE * (x + cube)
    const th = Math.tanh(arg)
    const cosh = Math.cosh(arg)
    const sech2 = 1 / (cosh * cosh)
    const local = 0.5 * (1 + th) + x * 0.5 * sech2 * GELU_SCALE * (1 + 3 * 0.044715 * x * x)
    dinp[i] += local * dout[i]
  }
}

export function residualForward(out: Vec, a: Vec, b: Vec, N: number): void {
  for (let i = 0; i < N; i++) out[i] = a[i] + b[i]
}

export function residualBackward(da: Vec, db: Vec, dout: Vec, N: number): void {
  for (let i = 0; i < N; i++) {
    da[i] += dout[i]
    db[i] += dout[i]
  }
}

export function softmaxForward(probs: Vec, logits: Vec, N: number, V: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * V
    let maxval = -Infinity
    for (let i = 0; i < V; i++) if (logits[o + i] > maxval) maxval = logits[o + i]
    let sum = 0
    for (let i = 0; i < V; i++) {
      const e = Math.exp(logits[o + i] - maxval)
      probs[o + i] = e
      sum += e
    }
    for (let i = 0; i < V; i++) probs[o + i] /= sum
  }
}

export function crossentropyForward(losses: Vec, probs: Vec, targets: ArrayLike<number>, N: number, V: number): void {
  for (let n = 0; n < N; n++) losses[n] = -Math.log(probs[n * V + targets[n]])
}

export function crossentropySoftmaxBackward(dlogits: Vec, dlosses: Vec, probs: Vec, targets: ArrayLike<number>, N: number, V: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * V
    const d = dlosses[n]
    const tgt = targets[n]
    for (let i = 0; i < V; i++) dlogits[o + i] += (probs[o + i] - (i === tgt ? 1 : 0)) * d
  }
}
