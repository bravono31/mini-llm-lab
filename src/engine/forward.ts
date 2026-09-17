import { alloc, type Vec } from './alloc'
import {
  attentionForward,
  crossentropyForward,
  encoderForward,
  geluForward,
  layernormForward,
  matmulForward,
  residualForward,
  softmaxForward,
} from './kernels'
import { view, type Params } from './params'

export interface LayerActs {
  /** input to this block (residual stream) */
  xIn: Vec
  ln1: Vec
  ln1Mean: Vec
  ln1Rstd: Vec
  qkv: Vec
  preatt: Vec
  att: Vec
  attnOut: Vec
  attnProj: Vec
  x1: Vec
  ln2: Vec
  ln2Mean: Vec
  ln2Rstd: Vec
  fc: Vec
  gelu: Vec
  mlpProj: Vec
  x2: Vec
}

export interface Activations {
  B: number
  T: number
  tokens: Int32Array
  targets: Int32Array | null
  encoded: Vec
  layers: LayerActs[]
  lnf: Vec
  lnfMean: Vec
  lnfRstd: Vec
  logits: Vec
  probs: Vec
  losses: Vec | null
  meanLoss: number | null
}

export function forward(params: Params, tokens: ArrayLike<number>, B: number, T: number, targets: ArrayLike<number> | null = null): Activations {
  const { vocabSize: V, dModel: C, nHeads: H, nLayers: L, dFF: F, ctxLen } = params.config
  if (T > ctxLen) throw new Error(`T=${T} exceeds context length ${ctxLen}`)
  const N = B * T
  const toks = Int32Array.from(tokens as ArrayLike<number>)
  const encoded = alloc(N * C)
  encoderForward(encoded, toks, view(params, 'wte'), view(params, 'wpe'), B, T, C)

  const layers: LayerActs[] = []
  let x: Vec = encoded
  for (let l = 0; l < L; l++) {
    const p = `layer${l}.`
    const a: LayerActs = {
      xIn: x,
      ln1: alloc(N * C),
      ln1Mean: alloc(N),
      ln1Rstd: alloc(N),
      qkv: alloc(N * 3 * C),
      preatt: alloc(B * H * T * T),
      att: alloc(B * H * T * T),
      attnOut: alloc(N * C),
      attnProj: alloc(N * C),
      x1: alloc(N * C),
      ln2: alloc(N * C),
      ln2Mean: alloc(N),
      ln2Rstd: alloc(N),
      fc: alloc(N * F),
      gelu: alloc(N * F),
      mlpProj: alloc(N * C),
      x2: alloc(N * C),
    }
    layernormForward(a.ln1, a.ln1Mean, a.ln1Rstd, x, view(params, p + 'ln1.g'), view(params, p + 'ln1.b'), N, C)
    matmulForward(a.qkv, a.ln1, view(params, p + 'attn.qkv.w'), view(params, p + 'attn.qkv.b'), N, C, 3 * C)
    attentionForward(a.attnOut, a.preatt, a.att, a.qkv, B, T, C, H)
    matmulForward(a.attnProj, a.attnOut, view(params, p + 'attn.proj.w'), view(params, p + 'attn.proj.b'), N, C, C)
    residualForward(a.x1, x, a.attnProj, N * C)
    layernormForward(a.ln2, a.ln2Mean, a.ln2Rstd, a.x1, view(params, p + 'ln2.g'), view(params, p + 'ln2.b'), N, C)
    matmulForward(a.fc, a.ln2, view(params, p + 'mlp.fc.w'), view(params, p + 'mlp.fc.b'), N, C, F)
    geluForward(a.gelu, a.fc, N * F)
    matmulForward(a.mlpProj, a.gelu, view(params, p + 'mlp.proj.w'), view(params, p + 'mlp.proj.b'), N, F, C)
    residualForward(a.x2, a.x1, a.mlpProj, N * C)
    layers.push(a)
    x = a.x2
  }

  const lnf = alloc(N * C)
  const lnfMean = alloc(N)
  const lnfRstd = alloc(N)
  layernormForward(lnf, lnfMean, lnfRstd, x, view(params, 'lnf.g'), view(params, 'lnf.b'), N, C)
  const logits = alloc(N * V)
  matmulForward(logits, lnf, view(params, 'wte'), null, N, C, V, true)
  const probs = alloc(N * V)
  softmaxForward(probs, logits, N, V)

  let losses: Vec | null = null
  let meanLoss: number | null = null
  let tgts: Int32Array | null = null
  if (targets) {
    tgts = Int32Array.from(targets as ArrayLike<number>)
    losses = alloc(N)
    crossentropyForward(losses, probs, tgts, N, V)
    let s = 0
    for (let i = 0; i < N; i++) s += losses[i]
    meanLoss = s / N
  }

  return { B, T, tokens: toks, targets: tgts, encoded, layers, lnf, lnfMean, lnfRstd, logits, probs, losses, meanLoss }
}

export interface NamedBuffer {
  key: string
  label: string
  data: Vec
  shape: number[]
  layer: number | null
}

/** Activation buffers in forward order (used to animate the backward pass in reverse). */
export function listActivations(acts: Activations): NamedBuffer[] {
  const { B, T } = acts
  const C = acts.encoded.length / (B * T)
  const out: NamedBuffer[] = [{ key: 'encoded', label: '埋め込み x₀', data: acts.encoded, shape: [B, T, C], layer: null }]
  acts.layers.forEach((a, l) => {
    const F = a.fc.length / (B * T)
    const H = a.att.length / (B * T * T)
    const add = (key: string, label: string, data: Vec, shape: number[]) => out.push({ key: `layer${l}.${key}`, label, data, shape, layer: l })
    add('ln1', 'LayerNorm 1', a.ln1, [B, T, C])
    add('qkv', 'Q・K・V', a.qkv, [B, T, 3 * C])
    add('preatt', '注意スコア', a.preatt, [B, H, T, T])
    add('att', '注意重み', a.att, [B, H, T, T])
    add('attnOut', 'ヘッド出力', a.attnOut, [B, T, C])
    add('attnProj', 'W_o 射影', a.attnProj, [B, T, C])
    add('x1', '残差 1', a.x1, [B, T, C])
    add('ln2', 'LayerNorm 2', a.ln2, [B, T, C])
    add('fc', 'MLP 拡大', a.fc, [B, T, F])
    add('gelu', 'GELU', a.gelu, [B, T, F])
    add('mlpProj', 'MLP 縮小', a.mlpProj, [B, T, C])
    add('x2', '残差 2', a.x2, [B, T, C])
  })
  const V = acts.logits.length / (B * T)
  out.push({ key: 'lnf', label: '最終 LayerNorm', data: acts.lnf, shape: [B, T, C], layer: null })
  out.push({ key: 'logits', label: 'ロジット', data: acts.logits, shape: [B, T, V], layer: null })
  out.push({ key: 'probs', label: '確率', data: acts.probs, shape: [B, T, V], layer: null })
  return out
}
