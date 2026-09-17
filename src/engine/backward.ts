import { alloc, type Vec } from './alloc'
import type { Activations, NamedBuffer } from './forward'
import {
  attentionBackward,
  crossentropySoftmaxBackward,
  encoderBackward,
  geluBackward,
  layernormBackward,
  matmulBackward,
  residualBackward,
} from './kernels'
import { view, viewIn, type Params } from './params'

export interface LayerActGrads {
  dln1: Vec
  dqkv: Vec
  dpreatt: Vec
  datt: Vec
  dattnOut: Vec
  dattnProj: Vec
  dx1: Vec
  dln2: Vec
  dfc: Vec
  dgelu: Vec
  dmlpProj: Vec
  dx2: Vec
}

export interface ActGrads {
  dencoded: Vec
  layers: LayerActGrads[]
  dlnf: Vec
  dlogits: Vec
  dlosses: Vec
}

export interface BackwardResult {
  /** gradient of mean loss w.r.t. every parameter, same layout as params.data */
  grads: Vec
  actGrads: ActGrads
}

export function backward(params: Params, acts: Activations): BackwardResult {
  if (!acts.targets) throw new Error('backward requires targets')
  const { vocabSize: V, dModel: C, nHeads: H, nLayers: L, dFF: F } = params.config
  const { B, T } = acts
  const N = B * T
  const grads = alloc(params.data.length)
  const g = (name: string) => viewIn(grads, params.byName.get(name)!)

  const dlosses = alloc(N)
  dlosses.fill(1 / N)
  const dlogits = alloc(N * V)
  crossentropySoftmaxBackward(dlogits, dlosses, acts.probs, acts.targets, N, V)

  const dlnf = alloc(N * C)
  matmulBackward(dlnf, g('wte'), null, dlogits, acts.lnf, view(params, 'wte'), N, C, V, true)

  const lastX = acts.layers[L - 1].x2
  const dxLast = alloc(N * C)
  layernormBackward(dxLast, g('lnf.g'), g('lnf.b'), dlnf, lastX, view(params, 'lnf.g'), acts.lnfMean, acts.lnfRstd, N, C)

  const layers: LayerActGrads[] = new Array(L)
  let dx = dxLast // gradient flowing into x2 of the current layer
  for (let l = L - 1; l >= 0; l--) {
    const a = acts.layers[l]
    const p = `layer${l}.`
    const d: LayerActGrads = {
      dln1: alloc(N * C),
      dqkv: alloc(N * 3 * C),
      dpreatt: alloc(B * H * T * T),
      datt: alloc(B * H * T * T),
      dattnOut: alloc(N * C),
      dattnProj: alloc(N * C),
      dx1: alloc(N * C),
      dln2: alloc(N * C),
      dfc: alloc(N * F),
      dgelu: alloc(N * F),
      dmlpProj: alloc(N * C),
      dx2: dx,
    }
    residualBackward(d.dx1, d.dmlpProj, d.dx2, N * C)
    matmulBackward(d.dgelu, g(p + 'mlp.proj.w'), g(p + 'mlp.proj.b'), d.dmlpProj, a.gelu, view(params, p + 'mlp.proj.w'), N, F, C)
    geluBackward(d.dfc, a.fc, d.dgelu, N * F)
    matmulBackward(d.dln2, g(p + 'mlp.fc.w'), g(p + 'mlp.fc.b'), d.dfc, a.ln2, view(params, p + 'mlp.fc.w'), N, C, F)
    layernormBackward(d.dx1, g(p + 'ln2.g'), g(p + 'ln2.b'), d.dln2, a.x1, view(params, p + 'ln2.g'), a.ln2Mean, a.ln2Rstd, N, C)
    const dxIn = alloc(N * C)
    residualBackward(dxIn, d.dattnProj, d.dx1, N * C)
    matmulBackward(d.dattnOut, g(p + 'attn.proj.w'), g(p + 'attn.proj.b'), d.dattnProj, a.attnOut, view(params, p + 'attn.proj.w'), N, C, C)
    attentionBackward(d.dqkv, d.dpreatt, d.datt, d.dattnOut, a.qkv, a.att, B, T, C, H)
    matmulBackward(d.dln1, g(p + 'attn.qkv.w'), g(p + 'attn.qkv.b'), d.dqkv, a.ln1, view(params, p + 'attn.qkv.w'), N, C, 3 * C)
    layernormBackward(dxIn, g(p + 'ln1.g'), g(p + 'ln1.b'), d.dln1, a.xIn, view(params, p + 'ln1.g'), a.ln1Mean, a.ln1Rstd, N, C)
    layers[l] = d
    dx = dxIn
  }
  const dencoded = dx
  encoderBackward(g('wte'), g('wpe'), dencoded, acts.tokens, B, T, C)

  return { grads, actGrads: { dencoded, layers, dlnf, dlogits, dlosses } }
}

/** Activation gradients in forward order (mirror of listActivations). */
export function listActGrads(acts: Activations, ag: ActGrads): NamedBuffer[] {
  const { B, T } = acts
  const C = acts.encoded.length / (B * T)
  const out: NamedBuffer[] = [{ key: 'encoded', label: '∂L/∂x₀', data: ag.dencoded, shape: [B, T, C], layer: null }]
  ag.layers.forEach((d, l) => {
    const F = d.dfc.length / (B * T)
    const H = d.datt.length / (B * T * T)
    const add = (key: string, label: string, data: Vec, shape: number[]) => out.push({ key: `layer${l}.${key}`, label, data, shape, layer: l })
    add('ln1', '∂L/∂LN1', d.dln1, [B, T, C])
    add('qkv', '∂L/∂QKV', d.dqkv, [B, T, 3 * C])
    add('preatt', '∂L/∂スコア', d.dpreatt, [B, H, T, T])
    add('att', '∂L/∂注意重み', d.datt, [B, H, T, T])
    add('attnOut', '∂L/∂ヘッド出力', d.dattnOut, [B, T, C])
    add('attnProj', '∂L/∂W_o射影', d.dattnProj, [B, T, C])
    add('x1', '∂L/∂残差1', d.dx1, [B, T, C])
    add('ln2', '∂L/∂LN2', d.dln2, [B, T, C])
    add('fc', '∂L/∂MLP拡大', d.dfc, [B, T, F])
    add('gelu', '∂L/∂GELU', d.dgelu, [B, T, F])
    add('mlpProj', '∂L/∂MLP縮小', d.dmlpProj, [B, T, C])
    add('x2', '∂L/∂残差2', d.dx2, [B, T, C])
  })
  const V = acts.logits.length / (B * T)
  out.push({ key: 'lnf', label: '∂L/∂最終LN', data: ag.dlnf, shape: [B, T, C], layer: null })
  out.push({ key: 'logits', label: '∂L/∂ロジット', data: ag.dlogits, shape: [B, T, V], layer: null })
  return out
}
