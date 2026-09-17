import { alloc, type Vec } from './alloc'
import type { ModelConfig } from './config'
import { randn, type Rng } from './rng'

export type ParamGroup = 'embedding' | 'weight' | 'bias' | 'norm'

export interface ParamSpec {
  name: string
  shape: number[]
  offset: number
  size: number
  layer: number | null
  group: ParamGroup
  /** Human-readable role, used by the UI */
  label: string
}

export interface Params {
  config: ModelConfig
  specs: ParamSpec[]
  data: Vec
  byName: Map<string, ParamSpec>
}

export function createParamLayout(cfg: ModelConfig): ParamSpec[] {
  const { vocabSize: V, ctxLen: T, dModel: D, dFF: F, nLayers: L } = cfg
  const specs: ParamSpec[] = []
  let offset = 0
  const add = (name: string, shape: number[], layer: number | null, group: ParamGroup, label: string) => {
    const size = shape.reduce((a, b) => a * b, 1)
    specs.push({ name, shape, offset, size, layer, group, label })
    offset += size
  }
  add('wte', [V, D], null, 'embedding', 'トークン埋め込み（出力層と共有）')
  add('wpe', [T, D], null, 'embedding', '位置埋め込み')
  for (let l = 0; l < L; l++) {
    const p = `layer${l}.`
    add(p + 'ln1.g', [D], l, 'norm', 'LayerNorm 1 ゲイン γ')
    add(p + 'ln1.b', [D], l, 'norm', 'LayerNorm 1 バイアス β')
    add(p + 'attn.qkv.w', [D, 3 * D], l, 'weight', 'Q・K・V 射影 W_qkv')
    add(p + 'attn.qkv.b', [3 * D], l, 'bias', 'Q・K・V バイアス')
    add(p + 'attn.proj.w', [D, D], l, 'weight', 'ヘッド出力を混ぜる行列 W_o（出力射影）')
    add(p + 'attn.proj.b', [D], l, 'bias', '注意出力バイアス')
    add(p + 'ln2.g', [D], l, 'norm', 'LayerNorm 2 ゲイン γ')
    add(p + 'ln2.b', [D], l, 'norm', 'LayerNorm 2 バイアス β')
    add(p + 'mlp.fc.w', [D, F], l, 'weight', 'MLP 拡大 W₁')
    add(p + 'mlp.fc.b', [F], l, 'bias', 'MLP 拡大バイアス b₁')
    add(p + 'mlp.proj.w', [F, D], l, 'weight', 'MLP 縮小 W₂')
    add(p + 'mlp.proj.b', [D], l, 'bias', 'MLP 縮小バイアス b₂')
  }
  add('lnf.g', [D], null, 'norm', '最終 LayerNorm ゲイン γ')
  add('lnf.b', [D], null, 'norm', '最終 LayerNorm バイアス β')
  return specs
}

export function totalSize(specs: ParamSpec[]): number {
  return specs.reduce((a, s) => a + s.size, 0)
}

export function createParams(cfg: ModelConfig, rng?: Rng): Params {
  const specs = createParamLayout(cfg)
  const data = alloc(totalSize(specs))
  const byName = new Map(specs.map((s) => [s.name, s]))
  const params: Params = { config: cfg, specs, data, byName }
  if (rng) initParams(params, rng)
  return params
}

/** GPT-2 style init: N(0, 0.02) for weights, residual projections scaled by 1/sqrt(2L). */
export function initParams(params: Params, rng: Rng): void {
  const scaleResid = 1 / Math.sqrt(2 * params.config.nLayers)
  for (const s of params.specs) {
    const v = view(params, s.name)
    if (s.group === 'norm') {
      v.fill(s.name.endsWith('.g') ? 1 : 0)
    } else if (s.group === 'bias') {
      v.fill(0)
    } else {
      const std = s.name.endsWith('proj.w') ? 0.02 * scaleResid : 0.02
      for (let i = 0; i < v.length; i++) v[i] = randn(rng) * std
    }
  }
}

export function view(params: Params, name: string): Vec {
  const s = params.byName.get(name)
  if (!s) throw new Error(`unknown param ${name}`)
  return params.data.subarray(s.offset, s.offset + s.size)
}

export function viewIn(data: Vec, spec: ParamSpec): Vec {
  return data.subarray(spec.offset, spec.offset + spec.size)
}

export function cloneParams(params: Params): Params {
  const data = alloc(params.data.length)
  data.set(params.data)
  return { ...params, data }
}
