import type { ModelConfig } from './config'
import { createParams, type Params } from './params'
import { Tokenizer, type Lang, type TokenizerJSON } from './tokenizer'

export interface Pretrained {
  lang: Lang
  seed: number
  config: ModelConfig
  tokenizer: TokenizerJSON
  weights: number[]
  lossHistory: number[]
  samples: { step: number; text: string }[]
  trainSteps: number
}

export function exportWeights(params: Params, digits = 5): number[] {
  const f = Math.pow(10, digits)
  return Array.from(params.data, (x) => Math.round(x * f) / f)
}

export function loadPretrained(p: Pretrained): { params: Params; tokenizer: Tokenizer } {
  const tokenizer = new Tokenizer(p.tokenizer)
  const params = createParams(p.config)
  if (p.weights.length !== params.data.length) throw new Error(`weight size mismatch: ${p.weights.length} vs ${params.data.length}`)
  params.data.set(p.weights)
  return { params, tokenizer }
}
