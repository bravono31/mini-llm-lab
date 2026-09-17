export interface ModelConfig {
  vocabSize: number
  ctxLen: number
  dModel: number
  nHeads: number
  nLayers: number
  dFF: number
}

export const BASE_CONFIG: Omit<ModelConfig, 'vocabSize'> = {
  ctxLen: 16,
  dModel: 16,
  nHeads: 2,
  nLayers: 2,
  dFF: 64,
}

export function makeConfig(vocabSize: number, overrides: Partial<ModelConfig> = {}): ModelConfig {
  return { ...BASE_CONFIG, vocabSize, ...overrides }
}
