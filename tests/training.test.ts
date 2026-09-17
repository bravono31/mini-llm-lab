import { beforeAll, describe, expect, it } from 'vitest'
import { CORPORA } from '../src/data/corpus'
import { usePrecision } from '../src/engine/alloc'
import { makeConfig } from '../src/engine/config'
import { createParams } from '../src/engine/params'
import { mulberry32 } from '../src/engine/rng'
import { Tokenizer } from '../src/engine/tokenizer'
import { Trainer } from '../src/engine/trainer'

beforeAll(() => usePrecision('f32'))

describe('training loop', () => {
  it('lowers the loss well below the uniform baseline in 300 steps', () => {
    const { sentences, targetVocab } = CORPORA.ja
    const tok = Tokenizer.train(sentences, 'ja', targetVocab)
    const stream = tok.encodeSentences(sentences)
    const params = createParams(makeConfig(tok.vocabSize), mulberry32(1))
    const trainer = new Trainer(params, stream, { seed: 1, lr: 3e-3 })
    const first = trainer.trainStep()
    expect(first.loss).toBeCloseTo(Math.log(tok.vocabSize), 0)
    expect(first.gradNorm).toBeGreaterThan(0)
    expect(first.before).not.toEqual(params.data)
    for (let i = 0; i < 299; i++) trainer.trainStep()
    const tail = trainer.lossHistory.slice(-20)
    const avg = tail.reduce((a, b) => a + b, 0) / tail.length
    expect(avg).toBeLessThan(first.loss * 0.75)
    expect(trainer.step).toBe(300)
  })
})
