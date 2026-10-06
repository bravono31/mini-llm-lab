import { describe, expect, it } from 'vitest'
import { ML_DATA } from '../src/data/ml-labels'
import { accuracy, buildDataset, confusion, featurize, segment, SoftmaxRegression } from '../src/engine/classifier'
import type { Lang } from '../src/engine/tokenizer'
import { numericGrad, randVec } from './helpers'

const LANGS: Lang[] = ['ja', 'en']

describe('ml data and features', () => {
  it.each(LANGS)('labels every %s sentence', (lang) => {
    const d = ML_DATA[lang]
    expect(d.sentences.filter((s) => !d.labels[s])).toEqual([])
    expect(Object.keys(d.labels).length).toBe(d.sentences.length)
  })

  it('segments every ja sentence with the dictionary, no unknown characters', () => {
    const d = ML_DATA.ja
    const bad = d.sentences.filter((s) => segment(s, d).some((w) => w.kind === 'unknown'))
    expect(bad).toEqual([])
  })

  it('prefers the split with known words (ねこ|は|なく, not ねこ|はな|く)', () => {
    expect(segment('ねこはなく', ML_DATA.ja).map((w) => w.text)).toEqual(['ねこ', 'は', 'なく'])
    expect(segment('はなはあかい', ML_DATA.ja).map((w) => w.text)).toEqual(['はな', 'は', 'あかい'])
    expect(segment('ねこはぺんぎん', ML_DATA.ja).map((w) => [w.text, w.kind])).toEqual([
      ['ねこ', 'feature'],
      ['は', 'stop'],
      ['ぺんぎん', 'unknown'],
    ])
  })

  it.each(LANGS)('throws away word order (%s)', (lang) => {
    const d = ML_DATA[lang]
    const ds = buildDataset(d)
    const [a, b] = lang === 'ja' ? ['ねこはいぬをみる', 'いぬはねこをみる'] : ['the cat sees the dog', 'the dog sees the cat']
    expect([...featurize(a, d, ds.vocab)]).toEqual([...featurize(b, d, ds.vocab)])
  })
})

describe('softmax regression', () => {
  it('analytic gradient matches finite differences', () => {
    const ds = buildDataset(ML_DATA.ja)
    const m = new SoftmaxRegression(3, ds.vocab.length)
    m.W.set(randVec(m.W.length, 0.5))
    m.b.set(randVec(3, 0.5))
    const idx = ds.train.slice(0, 12)
    const { dW, db } = m.gradient(ds.X, ds.y, idx)
    const f = () => m.loss(ds.X, ds.y, idx)
    const nW = numericGrad(f, m.W)
    const nb = numericGrad(f, m.b)
    for (let i = 0; i < dW.length; i++) expect(dW[i]).toBeCloseTo(nW[i], 6)
    for (let i = 0; i < 3; i++) expect(db[i]).toBeCloseTo(nb[i], 6)
  })

  it.each(LANGS)('learns: loss falls from ln 3 and test accuracy is well above chance (%s)', (lang) => {
    const ds = buildDataset(ML_DATA[lang])
    const m = new SoftmaxRegression(3, ds.vocab.length)
    const start = m.loss(ds.X, ds.y, ds.train)
    expect(start).toBeCloseTo(Math.log(3), 10)
    let prev = start
    for (let s = 0; s < 300; s++) {
      m.step(ds.X, ds.y, ds.train, 0.5)
      const l = m.loss(ds.X, ds.y, ds.train)
      expect(l).toBeLessThan(prev)
      prev = l
    }
    expect(prev).toBeLessThan(0.3)
    expect(accuracy(m, ds, ds.test)).toBeGreaterThan(0.7)
    const conf = confusion(m, ds, ds.test)
    expect(conf.reduce((a, b) => a + b, 0)).toBe(ds.test.length)
  })
})
