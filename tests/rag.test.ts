import { describe, expect, it } from 'vitest'
import { CORPORA } from '../src/data/corpus'
import pretrainedEn from '../src/data/pretrained-en.json'
import pretrainedJa from '../src/data/pretrained-ja.json'
import { RAG_SETS } from '../src/data/rag-docs'
import { cosine, idfWeights, retrieve } from '../src/engine/rag'
import { loadPretrained, type Pretrained } from '../src/engine/serialize'
import { UNK_ID, type Lang } from '../src/engine/tokenizer'

const PRE: Record<Lang, Pretrained> = { ja: pretrainedJa as unknown as Pretrained, en: pretrainedEn as unknown as Pretrained }

describe('rag retrieval math', () => {
  it('cosine is 1 for parallel, 0 for disjoint and for the zero vector', () => {
    expect(cosine([1, 2, 0], [2, 4, 0])).toBeCloseTo(1)
    expect(cosine([1, 0, 0], [0, 3, 0])).toBe(0)
    expect(cosine([0, 0, 0], [1, 1, 1])).toBe(0)
  })

  it('gives rarer tokens a larger idf', () => {
    const idf = idfWeights([[0, 1], [0, 2], [0, 3]], 4)
    expect(idf[0]).toBeLessThan(idf[1])
    expect(idf[1]).toBeCloseTo(idf[2])
  })

  it('ranks the document that shares the rare token first', () => {
    const r = retrieve([2, 0], [[0, 1], [0, 2], [0, 3]], 4)
    expect(r.order[0]).toBe(1)
  })
})

for (const lang of ['ja', 'en'] as Lang[]) {
  describe(`rag documents (${lang})`, () => {
    const { tokenizer } = loadPretrained(PRE[lang])
    const set = RAG_SETS[lang]
    const docIds = set.docs.map((d) => tokenizer.encode(d))

    it('uses only known tokens', () => {
      for (const s of [...set.docs, ...set.questions.map((q) => q.text)]) expect(tokenizer.encode(s)).not.toContain(UNK_ID)
    })

    it('is not part of the training corpus', () => {
      const corpus = new Set(CORPORA[lang].sentences)
      for (const d of set.docs) expect(corpus.has(d)).toBe(false)
    })

    it('retrieves the answering document first for every preset question', () => {
      for (const q of set.questions) expect(retrieve(tokenizer.encode(q.text), docIds, tokenizer.vocabSize).order[0]).toBe(q.answer)
    })
  })
}
