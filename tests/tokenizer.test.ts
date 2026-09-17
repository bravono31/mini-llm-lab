import { describe, expect, it } from 'vitest'
import { CORPORA } from '../src/data/corpus'
import { EOS_ID, Tokenizer, UNK_ID } from '../src/engine/tokenizer'

describe('mini BPE (ja)', () => {
  const { sentences, targetVocab } = CORPORA.ja
  const tok = Tokenizer.train(sentences, 'ja', targetVocab)

  it('reaches the target vocabulary deterministically', () => {
    expect(tok.vocabSize).toBe(targetVocab)
    expect(tok.mergeLog.length).toBe(tok.merges.length)
    const again = Tokenizer.train(sentences, 'ja', targetVocab)
    expect(JSON.stringify(again.toJSON())).toBe(JSON.stringify(tok.toJSON()))
  })

  it('round-trips every corpus sentence', () => {
    for (const s of sentences) expect(tok.decode(tok.encode(s))).toBe(s)
  })

  it('merges frequent words into single tokens', () => {
    const trace = tok.encodeTrace('ねこがすき')
    expect(trace.ids.length).toBeLessThan(5)
    expect(trace.steps.length).toBeGreaterThan(0)
    expect(trace.steps[trace.steps.length - 1].chunks.flat()).toEqual(trace.tokens)
    expect(tok.has('ねこ')).toBe(true)
  })

  it('reports unknown characters and maps them to <unk>', () => {
    const trace = tok.encodeTrace('ねこ猫')
    expect(trace.unknownChars).toEqual(['猫'])
    expect(trace.ids).toContain(UNK_ID)
  })

  it('builds an <eos>-delimited stream', () => {
    const stream = tok.encodeSentences(sentences.slice(0, 3))
    expect(stream[0]).toBe(EOS_ID)
    expect(stream[stream.length - 1]).toBe(EOS_ID)
    expect(Array.from(stream).filter((x) => x === EOS_ID)).toHaveLength(4)
  })

  it('survives JSON round trip', () => {
    const copy = new Tokenizer(JSON.parse(JSON.stringify(tok.toJSON())))
    expect(copy.encode('わたしはねこがすき')).toEqual(tok.encode('わたしはねこがすき'))
  })
})

describe('mini BPE (en)', () => {
  const { sentences, targetVocab } = CORPORA.en
  const tok = Tokenizer.train(sentences, 'en', targetVocab)

  it('round-trips every corpus sentence', () => {
    expect(tok.vocabSize).toBe(targetVocab)
    for (const s of sentences) expect(tok.decode(tok.encode(s))).toBe(s)
  })

  it('treats each word as a space-prefixed chunk', () => {
    const trace = tok.encodeTrace('The cat')
    expect(trace.chunks).toEqual([[' ', 't', 'h', 'e'], [' ', 'c', 'a', 't']])
    expect(trace.tokens).toEqual([' the', ' cat'])
  })
})
