// Minimal byte-pair-encoding tokenizer with full training / encoding traces
// so the UI can replay every merge.
export type Lang = 'ja' | 'en'
export const EOS = '<eos>'
export const UNK = '<unk>'
export const EOS_ID = 0
export const UNK_ID = 1

export type Pair = [string, string]

export interface PairCount {
  pair: Pair
  count: number
}

export interface MergeRecord {
  step: number
  pair: Pair
  token: string
  count: number
  /** most frequent pairs at this step (before merging), for display */
  topPairs: PairCount[]
  vocabSize: number
}

export interface EncodeStep {
  chunks: string[][]
  merged: Pair | null
  rank: number | null
}

export interface EncodeTrace {
  chunks: string[][]
  steps: EncodeStep[]
  tokens: string[]
  ids: number[]
  unknownChars: string[]
}

export interface TokenizerJSON {
  lang: Lang
  vocab: string[]
  merges: Pair[]
  mergeLog: MergeRecord[]
  tokenFreq: number[]
}

// separator that never appears in corpus text
const KEY = '∷'
const pairKey = (a: string, b: string) => a + KEY + b

/** Split text into chunks that merges never cross. EN: "␣word" per word. JA: the whole text. */
export function preTokenize(text: string, lang: Lang): string[] {
  if (lang === 'en') {
    const words = text.toLowerCase().trim().split(/\s+/).filter(Boolean)
    return words.map((w) => ' ' + w)
  }
  const t = text.replace(/\s+/g, '')
  return t ? [t] : []
}

export function displayToken(tok: string): string {
  if (tok === EOS) return '⟨eos⟩'
  if (tok === UNK) return '⟨unk⟩'
  return tok.replace(/ /g, '␣')
}

export class Tokenizer {
  readonly lang: Lang
  readonly vocab: string[]
  readonly merges: Pair[]
  readonly mergeLog: MergeRecord[]
  readonly tokenFreq: number[]
  private readonly tokenToId: Map<string, number>
  private readonly ranks: Map<string, number>

  constructor(json: TokenizerJSON) {
    this.lang = json.lang
    this.vocab = json.vocab
    this.merges = json.merges
    this.mergeLog = json.mergeLog
    this.tokenFreq = json.tokenFreq
    this.tokenToId = new Map(this.vocab.map((t, i) => [t, i]))
    this.ranks = new Map(this.merges.map((m, i) => [pairKey(m[0], m[1]), i]))
  }

  get vocabSize(): number {
    return this.vocab.length
  }

  idOf(tok: string): number {
    return this.tokenToId.get(tok) ?? UNK_ID
  }

  has(tok: string): boolean {
    return this.tokenToId.has(tok)
  }

  static train(sentences: string[], lang: Lang, targetVocab: number): Tokenizer {
    const chunkCounts = new Map<string, { syms: string[]; count: number }>()
    const charSet = new Set<string>()
    for (const s of sentences) {
      for (const ch of preTokenize(s, lang)) {
        const syms = [...ch]
        syms.forEach((c) => charSet.add(c))
        const key = syms.join(KEY)
        const e = chunkCounts.get(key)
        if (e) e.count++
        else chunkCounts.set(key, { syms, count: 1 })
      }
    }
    const vocab = [EOS, UNK, ...[...charSet].sort()]
    const merges: Pair[] = []
    const mergeLog: MergeRecord[] = []
    const chunks = [...chunkCounts.values()]

    while (vocab.length < targetVocab) {
      const counts = new Map<string, PairCount>()
      for (const ch of chunks) {
        for (let i = 0; i + 1 < ch.syms.length; i++) {
          const k = pairKey(ch.syms[i], ch.syms[i + 1])
          const e = counts.get(k)
          if (e) e.count += ch.count
          else counts.set(k, { pair: [ch.syms[i], ch.syms[i + 1]], count: ch.count })
        }
      }
      if (counts.size === 0) break
      const sorted = [...counts.values()].sort((a, b) => b.count - a.count)
      const best = sorted[0]
      if (best.count < 2) break
      const [a, b] = best.pair
      const token = a + b
      for (const ch of chunks) ch.syms = mergeSyms(ch.syms, a, b, token)
      vocab.push(token)
      merges.push([a, b])
      mergeLog.push({ step: merges.length, pair: [a, b], token, count: best.count, topPairs: sorted.slice(0, 5), vocabSize: vocab.length })
    }

    const tokenFreq = new Array<number>(vocab.length).fill(0)
    const idOf = new Map(vocab.map((t, i) => [t, i]))
    for (const ch of chunks) for (const s of ch.syms) tokenFreq[idOf.get(s)!] += ch.count
    tokenFreq[EOS_ID] = sentences.length
    return new Tokenizer({ lang, vocab, merges, mergeLog, tokenFreq })
  }

  encodeTrace(text: string): EncodeTrace {
    const unknown = new Set<string>()
    const chunks = preTokenize(text, this.lang).map((ch) =>
      [...ch].map((c) => {
        if (this.tokenToId.has(c)) return c
        unknown.add(c)
        return UNK
      }),
    )
    const initial = chunks.map((c) => [...c])
    const steps: EncodeStep[] = []
    for (;;) {
      let best: { pair: Pair; rank: number } | null = null
      for (const ch of chunks) {
        for (let i = 0; i + 1 < ch.length; i++) {
          const r = this.ranks.get(pairKey(ch[i], ch[i + 1]))
          if (r !== undefined && (best === null || r < best.rank)) best = { pair: [ch[i], ch[i + 1]], rank: r }
        }
      }
      if (!best) break
      const [a, b] = best.pair
      for (let c = 0; c < chunks.length; c++) chunks[c] = mergeSyms(chunks[c], a, b, a + b)
      steps.push({ chunks: chunks.map((c) => [...c]), merged: best.pair, rank: best.rank })
    }
    const tokens = chunks.flat()
    return { chunks: initial, steps, tokens, ids: tokens.map((t) => this.idOf(t)), unknownChars: [...unknown] }
  }

  encode(text: string): number[] {
    return this.encodeTrace(text).ids
  }

  /** Whole corpus as one token stream: <eos> s1 <eos> s2 ... <eos> */
  encodeSentences(sentences: string[]): Int32Array {
    const out: number[] = [EOS_ID]
    for (const s of sentences) out.push(...this.encode(s), EOS_ID)
    return Int32Array.from(out)
  }

  decode(ids: ArrayLike<number>): string {
    let s = ''
    for (let i = 0; i < ids.length; i++) {
      const t = this.vocab[ids[i]] ?? UNK
      s += t === EOS ? '\n' : t === UNK ? '?' : t
    }
    return this.lang === 'en' ? s.replace(/^ /, '').replace(/\n /g, '\n') : s
  }

  tokensOf(ids: ArrayLike<number>): string[] {
    return Array.from(ids, (i) => this.vocab[i] ?? UNK)
  }

  toJSON(): TokenizerJSON {
    return { lang: this.lang, vocab: this.vocab, merges: this.merges, mergeLog: this.mergeLog, tokenFreq: this.tokenFreq }
  }
}

export function mergeSyms(syms: string[], a: string, b: string, token: string): string[] {
  const out: string[] = []
  for (let i = 0; i < syms.length; i++) {
    if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
      out.push(token)
      i++
    } else out.push(syms[i])
  }
  return out
}
