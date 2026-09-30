// Fine-tunes the pretrained mini GPT to copy facts from a document placed before the prompt,
// and writes src/data/rag-<lang>.json. Usage: npm run rag-finetune   (STEPS=4000 LANGS=ja,en to override)
// Only Japanese is shipped: English words outside the vocabulary split into letters, so most
// "document + prompt" pairs exceed the 16-token context and the model cannot practise reading them.
import { writeFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import pretrainedEn from '../src/data/pretrained-en.json'
import pretrainedJa from '../src/data/pretrained-ja.json'
import { CORPORA } from '../src/data/corpus'
import { RAG_SETS } from '../src/data/rag-docs'
import { generate } from '../src/engine/generate'
import { mulberry32, type Rng } from '../src/engine/rng'
import { exportWeights, loadPretrained, type Pretrained } from '../src/engine/serialize'
import type { RagCheckpoints } from '../src/engine/rag'
import { EOS_ID, UNK_ID, type Lang, type Tokenizer } from '../src/engine/tokenizer'
import { Trainer, type Batch } from '../src/engine/trainer'
import type { Params } from '../src/engine/params'

const STEPS = Number(process.env.STEPS ?? 10000)
const CHECKPOINTS = [500, 2000, 5000, STEPS]
const LANGS = (process.env.LANGS ?? 'ja').split(',') as Lang[]
const SEED = 11
const SYN = Number(process.env.SYN ?? 12)
const PEAK_LR = 3e-3
const MIN_LR = 3e-4
const WARMUP = 100
const PRE: Record<Lang, Pretrained> = { ja: pretrainedJa as unknown as Pretrained, en: pretrainedEn as unknown as Pretrained }

const pick = <T>(rng: Rng, xs: T[]): T => xs[Math.floor(rng() * xs.length)]

/** A synthetic fact: `prefix` is what the prompt gives, `answer` must be read from the document. */
type Fact = { prefix: string; answer: string }

const randomWord = (rng: Rng, letters: string, min: number, max: number) => Array.from({ length: min + Math.floor(rng() * (max - min + 1)) }, () => pick(rng, [...letters])).join('')

/** Facts whose answer is a random string: they cannot be memorised, only copied. */
const RANDOM: Record<Lang, (rng: Rng) => Fact> = {
  ja: (rng) => ({ prefix: `${pick(rng, ['わたしの', 'あなたの', ''])}${pick(rng, ['ねこ', 'いぬ', 'とり', 'うま'])}${pick(rng, ['は', 'のなまえは'])}`, answer: randomWord(rng, 'たまぽちみけこもはなそらくろしきゆれぬふあいうえおかさつてとにのひへほやよりわ', 2, 3) }),
  en: (rng) => ({ prefix: `${pick(rng, ['my', 'your'])} ${pick(rng, ['cat', 'dog', 'bird'])} ${pick(rng, ['is', 'likes'])}`, answer: ` ${randomWord(rng, 'abcdefghiklmnoprstuvwy', 3, 5)}` }),
}

/** A fifth of the template facts never appear in training; the copy score is measured on them. */
function isHeldOutCombo(text: string): boolean {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619)
  return (h >>> 0) % 5 === 0
}

const GEN: Record<Lang, (rng: Rng) => Fact> = {
  ja: (rng) => {
    const owner = pick(rng, ['わたしの', 'あなたの'])
    const animal = pick(rng, ['ねこ', 'いぬ', 'とり', 'うま', 'さかな'])
    switch (Math.floor(rng() * 5)) {
      case 0:
        return { prefix: `${owner}${animal}は`, answer: pick(rng, ['しろい', 'くろい', 'あかい', 'あおい', 'ちいさい', 'おおきい', 'かわいい', 'きれい', 'はしる', 'ねる', 'なく', 'にくをたべる', 'さかなをたべる', 'くさをたべる', 'みずをのむ', 'おちゃをのむ']) }
      case 1: {
        const syl = ['た', 'ま', 'ぽ', 'ち', 'み', 'け', 'こ', 'も', 'は', 'な', 'そ', 'ら', 'く', 'ろ', 'し', 'き', 'ゆ', 'れ', 'ぬ', 'ふ']
        const len = 2 + Math.floor(rng() * 2)
        return { prefix: `${animal}のなまえは`, answer: Array.from({ length: len }, () => pick(rng, syl)).join('') }
      }
      case 2:
        return { prefix: `${pick(rng, ['きょう', 'あした', 'なつ', 'ふゆ', 'はる', 'あき'])}は`, answer: pick(rng, ['あめがふる', 'ゆきがふる', 'はれる', 'さむい', 'あつい', 'かぜがふく']) }
      case 3:
        return { prefix: `${animal}は`, answer: `${pick(rng, ['やま', 'うみ', 'いえ', 'にわ', 'そら', 'かわ'])}にいる` }
      default:
        return { prefix: `わたしは${pick(rng, ['はる', 'なつ', 'あき', 'ふゆ'])}に`, answer: `${pick(rng, ['うみ', 'やま', 'いえ', 'にわ'])}にいく` }
    }
  },
  en: (rng) => {
    const owner = pick(rng, ['my', 'your'])
    const animal = pick(rng, ['cat', 'dog', 'bird', 'horse', 'fish'])
    switch (Math.floor(rng() * 4)) {
      case 0: {
        const [verb, obj] = pick(rng, [['is', 'black'], ['is', 'white'], ['is', 'red'], ['is', 'small'], ['is', 'big'], ['likes', 'fish'], ['likes', 'meat'], ['likes', 'milk'], ['eats', 'seeds'], ['eats', 'grass'], ['eats', 'meat'], ['drinks', 'milk'], ['drinks', 'water'], ['drinks', 'tea']])
        return { prefix: `${owner} ${animal} ${verb}`, answer: ` ${obj}` }
      }
      case 1:
        return { prefix: `it ${pick(rng, ['rains', 'snows'])}`, answer: ` ${pick(rng, ['today', 'tomorrow'])}` }
      case 2:
        return { prefix: `the ${animal} is in the`, answer: ` ${pick(rng, ['mountain', 'garden', 'park', 'house', 'tree', 'sea', 'river', 'field'])}` }
      default:
        return { prefix: `we go to the ${pick(rng, ['sea', 'park', 'mountain', 'river'])} in`, answer: ` ${pick(rng, ['spring', 'summer', 'autumn', 'winter'])}` }
    }
  },
}

function lrAt(step: number): number {
  if (step < WARMUP) return (PEAK_LR * (step + 1)) / WARMUP
  const p = (step - WARMUP) / Math.max(1, STEPS - WARMUP)
  return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + Math.cos(Math.PI * p))
}

/** ⟨eos⟩ document ⟨eos⟩ document ⟨eos⟩ ... : the second copy can only be completed by reading the first. */
function episode(tok: Tokenizer, f: Fact): number[] {
  const doc = tok.encode(f.prefix + f.answer)
  return [EOS_ID, ...doc, EOS_ID, ...doc]
}

function run(lang: Lang) {
  const t0 = performance.now()
  const { params, tokenizer } = loadPretrained(PRE[lang])
  const T = params.config.ctxLen
  const rng = mulberry32(SEED)
  const heldOut = new Set(RAG_SETS[lang].docs)
  const corpus = tokenizer.encodeSentences(CORPORA[lang].sentences)
  const trainer = new Trainer(params, corpus, { batchSize: 16, seed: SEED })

  const fits = (f: Fact) => {
    const ids = tokenizer.encode(f.prefix + f.answer)
    return !ids.includes(UNK_ID) && 2 * ids.length + 2 <= T
  }
  const syntheticFact = (): Fact => {
    for (;;) {
      const f = rng() < 0.5 ? RANDOM[lang](rng) : GEN[lang](rng)
      const text = f.prefix + f.answer
      if (!heldOut.has(text) && !isHeldOutCombo(text) && fits(f)) return f
    }
  }

  const batch = (): Batch => {
    const corpusRows = trainer.sampleBatch(16 - SYN)
    const B = 16
    const inputs = new Int32Array(B * T)
    const targets = new Int32Array(B * T)
    inputs.set(corpusRows.inputs)
    targets.set(corpusRows.targets)
    for (let b = 16 - SYN; b < B; b++) {
      const seq: number[] = []
      while (seq.length < T + 1) seq.push(...episode(tokenizer, syntheticFact()))
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = seq[t]
        targets[b * T + t] = seq[t + 1]
      }
    }
    return { inputs, targets, starts: [] }
  }

  // copying accuracy on template facts that were never trained on (not the chapter's documents)
  const syntheticAccuracy = () => {
    const r = mulberry32(99)
    let hit = 0
    const n = 200
    for (let i = 0; i < n; i++) {
      let f: Fact
      do f = GEN[lang](r)
      while (!isHeldOutCombo(f.prefix + f.answer) || !fits(f))
      const ids = [EOS_ID, ...tokenizer.encode(f.prefix + f.answer), EOS_ID, ...tokenizer.encode(f.prefix)]
      const out = generate(params, ids, 8, { mode: 'greedy', temperature: 1, k: 5 }, mulberry32(1))
      if (tokenizer.decode(out.map((s) => s.chosen)).split('\n')[0].trim().startsWith(f.answer.trim())) hit++
    }
    return Math.round((100 * hit) / n)
  }
  // step 0 is the untouched pretrained model; the app reuses its weights, so none are stored
  const checkpoints: RagCheckpoints['checkpoints'] = [{ step: 0, copyAcc: syntheticAccuracy(), weights: [] }]

  for (let step = 0; step < STEPS; step++) {
    trainer.setLr(lrAt(step))
    trainer.trainStep(batch())
    if (CHECKPOINTS.includes(step + 1)) checkpoints.push({ step: step + 1, copyAcc: syntheticAccuracy(), weights: exportWeights(params, 4) })
    if ((step + 1) % 500 === 0) {
      const tail = trainer.lossHistory.slice(-100)
      console.log(`[${lang}] step ${step + 1}  loss ${(tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(3)}  held-out ${evaluate(params, tokenizer, lang).withDoc}  synthetic ${syntheticAccuracy()}%`)
    }
  }

  const ev = evaluate(params, tokenizer, lang, true)
  const tail = trainer.lossHistory.slice(-100)
  const loss: number[] = []
  for (let i = 0; i < trainer.lossHistory.length; i += 50) {
    const w = trainer.lossHistory.slice(i, i + 50)
    loss.push(Math.round((w.reduce((x, y) => x + y, 0) / w.length) * 1e3) / 1e3)
  }
  const out: RagCheckpoints = { lang, checkpoints, loss }
  const file = resolve('src/data', `rag-${lang}.json`)
  writeFileSync(file, JSON.stringify(out))
  console.log(`[${lang}] final loss ${(tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(3)}  held-out with doc ${ev.withDoc}  ${((performance.now() - t0) / 1000).toFixed(1)}s  ${(statSync(file).size / 1024).toFixed(0)} KB`)
}

/** Greedy completion of each chapter cloze, with and without its document in front. */
function evaluate(params: Params, tok: Tokenizer, lang: Lang, verbose = false) {
  const set = RAG_SETS[lang]
  let hit = 0
  for (const q of set.questions) {
    const doc = set.docs[q.answer]
    const cloze = q.cloze
    const answer = doc.slice(cloze.length)
    const complete = (ids: number[]) => {
      const steps = generate(params, ids, 8, { mode: 'greedy', temperature: 1, k: 5 }, mulberry32(1))
      return tok.decode(steps.map((s) => s.chosen)).split('\n')[0]
    }
    const cl = tok.encode(cloze)
    const withDoc = complete([EOS_ID, ...tok.encode(doc), EOS_ID, ...cl])
    const noDoc = complete([EOS_ID, ...cl])
    const norm = (s: string) => s.trim()
    if (norm(withDoc).startsWith(norm(answer))) hit++
    if (verbose) console.log(`  ${cloze} | 文書なし→${JSON.stringify(noDoc)}  文書あり→${JSON.stringify(withDoc)}  正解 ${JSON.stringify(answer)}`)
  }
  return { withDoc: `${hit}/${set.questions.length}` }
}

for (const lang of LANGS) run(lang)
