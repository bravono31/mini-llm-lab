// Trains the mini GPT on both corpora and writes src/data/pretrained-<lang>.json.
// Usage: npm run pretrain   (STEPS=4000 npm run pretrain to override)
import { writeFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { CORPORA } from '../src/data/corpus'
import { makeConfig } from '../src/engine/config'
import { generate } from '../src/engine/generate'
import { createParams, totalSize } from '../src/engine/params'
import { mulberry32 } from '../src/engine/rng'
import { exportWeights, type Pretrained } from '../src/engine/serialize'
import { Tokenizer, type Lang } from '../src/engine/tokenizer'
import { Trainer } from '../src/engine/trainer'

const STEPS = Number(process.env.STEPS ?? 3000)
const SEED = 1
const PEAK_LR = 5e-3
const MIN_LR = 5e-4
const WARMUP = 100

function lrAt(step: number): number {
  if (step < WARMUP) return (PEAK_LR * (step + 1)) / WARMUP
  const p = (step - WARMUP) / Math.max(1, STEPS - WARMUP)
  return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + Math.cos(Math.PI * p))
}

for (const lang of ['ja', 'en'] as Lang[]) {
  const t0 = performance.now()
  const { sentences, targetVocab, prompts } = CORPORA[lang]
  const tokenizer = Tokenizer.train(sentences, lang, targetVocab)
  const stream = tokenizer.encodeSentences(sentences)
  const config = makeConfig(tokenizer.vocabSize)
  const params = createParams(config, mulberry32(SEED))
  const trainer = new Trainer(params, stream, { batchSize: 16, seed: SEED })

  const sample = () => {
    const ids = tokenizer.encode(prompts[0])
    const out = generate(params, ids, 12, { mode: 'greedy', temperature: 1, k: 5 }, mulberry32(3))
    return tokenizer.decode([...ids, ...out.map((s) => s.chosen)]).replace(/\n$/, '')
  }
  const samples = [{ step: 0, text: sample() }]

  for (let step = 0; step < STEPS; step++) {
    trainer.setLr(lrAt(step))
    trainer.trainStep()
    if ((step + 1) % 500 === 0) samples.push({ step: step + 1, text: sample() })
    if ((step + 1) % 500 === 0) {
      const tail = trainer.lossHistory.slice(-100)
      console.log(`[${lang}] step ${step + 1}  loss ${(tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(3)}  lr ${lrAt(step).toExponential(1)}`)
    }
  }

  const tail = trainer.lossHistory.slice(-100)
  const out: Pretrained = {
    lang,
    seed: SEED,
    config,
    tokenizer: tokenizer.toJSON(),
    weights: exportWeights(params),
    lossHistory: trainer.lossHistory.map((x) => Math.round(x * 1e4) / 1e4),
    samples,
    trainSteps: STEPS,
  }
  const file = resolve('src/data', `pretrained-${lang}.json`)
  writeFileSync(file, JSON.stringify(out))
  console.log(
    `[${lang}] vocab ${tokenizer.vocabSize}  params ${totalSize(params.specs)}  stream ${stream.length} tokens  ` +
      `final loss ${(tail.reduce((a, b) => a + b, 0) / tail.length).toFixed(3)}  ${((performance.now() - t0) / 1000).toFixed(1)}s  ` +
      `${(statSync(file).size / 1024).toFixed(0)} KB`,
  )
  for (const s of samples) console.log(`  step ${String(s.step).padStart(5)}: ${JSON.stringify(s.text)}`)
}
