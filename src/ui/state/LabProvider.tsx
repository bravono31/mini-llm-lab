import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { CORPORA, type CorpusInfo } from '../../data/corpus'
import pretrainedEn from '../../data/pretrained-en.json'
import pretrainedJa from '../../data/pretrained-ja.json'
import { forward, type Activations } from '../../engine/forward'
import { cloneParams, createParams, type Params } from '../../engine/params'
import { mulberry32 } from '../../engine/rng'
import { loadPretrained, type Pretrained } from '../../engine/serialize'
import { EOS_ID, Tokenizer, type EncodeTrace, type Lang } from '../../engine/tokenizer'
import { Trainer } from '../../engine/trainer'

const PRETRAINED: Record<Lang, Pretrained> = {
  ja: pretrainedJa as unknown as Pretrained,
  en: pretrainedEn as unknown as Pretrained,
}

export type Origin = 'pretrained' | 'random'

export interface Session {
  params: Params
  trainer: Trainer
  origin: Origin
  /** the untouched pretrained weights, for comparison */
  pretrainedParams: Params
}

export interface Lab {
  lang: Lang
  setLang: (l: Lang) => void
  corpus: CorpusInfo
  tokenizer: Tokenizer
  pretrained: Pretrained
  session: Session
  params: Params
  /** bumps whenever params are mutated (training / reset) */
  version: number
  notifyParamsChanged: () => void
  resetToPretrained: () => void
  resetToRandom: () => void
  input: string
  setInput: (s: string) => void
  trace: EncodeTrace
  /** token ids fed to the model (capped to the context length; never empty) */
  ids: number[]
  truncated: boolean
  acts: Activations
}

const Ctx = createContext<Lab | null>(null)

function makeSession(lang: Lang, tokenizer: Tokenizer, corpus: CorpusInfo, origin: Origin, seed: number): Session {
  const { params: pre } = loadPretrained(PRETRAINED[lang])
  const params = origin === 'pretrained' ? cloneParams(pre) : createParams(pre.config, mulberry32(seed))
  const stream = tokenizer.encodeSentences(corpus.sentences)
  const trainer = new Trainer(params, stream, { batchSize: 16, seed, lr: 3e-3 })
  return { params, trainer, origin, pretrainedParams: pre }
}

export function LabProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>('ja')
  const corpus = CORPORA[lang]
  const pretrained = PRETRAINED[lang]
  const tokenizer = useMemo(() => new Tokenizer(pretrained.tokenizer), [pretrained])
  const [session, setSession] = useState<Session>(() => makeSession('ja', tokenizer, corpus, 'pretrained', 1))
  const [version, setVersion] = useState(0)
  const [inputs, setInputs] = useState<Record<Lang, string>>({ ja: CORPORA.ja.defaultInput, en: CORPORA.en.defaultInput })
  const [randomSeed, setRandomSeed] = useState(7)

  const setLang = useCallback(
    (l: Lang) => {
      if (l === lang) return
      const tok = new Tokenizer(PRETRAINED[l].tokenizer)
      setLangState(l)
      setSession(makeSession(l, tok, CORPORA[l], 'pretrained', 1))
      setVersion((v) => v + 1)
    },
    [lang],
  )

  const resetToPretrained = useCallback(() => {
    setSession(makeSession(lang, tokenizer, corpus, 'pretrained', 1))
    setVersion((v) => v + 1)
  }, [lang, tokenizer, corpus])

  const resetToRandom = useCallback(() => {
    const seed = randomSeed + 1
    setRandomSeed(seed)
    setSession(makeSession(lang, tokenizer, corpus, 'random', seed))
    setVersion((v) => v + 1)
  }, [lang, tokenizer, corpus, randomSeed])

  const notifyParamsChanged = useCallback(() => setVersion((v) => v + 1), [])

  const input = inputs[lang]
  const setInput = useCallback((s: string) => setInputs((prev) => ({ ...prev, [lang]: s })), [lang])

  const trace = useMemo(() => tokenizer.encodeTrace(input), [tokenizer, input])
  const ctxLen = session.params.config.ctxLen
  const ids = useMemo(() => (trace.ids.length ? trace.ids.slice(0, ctxLen) : [EOS_ID]), [trace, ctxLen])
  const truncated = trace.ids.length > ctxLen
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const acts = useMemo(() => forward(session.params, ids, 1, ids.length), [session, ids, version])

  const value: Lab = useMemo(
    () => ({
      lang, setLang, corpus, tokenizer, pretrained, session, params: session.params, version, notifyParamsChanged,
      resetToPretrained, resetToRandom, input, setInput, trace, ids, truncated, acts,
    }),
    [lang, setLang, corpus, tokenizer, pretrained, session, version, notifyParamsChanged, resetToPretrained, resetToRandom, input, setInput, trace, ids, truncated, acts],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useLab(): Lab {
  const v = useContext(Ctx)
  if (!v) throw new Error('LabProvider missing')
  return v
}
