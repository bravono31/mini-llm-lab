import type { Lang } from '../engine/tokenizer'
import { CORPUS_EN, PROMPTS_EN } from './corpus-en'
import { CORPUS_JA, PROMPTS_JA } from './corpus-ja'

export interface CorpusInfo {
  lang: Lang
  label: string
  sentences: string[]
  prompts: string[]
  targetVocab: number
  defaultInput: string
}

export const CORPORA: Record<Lang, CorpusInfo> = {
  ja: { lang: 'ja', label: '日本語', sentences: CORPUS_JA, prompts: PROMPTS_JA, targetVocab: 96, defaultInput: 'わたしはねこが' },
  en: { lang: 'en', label: 'English', sentences: CORPUS_EN, prompts: PROMPTS_EN, targetVocab: 64, defaultInput: 'the cat sits on the' },
}
