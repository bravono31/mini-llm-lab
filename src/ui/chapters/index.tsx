import type { ComponentType } from 'react'
import { CHAPTER_META } from '../chapterMeta'
import { Attention } from './Attention'
import { Embed } from './Embed'
import { Mlp } from './Mlp'
import { Output } from './Output'
import { Params } from './Params'
import { Train } from './Train'
import { Overview } from './Overview'
import { Tokenize } from './Tokenize'

export interface ChapterProps {
  onNavigate: (id: string) => void
}

export interface ChapterDef {
  id: string
  num: string
  title: string
  component: ComponentType<ChapterProps>
}

const COMPONENTS: Record<string, ComponentType<ChapterProps>> = { overview: Overview, tokenize: Tokenize, embed: Embed, attention: Attention, mlp: Mlp, output: Output, train: Train, params: Params }

export const CHAPTERS: ChapterDef[] = CHAPTER_META.map((m) => ({ ...m, component: COMPONENTS[m.id] }))
