import type { ComponentType } from 'react'
import { CHAPTER_META, type ChapterMeta } from '../chapterMeta'
import { Attention } from './Attention'
import { Embed } from './Embed'
import { MapChapter } from './Map'
import { Mlp } from './Mlp'
import { MlCompare } from './MlCompare'
import { MlData } from './MlData'
import { MlEval } from './MlEval'
import { MlFeatures } from './MlFeatures'
import { MlModel } from './MlModel'
import { MlOverview } from './MlOverview'
import { MlTrain } from './MlTrain'
import { Output } from './Output'
import { Params } from './Params'
import { Rag } from './Rag'
import { Train } from './Train'
import { Overview } from './Overview'
import { Tokenize } from './Tokenize'

export interface ChapterProps {
  onNavigate: (id: string) => void
}

export interface ChapterDef extends ChapterMeta {
  component: ComponentType<ChapterProps>
}

const COMPONENTS: Record<string, ComponentType<ChapterProps>> = {
  map: MapChapter,
  'ml-overview': MlOverview,
  'ml-data': MlData,
  'ml-features': MlFeatures,
  'ml-model': MlModel,
  'ml-train': MlTrain,
  'ml-eval': MlEval,
  'ml-compare': MlCompare, overview: Overview, tokenize: Tokenize, embed: Embed, attention: Attention, mlp: Mlp, output: Output, train: Train, params: Params, rag: Rag }

export const CHAPTERS: ChapterDef[] = CHAPTER_META.map((m) => ({ ...m, component: COMPONENTS[m.id] }))
