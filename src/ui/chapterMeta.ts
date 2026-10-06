export type Part = 'map' | 'ml' | 'llm'

export interface ChapterMeta {
  id: string
  part: Part
  num: string
  title: string
}

export const PART_TITLES: Record<Part, string> = { map: '全体像', ml: '機械学習編', llm: 'LLM 編' }

export const CHAPTER_META: ChapterMeta[] = [
  { id: 'overview', part: 'llm', num: 'LLM-1', title: '概要' },
  { id: 'tokenize', part: 'llm', num: 'LLM-2', title: 'トークン化' },
  { id: 'embed', part: 'llm', num: 'LLM-3', title: '埋め込み' },
  { id: 'attention', part: 'llm', num: 'LLM-4', title: '注意機構' },
  { id: 'mlp', part: 'llm', num: 'LLM-5', title: 'MLP と残差' },
  { id: 'output', part: 'llm', num: 'LLM-6', title: '出力と次トークン' },
  { id: 'train', part: 'llm', num: 'LLM-7', title: '学習' },
  { id: 'params', part: 'llm', num: 'LLM-8', title: 'パラメータ一覧' },
  { id: 'rag', part: 'llm', num: 'LLM-9', title: 'RAG（検索拡張）' },
]

export function chapterMeta(id: string): ChapterMeta | undefined {
  return CHAPTER_META.find((c) => c.id === id)
}

/** e.g. "LLM-7「学習」" */
export function chapterLabel(id: string): string {
  const m = chapterMeta(id)
  return m ? (m.part === 'map' ? `「${m.title}」` : `${m.num}「${m.title}」`) : id
}
