export interface ChapterMeta {
  id: string
  num: string
  title: string
}

export const CHAPTER_META: ChapterMeta[] = [
  { id: 'overview', num: '01', title: '概要' },
  { id: 'tokenize', num: '02', title: 'トークン化' },
  { id: 'embed', num: '03', title: '埋め込み' },
  { id: 'attention', num: '04', title: '注意機構' },
  { id: 'mlp', num: '05', title: 'MLP と残差' },
  { id: 'output', num: '06', title: '出力と次トークン' },
  { id: 'train', num: '07', title: '学習' },
  { id: 'params', num: '08', title: 'パラメータ一覧' },
]

export function chapterLabel(id: string): string {
  const m = CHAPTER_META.find((c) => c.id === id)
  return m ? `第 ${m.num} 章「${m.title}」` : id
}
