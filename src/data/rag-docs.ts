import type { Lang } from '../engine/tokenizer'

export interface RagQuestion {
  text: string
  /** index of the document that answers it */
  answer: number
}

export interface RagSet {
  /** facts that are NOT in the training corpus, written with the same vocabulary */
  docs: string[]
  questions: RagQuestion[]
}

export const RAG_SETS: Record<Lang, RagSet> = {
  ja: {
    docs: ['わたしのねこはくろい', 'わたしのねこのなまえはたま', 'あなたのいぬはしろい', 'あなたのいぬはにくをたべる', 'あしたはゆきがふる', 'うまはやまにいる', 'とりはにわでなく', 'わたしははるにうみにいく'],
    questions: [
      { text: 'わたしのねこはなにいろ', answer: 0 },
      { text: 'わたしのねこのなまえは', answer: 1 },
      { text: 'あなたのいぬはなにをたべる', answer: 3 },
      { text: 'あしたはなにがふる', answer: 4 },
      { text: 'はるにいくところ', answer: 7 },
    ],
  },
  en: {
    docs: ['my cat is black', 'my cat likes fish', 'your dog is white', 'your dog eats meat', 'it snows tomorrow', 'the horse is in the mountain', 'the bird sings in the garden', 'we go to the sea in spring'],
    questions: [
      { text: 'what does my cat like', answer: 1 },
      { text: 'what does your dog eat', answer: 3 },
      { text: 'what is the weather tomorrow', answer: 4 },
      { text: 'where is the horse', answer: 5 },
      { text: 'where do we go in spring', answer: 7 },
    ],
  },
}
