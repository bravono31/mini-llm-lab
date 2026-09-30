import type { Lang } from '../engine/tokenizer'

export interface RagQuestion {
  text: string
  /** index of the document that answers it */
  answer: number
  /** the answering document's opening, used as the prompt for the mini model (it can only continue text) */
  cloze: string
}

export interface RagSet {
  /** facts that are NOT in the training corpus, written with the same vocabulary */
  docs: string[]
  questions: RagQuestion[]
}

export const RAG_SETS: Record<Lang, RagSet> = {
  ja: {
    docs: ['わたしのねこはくろい', 'ねこのなまえはたま', 'あなたのいぬはしろい', 'あなたのいぬはにくをたべる', 'あしたはゆきがふる', 'うまはやまにいる', 'とりはにわでなく', 'わたしははるにうみにいく'],
    questions: [
      { text: 'わたしのねこはなにいろ', answer: 0, cloze: 'わたしのねこは' },
      { text: 'ねこのなまえは', answer: 1, cloze: 'ねこのなまえは' },
      { text: 'あなたのいぬはなにをたべる', answer: 3, cloze: 'あなたのいぬは' },
      { text: 'あしたはなにがふる', answer: 4, cloze: 'あしたは' },
      { text: 'はるにいくところ', answer: 7, cloze: 'わたしははるに' },
    ],
  },
  en: {
    docs: ['my cat is black', 'my cat likes fish', 'your dog is white', 'your dog eats meat', 'it snows tomorrow', 'the horse is in the mountain', 'the bird sings in the garden', 'we go to the sea in spring'],
    questions: [
      { text: 'what does my cat like', answer: 1, cloze: 'my cat likes' },
      { text: 'what does your dog eat', answer: 3, cloze: 'your dog eats' },
      { text: 'what is the weather tomorrow', answer: 4, cloze: 'it snows' },
      { text: 'where is the horse', answer: 5, cloze: 'the horse is in the' },
      { text: 'where do we go in spring', answer: 7, cloze: 'we go to the sea in' },
    ],
  },
}
