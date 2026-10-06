import type { Lang } from '../engine/tokenizer'
import { CORPUS_EN } from './corpus-en'
import { CORPUS_JA } from './corpus-ja'

/**
 * Data for the machine-learning part: the same sentences as the LLM corpus, each labelled by hand with its topic.
 * Rule used for labelling: "what is the sentence about?" — its subject / topic word.
 *   animal: an animal is the topic (「わたしのねこはしろい」 is about the cat, so it is animal)
 *   people: わたし / あなた (i / you / we) is the topic
 *   nature: sky, sea, weather, seasons, plants, stars …
 */

export type ClassId = 'animal' | 'nature' | 'people'
export const CLASS_IDS: ClassId[] = ['animal', 'nature', 'people']

export interface MlData {
  lang: Lang
  classNames: Record<ClassId, string>
  sentences: string[]
  labels: Record<string, ClassId>
  /** ja only: content words the segmenter knows (a hand-made dictionary) */
  dict: string[]
  /** words that are recognised but not used as features (particles / articles) */
  stop: string[]
}

const A: ClassId = 'animal'
const N: ClassId = 'nature'
const P: ClassId = 'people'

const LABELS_JA: Record<string, ClassId> = {
  ねこがすき: A, いぬがすき: A, とりがすき: A, わたしはねこがすき: P, わたしはいぬがすき: P, あなたはとりがすき: P,
  ねこはさかなをたべる: A, いぬはにくをたべる: A, とりはむしをたべる: A, うまはくさをたべる: A, わたしはごはんをたべる: P,
  ねこはみずをのむ: A, いぬはみずをのむ: A, わたしはおちゃをのむ: P, あなたはみずをのむ: P,
  いぬははしる: A, うまははしる: A, ねこははしる: A, ねこはねる: A, いぬもねる: A, わたしはねる: P,
  とりはそらをとぶ: A, さかなはうみをおよぐ: A, ねこはなく: A, とりはなく: A,
  わたしはそらをみる: P, あなたはうみをみる: P, ねこはつきをみる: A, いぬはほしをみる: A,
  そらはあおい: N, うみはあおい: N, はなはあかい: N, ゆきはしろい: N, ねこはしろい: A, いぬはくろい: A,
  ねこはかわいい: A, いぬはかわいい: A, とりはちいさい: A, うまはおおきい: A, やまはおおきい: N,
  はなはきれい: N, つきはきれい: N, ほしはひかる: N, つきはひかる: N,
  あめがふる: N, ゆきがふる: N, かぜがふく: N, ひがでる: N, つきがでる: N,
  きょうはあめがふる: N, あしたははれる: N, きょうはさむい: N, なつはあつい: N, ふゆはさむい: N,
  はるははながさく: N, あきはつきがきれい: N,
  ねこのこはちいさい: A, いぬのこはかわいい: A, わたしのねこはしろい: A, あなたのいぬはくろい: A,
  ねこといぬがねる: A, とりとさかながすき: A,
  わたしはやまにいく: P, あなたはうみにいく: P, ねこはいえにいる: A, いぬはにわにいる: A, ほしがそらにひかる: N,
}

const LABELS_EN: Record<string, ClassId> = {
  'the cat sits on the mat': A, 'the dog sits on the rug': A, 'the cat sleeps on the bed': A, 'the dog sleeps on the floor': A,
  'the bird sings in the tree': A, 'the bird flies in the sky': A, 'the fish swims in the sea': A, 'the fish swims in the river': A,
  'the cat eats fish': A, 'the dog eats meat': A, 'the bird eats seeds': A, 'the horse eats grass': A,
  'i like cats': P, 'i like dogs': P, 'you like birds': P, 'we like fish': P,
  'i see the moon': P, 'you see the sun': P, 'we see the stars': P, 'the cat sees the bird': A, 'the dog sees the cat': A,
  'the sky is blue': N, 'the sea is blue': N, 'the sun is bright': N, 'the moon is bright': N,
  'the snow is white': N, 'the cat is white': A, 'the dog is black': A, 'the cat is small': A, 'the horse is big': A,
  'the tree is tall': N, 'the flower is red': N, 'the flower is pretty': N,
  'the stars shine at night': N, 'the sun shines by day': N, 'it rains today': N, 'it snows today': N,
  'the wind blows': N, 'the rain falls': N, 'the snow falls': N,
  'the cat runs in the garden': A, 'the dog runs in the park': A, 'the horse runs in the field': A,
  'the cat is in the house': A, 'the dog is in the garden': A, 'the bird is in the tree': A,
  'i go to the park': P, 'you go to the sea': P, 'we go to the mountain': P,
  'my cat is white': A, 'your dog is black': A, 'the cat and the dog sleep': A, 'the bird and the fish are small': A,
  'the cat likes fish': A, 'the dog likes meat': A, 'i drink tea': P, 'you drink water': P, 'the cat drinks milk': A, 'the dog drinks water': A,
  'it is cold in winter': N, 'it is hot in summer': N, 'the flowers bloom in spring': N, 'the moon is pretty in autumn': N,
}

// 人が作った単語辞書（名詞・動詞・形容詞）。文はこの辞書の語と助詞の並びとして区切る。
const DICT_JA = [
  'ねこ', 'いぬ', 'とり', 'うま', 'さかな', 'むし', 'こ', 'わたし', 'あなた',
  'そら', 'うみ', 'やま', 'はな', 'つき', 'ほし', 'あめ', 'ゆき', 'かぜ', 'ひ', 'くさ', 'みず', 'いえ', 'にわ',
  'きょう', 'あした', 'はる', 'なつ', 'あき', 'ふゆ', 'にく', 'ごはん', 'おちゃ',
  'すき', 'たべる', 'のむ', 'はしる', 'ねる', 'なく', 'とぶ', 'およぐ', 'みる', 'いく', 'いる',
  'ふる', 'ふく', 'でる', 'はれる', 'さく', 'ひかる',
  'あおい', 'あかい', 'しろい', 'くろい', 'かわいい', 'ちいさい', 'おおきい', 'きれい', 'さむい', 'あつい',
]

export const ML_DATA: Record<Lang, MlData> = {
  ja: {
    lang: 'ja',
    classNames: { animal: 'いきもの', nature: 'しぜん', people: 'ひと' },
    sentences: CORPUS_JA,
    labels: LABELS_JA,
    dict: DICT_JA,
    stop: ['は', 'が', 'を', 'に', 'の', 'と', 'も'],
  },
  en: {
    lang: 'en',
    classNames: { animal: 'animals', nature: 'nature', people: 'people' },
    sentences: CORPUS_EN,
    labels: LABELS_EN,
    dict: [],
    stop: ['the', 'is', 'are', 'in', 'on', 'at', 'to', 'by', 'and'],
  },
}
