export type ValueKind = 'param' | 'computed' | 'design'

const LABEL: Record<ValueKind, string> = { param: '学習で決まる', computed: '入力から計算', design: '設計で固定' }
const TITLE: Record<ValueKind, string> = {
  param: 'パラメータ。初期値は乱数で、学習ステップごとに値が変わる',
  computed: '入力文とパラメータから一意に計算される中間結果。入力を変えると変わる',
  design: 'モデルを設計したときに人が決めた定数。学習でも入力でも変わらない',
}

/** Small pill that says where a number comes from. */
export function Tag({ kind }: { kind: ValueKind }) {
  return (
    <span className={`tag tag-${kind}`} title={TITLE[kind]}>
      {LABEL[kind]}
    </span>
  )
}
