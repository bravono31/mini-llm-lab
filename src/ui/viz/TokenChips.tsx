import { displayToken, EOS, UNK } from '../../engine/tokenizer'

interface Props {
  tokens: string[]
  ids?: ArrayLike<number>
  active?: number | null
  onSelect?: (i: number) => void
  newSet?: Set<number>
  dimSet?: Set<number>
  /** show position index above each chip */
  positions?: boolean
}

export function TokenChips({ tokens, ids, active, onSelect, newSet, dimSet, positions }: Props) {
  return (
    <div className="chips">
      {tokens.map((t, i) => {
        const cls = [
          'chip',
          onSelect ? 'clickable' : '',
          active === i ? 'active' : '',
          newSet?.has(i) ? 'new' : '',
          dimSet?.has(i) ? 'dim' : '',
          t === EOS || t === UNK ? 'special' : '',
        ]
          .filter(Boolean)
          .join(' ')
        return (
          <span key={i} className={cls} onClick={() => onSelect?.(i)} role={onSelect ? 'button' : undefined}>
            {positions && <span className="id">t={i}</span>}
            <span>{displayToken(t)}</span>
            {ids && <span className="id">id {ids[i]}</span>}
          </span>
        )
      })}
    </div>
  )
}
