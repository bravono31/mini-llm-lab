import type { ChapterDef } from '../chapters'
import { useLab } from '../state/LabProvider'
import { totalSize } from '../../engine/params'

interface Props {
  chapters: ChapterDef[]
  current: string
  onSelect: (id: string) => void
}

export function ChapterRail({ chapters, current, onSelect }: Props) {
  const lab = useLab()
  const cfg = lab.params.config
  return (
    <nav className="rail" aria-label="章">
      <ol>
        {chapters.map((c) => (
          <li key={c.id}>
            <button aria-current={c.id === current} onClick={() => onSelect(c.id)}>
              <span className="num">{c.num}</span>
              <span>{c.title}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="rail-foot">
        <div>
          モデル: {cfg.nLayers} 層 / {cfg.nHeads} ヘッド / d = {cfg.dModel}
        </div>
        <div>語彙 {cfg.vocabSize} ・ 文脈 {cfg.ctxLen}</div>
        <div>パラメータ {totalSize(lab.params.specs).toLocaleString()}</div>
      </div>
    </nav>
  )
}
