import { PART_TITLES } from '../chapterMeta'
import type { ChapterDef } from '../chapters'
import { useLab } from '../state/LabProvider'
import { useMl } from '../state/MlProvider'
import { totalSize } from '../../engine/params'

interface Props {
  chapters: ChapterDef[]
  current: string
  onSelect: (id: string) => void
}

export function ChapterRail({ chapters, current, onSelect }: Props) {
  const lab = useLab()
  const ml = useMl()
  const cfg = lab.params.config
  const part = chapters.find((c) => c.id === current)?.part
  return (
    <nav className="rail" aria-label="章">
      <ol>
        {chapters.map((c, i) => (
          <li key={c.id}>
            {c.part !== chapters[i - 1]?.part && <div className="rail-part">{PART_TITLES[c.part]}</div>}
            <button aria-current={c.id === current} onClick={() => onSelect(c.id)}>
              <span className="num">{c.num}</span>
              <span>{c.title}</span>
            </button>
          </li>
        ))}
      </ol>
      {part === 'ml' ? (
        <div className="rail-foot">
          <div>分類器: ロジスティック回帰</div>
          <div>
            辞書 {ml.ds.vocab.length} 語 ・ {ml.ds.classes.length} クラス
          </div>
          <div>パラメータ {ml.model.paramCount.toLocaleString()}</div>
        </div>
      ) : (
        <div className="rail-foot">
          <div>
            モデル: {cfg.nLayers} 層 / {cfg.nHeads} ヘッド / d = {cfg.dModel}
          </div>
          <div>語彙 {cfg.vocabSize} ・ 文脈 {cfg.ctxLen}</div>
          <div>パラメータ {totalSize(lab.params.specs).toLocaleString()}</div>
        </div>
      )}
    </nav>
  )
}
