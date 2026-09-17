import { useLab } from '../state/LabProvider'
import { useTheme, type ThemeChoice } from '../state/ThemeProvider'

export function TopBar() {
  const lab = useLab()
  const { theme, setTheme } = useTheme()
  const unknown = lab.trace.unknownChars
  const note = unknown.length
    ? `「${unknown.join('」「')}」はこのモデルの語彙にありません（⟨unk⟩ 扱い）`
    : lab.truncated
      ? `先頭 ${lab.params.config.ctxLen} トークンだけを使います（文脈長の上限）`
      : ''
  return (
    <header className="topbar">
      <div className="brand">
        <h1>Mini LLM Lab</h1>
        <small>LLM の中身を、数値のまま見る</small>
      </div>
      <div className="topbar-input">
        <label htmlFor="input-text">入力文</label>
        <input
          id="input-text"
          className={'text-input' + (unknown.length ? ' has-unknown' : '')}
          value={lab.input}
          onChange={(e) => lab.setInput(e.target.value)}
          placeholder={lab.corpus.defaultInput}
          spellCheck={false}
          autoComplete="off"
        />
        {note && <span className="input-note">{note}</span>}
      </div>
      <div className="segmented" role="group" aria-label="コーパス">
        <button aria-pressed={lab.lang === 'ja'} onClick={() => lab.setLang('ja')}>
          日本語
        </button>
        <button aria-pressed={lab.lang === 'en'} onClick={() => lab.setLang('en')}>
          English
        </button>
      </div>
      <div className="segmented" role="group" aria-label="テーマ">
        {(['auto', 'light', 'dark'] as ThemeChoice[]).map((t) => (
          <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>
            {t === 'auto' ? '自動' : t === 'light' ? 'ライト' : 'ダーク'}
          </button>
        ))}
      </div>
    </header>
  )
}
