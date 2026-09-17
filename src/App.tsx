import { CHAPTERS } from './ui/chapters'
import { ChapterRail } from './ui/shell/ChapterRail'
import { TopBar } from './ui/shell/TopBar'
import { LabProvider } from './ui/state/LabProvider'
import { NavProvider, useNav } from './ui/state/NavProvider'
import { ThemeProvider } from './ui/state/ThemeProvider'

function Body() {
  const nav = useNav()
  const chapter = CHAPTERS.find((c) => c.id === nav.current) ?? CHAPTERS[0]
  const Comp = chapter.component
  return (
    <div className="app">
      <TopBar />
      <div className="body">
        <ChapterRail chapters={CHAPTERS} current={chapter.id} onSelect={nav.go} />
        <Comp key={chapter.id} onNavigate={nav.go} />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <NavProvider>
        <LabProvider>
          <Body />
        </LabProvider>
      </NavProvider>
    </ThemeProvider>
  )
}
