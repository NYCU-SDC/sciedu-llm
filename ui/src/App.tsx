import { Link, Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/AppShell'
import { PageHeader } from './components/States'
import { EvalsScreen } from './screens/evals/EvalsScreen'
import { RunDetailScreen } from './screens/evals/RunDetailScreen'
import { PlaygroundScreen } from './screens/playground/PlaygroundScreen'
import { PresetEditorScreen } from './screens/presets/PresetEditorScreen'
import { PresetsScreen } from './screens/presets/PresetsScreen'
import { RagScreen } from './screens/rag/RagScreen'
import { ReferenceScreen } from './screens/reference/ReferenceScreen'

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/rag" replace />} />
        <Route path="/rag" element={<RagScreen />} />
        <Route path="/presets" element={<PresetsScreen />} />
        {/* `new` before `:name`, so it is never read as a preset called "new"
            (the preset id pattern would happily allow that name). */}
        <Route path="/presets/new" element={<PresetEditorScreen key="new" />} />
        <Route path="/presets/:name" element={<PresetEditorScreen />} />
        <Route path="/evals" element={<EvalsScreen />} />
        <Route path="/evals/runs/:runId" element={<RunDetailScreen />} />
        <Route path="/playground" element={<PlaygroundScreen />} />
        <Route path="/reference" element={<ReferenceScreen />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

function NotFound() {
  return (
    <PageHeader
      title="No such page"
      lede={
        <>
          That address is not part of this console. <Link to="/rag">Retrieval settings</Link>{' '}
          is the front door.
        </>
      }
    />
  )
}
