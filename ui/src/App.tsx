import { Link, Navigate, Route, Routes, useParams } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { PageHeader } from "./components/States";
import { EvalsScreen } from "./screens/evals/EvalsScreen";
import { RunDetailScreen } from "./screens/evals/RunDetailScreen";
import { PlaygroundScreen } from "./screens/playground/PlaygroundScreen";
import { PresetEditorScreen } from "./screens/presets/PresetEditorScreen";
import { PresetsScreen } from "./screens/presets/PresetsScreen";
import { ServerSettingsScreen } from "./screens/server/ServerSettingsScreen";

export function App() {
    return (
        <Routes>
            <Route element={<AppShell />}>
                <Route index element={<Navigate to="/presets" replace />} />
                <Route path="/server" element={<ServerSettingsScreen />} />
                <Route
                    path="/rag"
                    element={<Navigate to="/server" replace />}
                />
                <Route path="/presets" element={<PresetsScreen />} />
                {/* `new` before `:name`, so it is never read as a preset called "new"
            (the preset id pattern would happily allow that name). */}
                <Route
                    path="/presets/new"
                    element={<PresetEditorScreen key="new" />}
                />
                <Route path="/presets/:name" element={<PresetEditorRoute />} />
                <Route path="/evals" element={<EvalsScreen />} />
                <Route
                    path="/evals/runs/:runId"
                    element={<RunDetailScreen />}
                />
                <Route path="/playground" element={<PlaygroundScreen />} />
                <Route
                    path="/reference"
                    element={
                        <Navigate to="/server?section=resources" replace />
                    }
                />
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    );
}

function PresetEditorRoute() {
    const { name } = useParams();
    return <PresetEditorScreen key={name} />;
}

function NotFound() {
    return (
        <PageHeader
            title="找不到此頁面"
            lede={
                <>
                    此網址不屬於本管理介面。
                    <Link to="/presets">前往行為預設</Link> is the front door.
                </>
            }
        />
    );
}
