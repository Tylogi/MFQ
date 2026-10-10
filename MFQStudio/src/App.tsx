/** 组合应用级 Provider 与业务路由，页面状态和请求由对应领域拥有。 */
import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router';
import { SettingsProvider } from './features/settings/SettingsProvider';
import { StudioUpdateProvider } from './features/settings/UpdateManager';
import { RuntimeProvider } from './app/RuntimeProvider';
import { ChatProvider } from './features/chat/ChatProvider';
import { StudioShell } from './app/StudioShell';
import { LabLayout } from './app/LabLayout';
import { LoadingPage } from './app/LoadingPage';
import { NotFoundPage } from './app/NotFoundPage';
import './shared/ui/primitives.css';

const ChatPage = lazy(() =>
  import('./features/chat/ChatPage').then((module) => ({ default: module.ChatPage })),
);
const OverviewPage = lazy(() =>
  import('./features/runtime/OverviewPage').then((module) => ({ default: module.OverviewPage })),
);
const ModelsPage = lazy(() =>
  import('./features/models/ModelsPage').then((module) => ({ default: module.ModelsPage })),
);
const ConnectionsPage = lazy(() =>
  import('./features/connections/ConnectionsPage').then((module) => ({
    default: module.ConnectionsPage,
  })),
);
const CachePage = lazy(() =>
  import('./features/runtime/CachePage').then((module) => ({ default: module.CachePage })),
);
const LogsPage = lazy(() =>
  import('./features/runtime/LogsPage').then((module) => ({ default: module.LogsPage })),
);
const SettingsRoute = lazy(() =>
  import('./features/settings/SettingsRoute').then((module) => ({ default: module.SettingsRoute })),
);
const ModelHubPage = lazy(() =>
  import('./features/models/ModelHubPage').then((module) => ({ default: module.ModelHubPage })),
);
const EvaluationsPage = lazy(() =>
  import('./features/evaluations/EvaluationsPage').then((module) => ({
    default: module.EvaluationsPage,
  })),
);
const QuantizationPage = lazy(() =>
  import('./features/jobs/QuantizationPage').then((module) => ({
    default: module.QuantizationPage,
  })),
);
const AnalysisPage = lazy(() =>
  import('./features/analysis/AnalysisPage').then((module) => ({ default: module.AnalysisPage })),
);
const ApplicationsPage = lazy(() =>
  import('./features/applications/ApplicationsPage').then((module) => ({ default: module.ApplicationsPage })),
);
const VersionsPage = lazy(() =>
  import('./features/settings/VersionsPage').then(module => ({ default: module.VersionsPage })),
);

/** 挂载共享服务与独立业务页面，未知地址显示 404。 */
export default function App() {
  return (
    <SettingsProvider>
      <StudioUpdateProvider>
        <RuntimeProvider>
          <ChatProvider>
            <Suspense fallback={<main className="fatal-workspace"><LoadingPage /></main>}>
              <Routes>
                <Route element={<StudioShell />}>
                  <Route index element={<OverviewPage />} />
                  <Route path="chat" element={<ChatPage />} />
                  <Route path="applications" element={<ApplicationsPage />} />
                  <Route path="models" element={<ModelsPage />} />
                  <Route path="runtime" element={<ConnectionsPage />} />
                  <Route path="resources" element={<CachePage />} />
                  <Route path="logs" element={<LogsPage />} />
                  <Route path="settings" element={<SettingsRoute />} />
                  <Route path="versions" element={<VersionsPage />} />
                  <Route element={<LabLayout />}>
                    <Route path="analysis" element={<AnalysisPage />} />
                    <Route path="model-hub" element={<ModelHubPage />} />
                    <Route path="evaluations" element={<EvaluationsPage />} />
                    <Route path="quantization" element={<QuantizationPage />} />
                  </Route>
                  <Route path="*" element={<NotFoundPage />} />
                </Route>
              </Routes>
            </Suspense>
          </ChatProvider>
        </RuntimeProvider>
      </StudioUpdateProvider>
    </SettingsProvider>
  );
}
