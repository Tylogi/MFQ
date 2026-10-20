import { useNavigate } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';
import { useRuntime } from './RuntimeProvider';

export function ServerConnectionPage() {
  const { tr } = useSettings();
  const navigate = useNavigate();
  const { studio, loading, connectionError, connectExistingServer, reloadService } = useRuntime();
  const incompatible = studio?.connection === 'incompatible';
  return (
    <section className="failure-page" aria-labelledby="server-connection-title">
      <div className="failure-page-content">
        <h1 id="server-connection-title">{incompatible
          ? tr('已有服务与当前 App 不兼容', 'Existing server is incompatible with this app')
          : tr('检测到独立运行的服务', 'An independently managed server is running')}</h1>
        <p className="failure-page-description">{incompatible
          ? tr('请更新或停止已有服务，再由 App 启动配套后端。不会自动结束已有进程。', 'Update or stop the existing server, then let the app start its bundled backend. Existing processes will not be stopped automatically.')
          : tr('连接后可在 App 中管理此服务；退出 App 不会停止它，也不会额外启动后端。', 'Connect to manage this server in the app. Quitting the app will leave it running; no additional backend will be started.')}</p>
        <p>{studio?.service_url}{studio?.server && ` · MFQ Server ${studio.server.version} · PID ${studio.server.pid}`}</p>
        <div className="failure-page-actions">
          {!incompatible && <button className="failure-page-primary" disabled={loading}
            onClick={() => void connectExistingServer()} type="button">{tr('连接已有服务', 'Connect to existing server')}</button>}
          <button disabled={loading} onClick={() => void reloadService()} type="button">{tr('重新检测', 'Check again')}</button>
          <button onClick={() => navigate('/runtime')} type="button">{tr('连接设置', 'Connection settings')}</button>
        </div>
        {connectionError && <p role="alert">{connectionError}</p>}
      </div>
    </section>
  );
}
