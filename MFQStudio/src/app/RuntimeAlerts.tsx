/** 应用外壳中的连接、刷新与任务流告警。 */
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import { Icon } from './display';

/** 展示当前运行时异常及各自独立的重试操作。 */
export function RuntimeAlerts({ connectionProblem }: { connectionProblem: string | null }) {
  const { tr } = useSettings();
  const { ready, refreshError, jobStreamErrors, loading: selectedModelLoading,
    reloadService, refreshRuntime, retryJobStreams } = useRuntime();
  const streamFailures = Object.entries(jobStreamErrors);
  return (
    <div className="runtime-alerts">
      {connectionProblem && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{tr('无法连接服务', 'Unable to connect to the service')}</strong>
            <span>{connectionProblem}</span>
          </div>
          <button disabled={selectedModelLoading} onClick={() => void reloadService()} type="button">
            <Icon name="refresh" size={14} />{tr('重试连接', 'Retry connection')}
          </button>
        </div>
      )}
      {ready && refreshError && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{tr('运行状态更新失败，数据可能不是最新的',
              'Runtime status could not be updated; data may be stale')}</strong>
            <span>{refreshError}</span>
          </div>
          <button disabled={selectedModelLoading}
            onClick={() => void refreshRuntime(false)} type="button">
            <Icon name="refresh" size={14} />{tr('重新刷新', 'Refresh again')}
          </button>
        </div>
      )}
      {ready && streamFailures.length > 0 && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{tr('任务进度暂时无法更新', 'Task progress is not updating')}</strong>
            <span>{streamFailures.map(([id, message]) => `${id}: ${message}`).join('; ')}</span>
          </div>
          <button onClick={retryJobStreams} type="button">
            <Icon name="refresh" size={14} />{tr('重新连接任务流', 'Reconnect task streams')}
          </button>
        </div>
      )}
    </div>
  );
}
