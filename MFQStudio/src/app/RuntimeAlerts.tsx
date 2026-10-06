/** Connection, refresh, and job-stream alerts in the application shell. */
import { useTranslation } from 'react-i18next';
import { useRuntime } from './RuntimeProvider';
import { Icon } from './display';
/** Display current runtime failures with independent retry actions. */
export function RuntimeAlerts({ connectionProblem }: { connectionProblem: string | null }) {
  const { t } = useTranslation();
  const { ready, refreshError, jobStreamErrors, loading: selectedModelLoading,
    reloadService, refreshRuntime, retryJobStreams } = useRuntime();
  const streamFailures = Object.entries(jobStreamErrors);
  return (
    <div className="runtime-alerts">
      {connectionProblem && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{t('app:runtimeAlerts.unableToConnectToTheService')}</strong>
            <span>{connectionProblem}</span>
          </div>
          <button disabled={selectedModelLoading} onClick={() => void reloadService()} type="button">
            <Icon name="refresh" size={14} />{t('app:runtimeAlerts.retryConnection')}
          </button>
        </div>
      )}
      {ready && refreshError && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{t('app:runtimeAlerts.runtimeStatusCouldNotBeUpdatedDataMayBeStale')}</strong>
            <span>{refreshError}</span>
          </div>
          <button disabled={selectedModelLoading}
            onClick={() => void refreshRuntime(false)} type="button">
            <Icon name="refresh" size={14} />{t('app:runtimeAlerts.refreshAgain')}
          </button>
        </div>
      )}
      {ready && streamFailures.length > 0 && (
        <div className="runtime-alert" role="alert">
          <div>
            <strong>{t('app:runtimeAlerts.taskProgressIsNotUpdating')}</strong>
            <span>{streamFailures.map(([id, message]) => `${id}: ${message}`).join('; ')}</span>
          </div>
          <button onClick={retryJobStreams} type="button">
            <Icon name="refresh" size={14} />{t('app:runtimeAlerts.reconnectTaskStreams')}
          </button>
        </div>
      )}
    </div>
  );
}
