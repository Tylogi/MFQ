/** Display job results, actions, and event logs for the quantization workspace. */
import { i18n } from '../../i18n';
import { useQuantization } from './QuantizationContext';
import { formatNumber } from '../../app/formatters';
import { isTerminalJob } from './jobSchema';
/** Read the data and business actions required by this panel from page state. */
export function JobDetail() {
  const {
    t,
    busy,
    error,
    jobLogs,
    jobCleanupBusy,
    selectedJob,
    cancelSelectedJob,
    retrySelectedJob,
    deleteJobRecord,
    removeSelectedArtifact,
  } = useQuantization();
  return (
    <>
      {selectedJob && (
        <section className="dashboard-panel job-detail" key="detail">
          <div className="panel-heading">
            <div>
              <h2>{selectedJob.kind}</h2>
              <p>{selectedJob.id}</p>
            </div>
            <b>{selectedJob.status}</b>
          </div>
          <progress max={1} value={selectedJob.progress} />
          <div className="job-result-grid">
            <div>
              <span>{t('jobs:jobDetail.progress')}</span>
              <strong>{formatNumber(selectedJob.progress * 100)}%</strong>
            </div>
            <div>
              <span>{t('jobs:jobDetail.updated')}</span>
              <strong>{new Date(selectedJob.updated_at).toLocaleTimeString(i18n.resolvedLanguage)}</strong>
            </div>
          </div>
          <div className="job-actions">
            {['queued', 'running', 'cancelling'].includes(selectedJob.status) && (
              <button
                className="secondary"
                disabled={selectedJob.status === 'cancelling'}
                onClick={() => void cancelSelectedJob()}
                type="button"
              >
                {t('jobs:jobDetail.cancelJob')}
              </button>
            )}
            {['failed', 'cancelled', 'interrupted'].includes(selectedJob.status) && (
              <button
                className="secondary"
                disabled={busy}
                onClick={() => void retrySelectedJob()}
                type="button"
              >
                {t('common:retry')}
              </button>
            )}
            {isTerminalJob(selectedJob) && (
              <button
                className="secondary"
                disabled={jobCleanupBusy}
                onClick={() => void deleteJobRecord(selectedJob.id)}
                type="button"
              >
                {t('jobs:jobDetail.removeFromHistory')}
              </button>
            )}
            {String(selectedJob.result?.artifact || '').startsWith('workspace://') && (
              <button
                className="secondary danger"
                disabled={busy}
                onClick={() => void removeSelectedArtifact()}
                type="button"
              >
                {t('jobs:jobDetail.deleteLocalArtifact')}
              </button>
            )}
          </div>
          {selectedJob.error && <div className="job-error">{selectedJob.error.message}</div>}
          {selectedJob.result && <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>}
          <div className="job-log">
            <header>
              <span>{t('jobs:jobDetail.eventsAndLogs')}</span>
            </header>
            {jobLogs.map((entry) => (
              <div className={entry.level} key={entry.sequence}>
                <time>{new Date(entry.created_at).toLocaleTimeString(i18n.resolvedLanguage)}</time>
                <code>{entry.message}</code>
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
