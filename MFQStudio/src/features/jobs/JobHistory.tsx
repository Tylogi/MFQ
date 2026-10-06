/** Display active jobs and terminal records for the quantization workspace. */
import { i18n } from '../../i18n';
import { useQuantization } from './QuantizationContext';
import { formatNumber } from '../../app/formatters';
/** Read the data and business actions required by this panel from page state. */
export function JobHistory() {
  const {
    t,
    selectedJobId,
    setSelectedJobId,
    jobCleanupBusy,
    activeJobs,
    completedJobs,
    clearCompletedJobRecords,
  } = useQuantization();
  return (
    <>
      <section className="dashboard-panel job-history" key="history">
        <div className="panel-heading">
          <div>
            <h2>{t('jobs:jobHistory.jobHistory')}</h2>
          </div>
          <b>{activeJobs.length}</b>
        </div>
        {activeJobs.length > 0 && (
          <div className="job-list">
            {activeJobs.map((job) => (
              <button
                className={job.id === selectedJobId ? 'active' : ''}
                key={job.id}
                onClick={() => setSelectedJobId(job.id)}
                type="button"
              >
                <span className={`job-status ${job.status}`} />
                <div>
                  <strong>{job.kind}</strong>
                  <small>
                    {job.status} · {new Date(job.updated_at).toLocaleString(i18n.resolvedLanguage)}
                  </small>
                </div>
                <b>{formatNumber(job.progress * 100)}%</b>
              </button>
            ))}
          </div>
        )}
        {completedJobs.length > 0 && (
          <details className="completed-jobs">
            <summary>
              <span>
                {t('jobs:jobHistory.completed')} <b>{completedJobs.length}</b>
              </span>
              <button
                disabled={jobCleanupBusy}
                onClick={(event) => {
                  event.preventDefault();
                  void clearCompletedJobRecords();
                }}
                type="button"
              >
                {t('jobs:jobHistory.clearCompleted')}
              </button>
            </summary>
            <div className="job-list">
              {completedJobs.map((job) => (
                <button
                  className={job.id === selectedJobId ? 'active' : ''}
                  key={job.id}
                  onClick={() => setSelectedJobId(job.id)}
                  type="button"
                >
                  <span className={`job-status ${job.status}`} />
                  <div>
                    <strong>{job.kind}</strong>
                    <small>
                      {job.status} · {new Date(job.updated_at).toLocaleString(i18n.resolvedLanguage)}
                    </small>
                  </div>
                  <b>{formatNumber(job.progress * 100)}%</b>
                </button>
              ))}
            </div>
          </details>
        )}
      </section>
    </>
  );
}
