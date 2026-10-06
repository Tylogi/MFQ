/** Stream requests and events on demand and manage job history cleanup on the logs page. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { Icon, ScreenHeader, SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatDateTime, formatNumber } from '../../app/formatters';
import { isTerminalJob } from '../jobs/jobSchema';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';
import { useRuntimeHistory } from './useRuntimeHistory';
import { RuntimeHistoryList } from './RuntimeHistoryList';
import type { RuntimeMetricSnapshot } from '../../shared/api/types';

/** Use the durable row identity so different instances may reuse backend request IDs. */
function requestKey(entry: RuntimeMetricSnapshot): string {
  return String(entry.sequence);
}

/** Read initial history and subscribe to both SSE channels while the logs page is open. */
export function LogsPage() {
  const jobs = useJobStore((state) => state.jobs);
  const { ready, connectionRevision, refreshRuntime } = useRuntime();
  const { t } = useTranslation();
  const logHistory = useRuntimeHistory(ready, connectionRevision, runtimeApi.runtimeLogPage, runtimeApi.streamRuntimeLogs);
  const requestHistory = useRuntimeHistory(
    ready,
    connectionRevision,
    runtimeApi.runtimeRequestPage,
    runtimeApi.streamRuntimeRequests,
    requestKey,
  );
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);
  const activeJobs = jobs.filter((job) => ['queued', 'running', 'cancelling'].includes(job.status));
  const completedJobs = jobs.filter(isTerminalJob);
  /** Delete one or all completed job records, then reload the shared job list. */
  async function cleanup(id?: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      if (id) await jobsApi.deleteJob(id);
      else await jobsApi.clearCompletedJobs();
      await refreshRuntime(false);
      toast.success(localized('runtime:logsPage.jobRecordsCleanedUp'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }
  const deleteJobRecord = cleanup;
  const clearCompletedJobRecords = () => cleanup();
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={t('runtime:logsPage.logs')}
        subtitle={t('runtime:logsPage.requestsJobsAndRuntimeEvents')}
      />

      <SectionLabel
        title={t('runtime:logsPage.runtimeActivity')}
        subtitle={t('runtime:logsPage.requestsJobsAndEvents')}
      />
      <div className="dashboard-grid logs-grid">
        <TMPanel>
          <div className="panel-heading">
            <div>
              <h2>{t('runtime:logsPage.recentRequests')}</h2>
            </div>
            <b>{requestHistory.items.length}</b>
          </div>
          <RuntimeHistoryList
            history={requestHistory}
            label={t('runtime:logsPage.recentRequests')}
            empty={t('runtime:logsPage.noRequestsRecordedYet')}
            t={t}
            renderEntry={(snapshot) => {
              const request = snapshot.values.last_request;
              if (!request) return null;
              return (
                <div className="request-row">
                  <div>
                    <strong>{request.id}</strong>
                    <small>
                      {request.status === 'failed' ? t('runtime:logsPage.failed')
                        : request.status === 'cancelled' ? t('runtime:logsPage.cancelled') : ''}
                      {request.completed_at
                        ? formatDateTime(request.completed_at * 1000)
                        : request.endpoint || 'completion'}
                    </small>
                  </div>
                  <span>
                    {formatNumber(request.prompt_tokens)} →{' '}
                    {formatNumber(request.completion_tokens)}
                  </span>
                  <b>{formatNumber(request.decode_tps, 1)} tok/s</b>
                </div>
              );
            }}
          />
        </TMPanel>
        <TMPanel>
          <div className="panel-heading">
            <div>
              <h2>{t('runtime:logsPage.backgroundJobs')}</h2>
            </div>
            <b>{activeJobs.length}</b>
          </div>
          {activeJobs.length > 0 && (
            <div className="request-table">
              {activeJobs.slice(0, 8).map((job) => (
                <div className="job-row" key={job.id}>
                  <div>
                    <strong>{job.kind}</strong>
                    <small>
                      {formatDateTime(job.updated_at)} · {job.status}
                    </small>
                  </div>
                  <progress max={1} value={job.progress} />
                  <b>{formatNumber(job.progress * 100)}%</b>
                </div>
              ))}
            </div>
          )}
          {completedJobs.length > 0 && (
            <details className="completed-jobs">
              <summary>
                <span>
                  {t('runtime:logsPage.completed')} <b>{completedJobs.length}</b>
                </span>
                <button
                  disabled={jobCleanupBusy}
                  onClick={(event) => {
                    event.preventDefault();
                    void clearCompletedJobRecords();
                  }}
                  type="button"
                >
                  {t('runtime:logsPage.clearCompleted')}
                </button>
              </summary>
              <div className="request-table">
                {completedJobs.slice(0, 8).map((job) => (
                  <div className="job-row completed" key={job.id}>
                    <div>
                      <strong>{job.kind}</strong>
                      <small>
                        {formatDateTime(job.updated_at)} · {job.status}
                      </small>
                    </div>
                    <progress max={1} value={job.progress} />
                    <b>{formatNumber(job.progress * 100)}%</b>
                    <button
                      aria-label={t('runtime:logsPage.removeFromJobHistory')}
                      disabled={jobCleanupBusy}
                      onClick={() => void deleteJobRecord(job.id)}
                      type="button"
                    >
                      <Icon name="trash" size={12} />
                    </button>
                  </div>
                ))}
              </div>
            </details>
          )}
        </TMPanel>
        <TMPanel className="runtime-log-panel">
          <div className="panel-heading">
            <div>
              <h2>{t('runtime:logsPage.runtimeLogs')}</h2>
            </div>
            <b>{logHistory.items.length}</b>
          </div>
          <RuntimeHistoryList
            history={logHistory}
            label={t('runtime:logsPage.runtimeLogs')}
            empty={t('runtime:logsPage.noRuntimeEventsYet')}
            t={t}
            renderEntry={(entry) => (
              <div className={`runtime-log ${entry.level}`}>
                <time dateTime={entry.created_at}>
                  {formatDateTime(entry.created_at)}
                </time>
                <p>{entry.message}</p>
              </div>
            )}
          />
        </TMPanel>
      </div>
    </section>
  );
}
