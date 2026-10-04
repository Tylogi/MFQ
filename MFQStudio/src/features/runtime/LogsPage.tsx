/** Poll requests and events on demand and manage job history cleanup on the logs page. */
import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { RuntimeLogEntry, RuntimeRequestMetrics } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader, SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { isTerminalJob } from '../jobs/jobSchema';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';

/** Read metric history only while the logs page is open; schedule each poll after the previous request completes to avoid overlap. */
export function LogsPage() {
  const jobs = useJobStore((state) => state.jobs);
  const { ready, refreshRuntime } = useRuntime();
  const { tr } = useSettings();
  const [runtimeLogs, setRuntimeLogs] = useState<RuntimeLogEntry[]>([]);
  const [requestHistory, setRequestHistory] = useState<RuntimeRequestMetrics[]>([]);
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);
  const activeJobs = jobs.filter((job) => ['queued', 'running', 'cancelling'].includes(job.status));
  const completedJobs = jobs.filter(isTerminalJob);
  useEffect(() => {
    if (!ready) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    /** Read logs and metrics, stopping polling and ignoring late results on unmount. */
    async function refresh() {
      try {
        const [logs, metrics] = await Promise.all([runtimeApi.runtimeLogs(100), runtimeApi.runtimeMetrics(200)]);
        if (!active) return;
        setRuntimeLogs(logs);
        const unique = new Map<string, RuntimeRequestMetrics>();
        for (const snapshot of metrics) {
          const request = snapshot.values.last_request;
          if (request?.id) unique.set(request.id, request);
        }
        setRequestHistory([...unique.values()].slice(-24).reverse());
      } catch {
        // Skip polling failures silently and retry on the next cycle.
      } finally {
        if (active) timer = setTimeout(() => void refresh(), 4000);
      }
    }
    void refresh();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [ready]);
  /** Delete one or all completed job records, then reload the shared job list. */
  async function cleanup(id?: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      if (id) await jobsApi.deleteJob(id);
      else await jobsApi.clearCompletedJobs();
      await refreshRuntime(false);
      toast.success(tr('任务记录已清理', 'Job records cleaned up'));
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
        title={tr('日志', 'Logs')}
        subtitle={tr('请求、任务与运行事件。', 'Requests, jobs, and runtime events.')}
      />

      <SectionLabel
        title={tr('Runtime 活动', 'Runtime activity')}
        subtitle={tr('请求、任务与事件', 'Requests, jobs, and events')}
      />
      <div className="dashboard-grid logs-grid">
        <TMPanel>
          <div className="panel-heading">
            <div>
              <h2>{tr('最近请求', 'Recent requests')}</h2>
            </div>
          </div>
          {requestHistory.length > 0 ? (
            <div className="request-table">
              {requestHistory.slice(0, 8).map((request) => (
                <div className="request-row" key={request.id}>
                  <div>
                    <strong>{request.id}</strong>
                    <small>
                      {request.completed_at
                        ? new Date(request.completed_at * 1000).toLocaleTimeString()
                        : request.endpoint || 'completion'}
                    </small>
                  </div>
                  <span>
                    {formatNumber(request.prompt_tokens)} →{' '}
                    {formatNumber(request.completion_tokens)}
                  </span>
                  <b>{formatNumber(request.decode_tps, 1)} tok/s</b>
                </div>
              ))}
            </div>
          ) : (
            <div className="inline-empty">
              {tr('还没有完成的请求。', 'No completed requests yet.')}
            </div>
          )}
        </TMPanel>
        <TMPanel>
          <div className="panel-heading">
            <div>
              <h2>{tr('后台任务', 'Background jobs')}</h2>
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
                      {new Date(job.updated_at).toLocaleTimeString()} · {job.status}
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
                  {tr('已完成', 'Completed')} <b>{completedJobs.length}</b>
                </span>
                <button
                  disabled={jobCleanupBusy}
                  onClick={(event) => {
                    event.preventDefault();
                    void clearCompletedJobRecords();
                  }}
                  type="button"
                >
                  {tr('清理已完成', 'Clear completed')}
                </button>
              </summary>
              <div className="request-table">
                {completedJobs.slice(0, 8).map((job) => (
                  <div className="job-row completed" key={job.id}>
                    <div>
                      <strong>{job.kind}</strong>
                      <small>
                        {new Date(job.updated_at).toLocaleTimeString()} · {job.status}
                      </small>
                    </div>
                    <progress max={1} value={job.progress} />
                    <b>{formatNumber(job.progress * 100)}%</b>
                    <button
                      aria-label={tr('移出任务历史', 'Remove from job history')}
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
              <h2>{tr('Runtime 日志', 'Runtime logs')}</h2>
            </div>
            <b>{runtimeLogs.length}</b>
          </div>
          {runtimeLogs.length > 0 ? (
            <div className="runtime-log-list">
              {runtimeLogs
                .slice(-8)
                .reverse()
                .map((entry) => (
                  <div className={`runtime-log ${entry.level}`} key={entry.sequence}>
                    <span>{new Date(entry.created_at).toLocaleTimeString()}</span>
                    <p>{entry.message}</p>
                  </div>
                ))}
            </div>
          ) : (
            <div className="inline-empty">
              {tr('暂无 Runtime 事件。', 'No runtime events yet.')}
            </div>
          )}
        </TMPanel>
      </div>
    </section>
  );
}
