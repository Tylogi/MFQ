/** 展示量化工作台的任务结果、操作和事件日志。 */
import { useQuantization } from './QuantizationContext';
import { formatNumber } from '../../app/formatters';
import { isTerminalJob } from './jobSchema';
/** 从页面状态读取本面板所需数据与业务操作。 */
export function JobDetail() {
  const {
    tr,
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
              <span>{tr('进度', 'Progress')}</span>
              <strong>{formatNumber(selectedJob.progress * 100)}%</strong>
            </div>
            <div>
              <span>{tr('更新时间', 'Updated')}</span>
              <strong>{new Date(selectedJob.updated_at).toLocaleTimeString()}</strong>
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
                {tr('取消任务', 'Cancel job')}
              </button>
            )}
            {['failed', 'cancelled', 'interrupted'].includes(selectedJob.status) && (
              <button
                className="secondary"
                disabled={busy}
                onClick={() => void retrySelectedJob()}
                type="button"
              >
                {tr('重试', 'Retry')}
              </button>
            )}
            {isTerminalJob(selectedJob) && (
              <button
                className="secondary"
                disabled={jobCleanupBusy}
                onClick={() => void deleteJobRecord(selectedJob.id)}
                type="button"
              >
                {tr('移出任务历史', 'Remove from history')}
              </button>
            )}
            {String(selectedJob.result?.artifact || '').startsWith('workspace://') && (
              <button
                className="secondary danger"
                disabled={busy}
                onClick={() => void removeSelectedArtifact()}
                type="button"
              >
                {tr('删除本地产物', 'Delete local artifact')}
              </button>
            )}
          </div>
          {selectedJob.error && <div className="job-error">{selectedJob.error.message}</div>}
          {selectedJob.result && <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>}
          <div className="job-log">
            <header>
              <span>{tr('事件与日志', 'Events and logs')}</span>
            </header>
            {jobLogs.map((entry) => (
              <div className={entry.level} key={entry.sequence}>
                <time>{new Date(entry.created_at).toLocaleTimeString()}</time>
                <code>{entry.message}</code>
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
