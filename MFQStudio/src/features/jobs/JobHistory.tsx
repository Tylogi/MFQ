/** 展示量化工作台的活动任务与终态记录。 */
import { useQuantization } from './QuantizationContext';
import { formatNumber } from '../../app/formatters';
/** 从页面状态读取本面板所需数据与业务操作。 */
export function JobHistory() {
  const {
    tr,
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
            <h2>{tr('任务历史', 'Job history')}</h2>
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
                    {job.status} · {new Date(job.updated_at).toLocaleString()}
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
                      {job.status} · {new Date(job.updated_at).toLocaleString()}
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
