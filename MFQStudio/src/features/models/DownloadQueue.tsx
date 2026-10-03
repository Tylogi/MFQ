import { useState } from 'react';
import type { JobResource } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { Icon } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import type { DownloadOrigin } from './ModelBrowser';

export const isModelDownload = (job: JobResource) => job.kind === 'download.huggingface' || job.kind === 'download.modelscope';
export const isActiveDownload = (job: JobResource) => ['queued', 'running', 'cancelling'].includes(job.status);
export const downloadProgress = (job: JobResource) => Number.isFinite(job.progress) ? Math.max(0, Math.min(1, job.progress)) : 0;

export function DownloadQueue({ jobs, onJobCreated }: {
  jobs: JobResource[];
  onJobCreated(job: JobResource, origin: DownloadOrigin): void;
}) {
  const { tr } = useSettings();
  const addJob = useJobStore((state) => state.addJob);
  const [busyId, setBusyId] = useState<string | null>(null);
  const statusLabels: Record<JobResource['status'], string> = {
    queued: tr('等待下载', 'Queued'), running: tr('下载中', 'Downloading'),
    cancelling: tr('正在取消', 'Cancelling'), succeeded: tr('下载完成', 'Completed'),
    failed: tr('下载失败', 'Failed'), cancelled: tr('已取消', 'Cancelled'),
    interrupted: tr('下载中断', 'Interrupted'),
  };

  async function act(job: JobResource, retry: boolean, origin: DownloadOrigin) {
    if (busyId) return;
    setBusyId(job.id);
    try {
      if (retry) onJobCreated(await jobsApi.retryJob(job.id), origin);
      else addJob(await jobsApi.cancelJob(job.id));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section className="download-queue" aria-label={tr('下载队列', 'Download queue')}>
      <div className="download-queue-heading">
        <h3>{tr('下载队列', 'Download queue')}</h3>
        <span>{tr(`${jobs.filter(isActiveDownload).length} 个下载中`, `${jobs.filter(isActiveDownload).length} active`)}</span>
      </div>
      {!jobs.length && <div className="download-queue-empty"><Icon name="download" size={28} />
        <strong>{tr('暂无下载', 'No downloads yet')}</strong>
      </div>}
      {jobs.map((job) => {
        const progress = downloadProgress(job);
        const retry = ['failed', 'cancelled', 'interrupted'].includes(job.status);
        return (
          <article className={`download-queue-item ${job.status}`} key={job.id} data-job-id={job.id}>
            <div className="download-item-icon"><Icon name={job.status === 'succeeded' ? 'check' : 'download'} size={20} /></div>
            <div className="download-item-body">
              <div className="download-item-heading"><strong>{String(job.payload.repo_id || job.kind)}</strong><span>{statusLabels[job.status]}</span></div>
              <small>{job.kind === 'download.huggingface' ? 'Hugging Face' : 'ModelScope'} · {String(job.payload.destination || '')}</small>
              <div className="download-item-progress"><progress aria-label={tr('下载进度', 'Download progress')} max={1} value={progress} /><span>{formatNumber(progress * 100, 1)}%</span></div>
              {job.error && <p className="download-item-error">{job.error.message}</p>}
            </div>
            {(isActiveDownload(job) || retry) && <button disabled={busyId !== null || job.status === 'cancelling'} onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              void act(job, retry, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
            }} type="button">{retry ? tr('重试', 'Retry') : tr('取消', 'Cancel')}</button>}
          </article>
        );
      })}
    </section>
  );
}
