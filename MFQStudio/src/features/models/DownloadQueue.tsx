/** Provide DownloadQueue interface behavior. */
import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import type { JobResource } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { Icon } from '../../app/display';
import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import type { DownloadOrigin } from './ModelBrowser';
import { ModelVendorMark } from '../../app/ModelVendorMark';

export const isModelDownload = (job: JobResource) => job.kind === 'download.huggingface' || job.kind === 'download.modelscope';
export const isActiveDownload = (job: JobResource) => ['queued', 'running', 'cancelling'].includes(job.status);
export const downloadProgress = (job: JobResource) => Number.isFinite(job.progress) ? Math.max(0, Math.min(1, job.progress)) : 0;

/** Display download progress and expose retry and cancellation actions. */
export function DownloadQueue({ jobs, onJobCreated }: {
  jobs: JobResource[];
  onJobCreated(job: JobResource, origin: DownloadOrigin): void;
}) {
  const { t } = useTranslation();
  const addJob = useJobStore((state) => state.addJob);
  const [busyId, setBusyId] = useState<string | null>(null);
  const statusLabels: Record<JobResource['status'], string> = {
    queued: t('models:downloadQueue.queued'), running: t('models:downloadQueue.downloading'),
    cancelling: t('models:downloadQueue.cancelling'), succeeded: t('models:downloadQueue.completed'),
    failed: t('models:downloadQueue.failed'), cancelled: t('models:downloadQueue.cancelled'),
    interrupted: t('models:downloadQueue.interrupted'),
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

  async function remove(job: JobResource) {
    setBusyId(job.id);
    try {
      await jobsApi.deleteJob(job.id);
      useJobStore.getState().setJobs(useJobStore.getState().jobs.filter((item) => item.id !== job.id));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setBusyId(null); }
  }

  return (
    <section className="download-queue" aria-label={t('models:downloadQueue.downloadQueue')}>
      <div className="download-queue-heading">
        <h3>{t('models:downloadQueue.downloadQueue')}</h3>
        <span>{t('models:downloadQueue.active', { count: jobs.filter(isActiveDownload).length })}</span>
      </div>
      {!jobs.length && <div className="download-queue-empty"><Icon name="download" size={28} />
        <strong>{t('models:downloadQueue.noDownloadsYet')}</strong>
      </div>}
      {jobs.map((job) => {
        const progress = downloadProgress(job);
        const retry = ['failed', 'cancelled', 'interrupted'].includes(job.status);
        return (
          <article className={`download-queue-item ${job.status}`} key={job.id} data-job-id={job.id}>
            <div className="download-item-icon"><Icon name={job.status === 'succeeded' ? 'check' : 'download'} size={20} /></div>
            <div className="download-item-body">
              <div className="download-item-heading"><strong>{String(job.payload.repo_id || job.kind)}</strong><span className="model-identity-trailing">{statusLabels[job.status]}<ModelVendorMark name={String(job.payload.repo_id || '')} /></span></div>
              <small>{job.kind === 'download.huggingface' ? 'Hugging Face' : 'ModelScope'} · {String(job.payload.destination || '')}</small>
              <div className="download-item-progress"><progress aria-label={t('models:downloadQueue.downloadProgress')} max={1} value={progress} /><span>{formatNumber(progress * 100, 1)}%</span></div>
              {isActiveDownload(job) && job.progress_data?.downloaded_bytes != null && <small className="download-transfer-stats">
                {(job.progress_data.bytes_per_second ? formatBytes(job.progress_data.bytes_per_second).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`) : '0 B')}/s
                {' · '}{formatBytes(job.progress_data.downloaded_bytes)}{job.progress_data.total_bytes ? ` / ${formatBytes(job.progress_data.total_bytes)}` : ''}
                {' · '}{job.progress_data.files_completed ?? 0} {t('models:downloadQueue.filesCompleted')}
              </small>}
              {job.error && <p className="download-item-error">{job.error.message}</p>}
            </div>
            {(isActiveDownload(job) || retry) && <button disabled={busyId !== null || job.status === 'cancelling'} onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              void act(job, retry, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
            }} type="button">{retry ? t('common:retry') : t('common:cancel')}</button>}
            {!isActiveDownload(job) && <button disabled={busyId !== null} aria-label={t('models:downloadQueue.deleteDownloadRecord')}
              title={t('models:downloadQueue.removeTaskHistoryKeepDownloadedFiles')}
              onClick={() => void remove(job)} type="button"><Icon name="trash" size={16} /></button>}
          </article>
        );
      })}
    </section>
  );
}
