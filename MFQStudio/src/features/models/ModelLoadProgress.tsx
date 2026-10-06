/** Provide ModelLoadProgress interface behavior. */
import { useTranslation } from 'react-i18next';
import { useJobStore } from '../../stores/jobStore';
import { formatNumber } from '../../app/formatters';

/** Display the latest loading progress reported for a model. */
export function ModelLoadProgress({ model }: { model: string }) {
  const { t } = useTranslation();
  const job = useJobStore((state) => state.jobs.find((item) => item.kind === 'model.load'
    && item.payload.model === model && ['queued', 'running', 'cancelling'].includes(item.status)));
  if (!job) return null;
  const measured = Number.isFinite(job.progress) && job.progress > 0.02;
  const progress = measured ? Math.min(1, Math.max(0, job.progress)) : undefined;
  return (
    <div className="model-load-progress">
      <progress aria-label={t('models:modelLoadProgress.loadProgress', { model: model })} max={1} value={progress} />
      <span>{job.status === 'cancelling' ? t('models:modelLoadProgress.cancelling')
        : job.status === 'queued' ? t('models:modelLoadProgress.queued')
        : measured ? `${formatNumber(progress! * 100, 1)}%`
        : t('models:modelLoadProgress.preparing')}</span>
    </div>
  );
}
