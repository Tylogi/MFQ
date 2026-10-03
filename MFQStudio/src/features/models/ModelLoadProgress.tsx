import { useJobStore } from '../../stores/jobStore';
import { useSettings } from '../settings/SettingsProvider';
import { formatNumber } from '../../app/formatters';

export function ModelLoadProgress({ model }: { model: string }) {
  const { tr } = useSettings();
  const job = useJobStore((state) => state.jobs.find((item) => item.kind === 'model.load'
    && item.payload.model === model && ['queued', 'running', 'cancelling'].includes(item.status)));
  if (!job) return null;
  const measured = Number.isFinite(job.progress) && job.progress > 0.02;
  const progress = measured ? Math.min(1, Math.max(0, job.progress)) : undefined;
  return (
    <div className="model-load-progress">
      <progress aria-label={tr(`${model} 加载进度`, `${model} load progress`)} max={1} value={progress} />
      <span>{job.status === 'cancelling' ? tr('正在取消', 'Cancelling')
        : job.status === 'queued' ? tr('排队中', 'Queued')
        : measured ? `${formatNumber(progress! * 100, 1)}%`
        : tr('准备中', 'Preparing')}</span>
    </div>
  );
}
