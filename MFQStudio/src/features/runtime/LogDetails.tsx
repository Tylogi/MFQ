import { useEffect, useState, type ReactNode } from 'react';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobResource, RuntimeLogEntry } from '../../shared/api/types';
import { errorMessage, formatNumber } from '../../app/formatters';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';
import { displayPrefillMetric, preferPositiveMetric } from './metrics';
import type { RequestLogRecord } from './useLogHistory';

export function LogDetails({ value, children }: { value: unknown; children?: ReactNode }) {
  const { tr } = useSettings();
  const serialized = JSON.stringify(value, null, 2);
  async function copy() {
    try { await navigator.clipboard.writeText(serialized); toast.success(tr('已复制', 'Copied')); }
    catch (cause) { toast.error(errorMessage(cause)); }
  }
  return <div className="log-record-detail">
    {children}
    <div className="log-detail-heading"><span>{tr('原始记录', 'Raw record')}</span>
      <button type="button" onClick={() => void copy()}><Icon name="copy" size={13} />{tr('复制', 'Copy')}</button></div>
    <pre>{serialized}</pre>
  </div>;
}

export function RequestDetails({ record }: { record: RequestLogRecord }) {
  const { tr } = useSettings();
  const { request } = record;
  const prefill = displayPrefillMetric(request);
  const metric = (value: number | undefined, suffix = '') => value == null ? '--' : `${formatNumber(value, 1)}${suffix}`;
  const values = [
    [tr('输入 / 输出 tokens', 'Input / output tokens'), `${metric(request.prompt_tokens)} / ${metric(request.completion_tokens)}`],
    [tr('预填充', 'Prefill'), `${metric(prefill.tokensPerSecond, ' tok/s')} · ${metric(prefill.milliseconds, ' ms')}`],
    [tr('解码', 'Decode'), `${metric(request.decode_tps, ' tok/s')} · ${metric(request.decode_ms, ' ms')}`],
    [tr('首字延迟', 'TTFT'), metric(request.ttft_ms, ' ms')],
    [tr('请求总耗时', 'Total request time'), metric(preferPositiveMetric(request.complete_generation_ms, request.generation_ms), ' ms')],
    [tr('停止原因', 'Finish reason'), request.finish_reason || '--'],
  ];
  return <LogDetails value={record}><div className="log-detail-metrics">{values.map(([label, value]) =>
    <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></LogDetails>;
}

export function JobDetails({ job, onRemove, busy, refreshAt }: { job: JobResource; onRemove: () => void; busy: boolean; refreshAt?: number | null }) {
  const { tr } = useSettings();
  const [logs, setLogs] = useState<RuntimeLogEntry[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => { setLogs([]); }, [job.id]);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    setLoading(true); setError('');
    void jobsApi.jobEvents(job.id, controller.signal).then((next) => { if (!disposed) setLogs(next); })
      .catch((cause) => { if (!disposed) setError(errorMessage(cause)); })
      .finally(() => { if (!disposed) setLoading(false); });
    return () => { disposed = true; controller.abort(); };
  }, [job.id, job.updated_at, refreshAt]);
  const completed = ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status);
  return <LogDetails value={job}>
    {job.error && <p className="logs-error" role="status">{job.error.message}</p>}
    <div className="log-detail-heading"><span>{tr('任务事件', 'Job events')}</span>
      {completed && <button type="button" className="danger" disabled={busy} onClick={onRemove}>
        <Icon name="trash" size={13} />{tr('删除此记录', 'Delete this record')}</button>}</div>
    {error && <p className="logs-error" role="status">{error}</p>}
    <div className="log-job-events">{logs.length ? logs.map((entry) => <div key={entry.sequence}>
      <time>{new Date(entry.created_at).toLocaleTimeString()}</time><p>{entry.message}</p>
    </div>) : <p>{loading ? tr('读取任务事件…', 'Loading job events…') : error
      ? tr('任务事件读取失败', 'Job events unavailable') : tr('没有附加事件', 'No additional events')}</p>}</div>
  </LogDetails>;
}
