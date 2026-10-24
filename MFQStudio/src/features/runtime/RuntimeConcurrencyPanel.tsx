import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeInstance } from '../../shared/api/types';
import { useSettings } from '../settings/SettingsProvider';
import { ConcurrencyFields } from '../models/ConcurrencyFields';
import { validConcurrency, type ConcurrencyDefaults } from '../models/concurrencySettings';
import { useJobStore } from '../../stores/jobStore';

function InstanceConcurrency({ instance }: { instance: RuntimeInstance }) {
  const { tr } = useSettings();
  const { refreshRuntime, addJob } = useRuntime();
  const connectionScope = useConnectionScope();
  const concurrency = instance.concurrency!;
  const [draft, setDraft] = useState<ConcurrencyDefaults>(concurrency.requested);
  const [applied, setApplied] = useState(concurrency);
  useEffect(() => { setApplied(concurrency); }, [concurrency]);
  const [pending, setPending] = useState(false);
  const [reloadJobId, setReloadJobId] = useState('');
  const reloadJob = useJobStore(state => state.jobs.find(job => job.id === reloadJobId));
  const reloadSubmitted = !!reloadJobId
    && (!reloadJob || ['queued', 'running', 'cancelling'].includes(reloadJob.status));
  const reloadError = reloadJob?.status === 'failed' ? reloadJob.error?.message
    ?? tr('重新加载失败，请重试。', 'Reload failed. Please retry.') : '';
  const [error, setError] = useState('');
  const effective = {
    max_requests: draft.max_requests ?? applied.requested.max_requests,
    max_queued_requests: draft.max_queued_requests ?? applied.requested.max_queued_requests,
    queue_timeout_seconds: draft.queue_timeout_seconds ?? applied.requested.queue_timeout_seconds,
  };
  const reloadRequired = effective.max_requests !== applied.requested.max_requests;
  const valid = validConcurrency(draft, { defaults: applied.requested, limits: applied.limits });
  async function submit(reload: boolean) {
    const current = connectionScope();
    setPending(true);
    setError('');
    try {
      if (reload) {
        const job = await runtimeApi.reloadConcurrency(instance.id, effective);
        if (!current()) return;
        addJob(job);
        setReloadJobId(job.id);
      } else {
        const result = await runtimeApi.updateConcurrency(instance.id, effective);
        if (!current()) return;
        setApplied(result.concurrency);
      }
      await refreshRuntime(false);
    } catch (cause) {
      if (current()) setError(errorMessage(cause, tr));
    } finally {
      if (current()) setPending(false);
    }
  }
  return <section className="model-load-context" aria-label={instance.model}>
    <h3>{instance.model}</h3>
    <p>{tr(`要求 ${applied.requested.max_requests}、模型实际 ${instance.request_capacity ?? applied.effective.max_requests}`,
      `Requested ${applied.requested.max_requests}, model effective `
      + `${instance.request_capacity ?? applied.effective.max_requests}`)}</p>
    <p>{tr(`排队 ${instance.queued_requests} · 平均等待 ${formatNumber(instance.queue_wait_ms_avg, 1)} ms · `
      + `最大等待 ${formatNumber(instance.queue_wait_ms_max, 1)} ms · 拒绝 ${instance.rejected_requests ?? 0}`,
    `Queued ${instance.queued_requests} · Average wait ${formatNumber(instance.queue_wait_ms_avg, 1)} ms · `
      + `Max wait ${formatNumber(instance.queue_wait_ms_max, 1)} ms · Rejected ${instance.rejected_requests ?? 0}`)}</p>
    <ConcurrencyFields value={draft} onChange={setDraft}
      policy={{ defaults: applied.requested, limits: applied.limits }} disabled={pending || reloadSubmitted} />
    {reloadRequired && <p role="status">{tr('并发请求数需要重新加载后生效。保存队列设置不会重新加载模型。',
      'Concurrent request rows require a reload. Saving queue settings keeps the model running.')}</p>}
    {(error || reloadError) && <p role="alert">{error || reloadError}</p>}
    {reloadSubmitted && <p role="status">{tr('已提交重新加载任务。', 'Reload job submitted.')}</p>}
    <button type="button" disabled={pending || reloadSubmitted || !valid} onClick={() => void submit(false)}>
      {tr('保存队列设置', 'Save queue settings')}</button>
    {reloadRequired && <button type="button" className="primary"
      disabled={pending || reloadSubmitted || !valid || instance.state !== 'ready' || instance.queued_requests > 0}
      onClick={() => void submit(true)}>{tr('以新设置重新加载', 'Reload with new settings')}</button>}
  </section>;
}

/** Show every loaded model's negotiated capacity and live queue controls. */
export function RuntimeConcurrencyPanel() {
  const { instances, connectionRevision } = useRuntime();
  const { tr } = useSettings();
  const models = instances.filter(item => item.concurrency && ['ready', 'busy'].includes(item.state));
  if (!models.length) return null;
  return <TMPanel>
    <h2>{tr('模型并发', 'Model concurrency')}</h2>
    {models.map(instance => <InstanceConcurrency key={`${connectionRevision}:${instance.id}`} instance={instance} />)}
  </TMPanel>;
}
