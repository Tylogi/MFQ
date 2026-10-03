import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useJobStore } from '../../stores/jobStore';
import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

export function MemoryBudgetControls({ residency }: { residency: string }) {
  const { tr } = useSettings();
  const { addJob } = useRuntime();
  const [modelManual, setModelManual] = useState(false);
  const [prefixManual, setPrefixManual] = useState(false);
  const [modelDraft, setModelDraft] = useState('64');
  const [prefixDraft, setPrefixDraft] = useState('4');
  const [available, setAvailable] = useState(false);
  const [saving, setSaving] = useState(false);
  const jobs = useJobStore((state) => state.jobs);
  const pending = jobs.some((job) => job.kind === 'runtime.memory.configure' && ['queued', 'running', 'cancelling'].includes(job.status));
  useEffect(() => {
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      setModelManual(policy.model_limit_bytes != null);
      setPrefixManual(policy.prefix_limit_bytes != null);
      if (policy.model_limit_bytes != null) setModelDraft(String(policy.model_limit_bytes / 2 ** 30));
      if (policy.prefix_limit_bytes != null) setPrefixDraft(String(policy.prefix_limit_bytes / 2 ** 30));
      setAvailable(true);
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);
  const valid = (!modelManual || modelDraft.trim() !== '' && Number.isFinite(Number(modelDraft)) && Number(modelDraft) > 0)
    && (!prefixManual || prefixDraft.trim() !== '' && Number.isFinite(Number(prefixDraft)) && Number(prefixDraft) >= 0);
  async function apply() {
    setSaving(true);
    try {
      const accepted = await runtimeApi.configureMemoryPolicy({
        model_limit_bytes: modelManual ? Math.round(Number(modelDraft) * 2 ** 30) : null,
        prefix_limit_bytes: prefixManual ? Math.round(Number(prefixDraft) * 2 ** 30) : null,
      });
      addJob(await jobsApi.getJob(accepted.operation_id));
      toast.success(tr('内存规划已提交，将重新加载当前模型', 'Memory plan submitted; loaded models will be reloaded'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  function control(label: string, manual: boolean, setManual: (value: boolean) => void, draft: string, setDraft: (value: string) => void, min: number) {
    return <div className="server-row-actions">
      <select aria-label={`${label} ${tr('模式', 'mode')}`} disabled={!available || saving || pending} value={manual ? 'manual' : 'automatic'}
        onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="automatic">{tr('自动', 'Automatic')}</option><option value="manual">{tr('手动', 'Manual')}</option>
      </select>
      {manual && <div className="server-input-unit"><input aria-label={`${label} ${tr('预算上限', 'budget limit')}`} className="server-number-input"
        disabled={saving || pending} type="number" min={min} step="0.5" value={draft}
        onChange={(event) => setDraft(event.target.value)} /><span>GiB</span></div>}
    </div>;
  }
  return <>
    <SettingRow title={tr('模型总驻留', 'Total model residency')}
      detail={tr('所有模型权重的总上限；放不下的 MoE 专家自动转为 SSD 缓存。', 'Total weight ceiling across models; overflowing MoE experts use SSD-backed caching.')}
      trailing={control(tr('模型总驻留', 'Total model residency'), modelManual, setModelManual, modelDraft, setModelDraft, 0.1)} />
    <SettingRow title={tr('前缀 RAM 配额', 'Prefix RAM allowance')}
      detail={tr('所有模型的可复用前缀热缓存总上限，不包括活跃 KV。', 'Combined reusable prefix hot-cache limit across models, excluding live KV.')}
      trailing={control(tr('前缀 RAM 配额', 'Prefix RAM allowance'), prefixManual, setPrefixManual, prefixDraft, setPrefixDraft, 0)} />
    <div className="memory-budget-actions"><small>{tr('当前模型驻留', 'Current weight residency')}: {residency}</small>
      <button disabled={!available || saving || pending || !valid} onClick={() => void apply()} type="button">
        {saving || pending ? tr('应用中…', 'Applying…') : tr('应用并重新规划', 'Apply and replan')}
      </button>
    </div>
  </>;
}
