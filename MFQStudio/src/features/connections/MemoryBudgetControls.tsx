import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useJobStore } from '../../stores/jobStore';
import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { hasSeparateVram, residentMemoryCapacity } from '../runtime/memoryArchitecture';

export function MemoryBudgetControls({ residency }: { residency: string }) {
  const { tr } = useSettings();
  const { addJob, runtime } = useRuntime();
  const connectionScope = useConnectionScope();
  const [modelManual, setModelManual] = useState(false);
  const [totalManual, setTotalManual] = useState(false);
  const [totalDraft, setTotalDraft] = useState('96');
  const [totalDetected, setTotalDetected] = useState<number | null>(null);
  const [prefixManual, setPrefixManual] = useState(false);
  const [modelDraft, setModelDraft] = useState('64');
  const [prefixDraft, setPrefixDraft] = useState('4');
  const [available, setAvailable] = useState(false);
  const [saving, setSaving] = useState(false);
  const jobs = useJobStore((state) => state.jobs);
  const pending = jobs.some((job) => job.kind === 'runtime.memory.configure' && ['queued', 'running', 'cancelling'].includes(job.status));
  const separate = hasSeparateVram(runtime);
  const totalLabel = separate ? tr('总显存常驻预算', 'Total resident VRAM budget') : tr('总常驻内存预算', 'Total resident memory budget');
  const modelLabel = separate ? tr('模型显存驻留', 'Model VRAM residency') : tr('模型总驻留', 'Total model residency');
  const prefixLabel = separate ? tr('前缀显存配额', 'Prefix VRAM allowance') : tr('前缀 RAM 配额', 'Prefix RAM allowance');
  const effectiveTotal = residentMemoryCapacity(runtime) ?? totalDetected;
  useEffect(() => {
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      setModelManual(policy.model_limit_bytes != null);
      setTotalManual(policy.total_limit_bytes != null);
      setTotalDetected(policy.effective_total_limit_bytes ?? null);
      if (policy.total_limit_bytes != null) setTotalDraft(String(policy.total_limit_bytes / 2 ** 30));
      else if (policy.effective_total_limit_bytes) setTotalDraft(String(Math.floor(policy.effective_total_limit_bytes / 2 ** 30)));
      setPrefixManual(policy.prefix_limit_bytes != null);
      if (policy.model_limit_bytes != null) setModelDraft(String(policy.model_limit_bytes / 2 ** 30));
      if (policy.prefix_limit_bytes != null) setPrefixDraft(String(policy.prefix_limit_bytes / 2 ** 30));
      setAvailable(true);
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);
  const valid = (!totalManual || totalDraft.trim() !== '' && Number.isFinite(Number(totalDraft)) && Number(totalDraft) > 0)
    && (!modelManual || modelDraft.trim() !== '' && Number.isFinite(Number(modelDraft)) && Number(modelDraft) > 0)
    && (!prefixManual || prefixDraft.trim() !== '' && Number.isFinite(Number(prefixDraft)) && Number(prefixDraft) >= 0);
  async function apply() {
    const current = connectionScope();
    setSaving(true);
    try {
      const accepted = await runtimeApi.configureMemoryPolicy({
        total_limit_bytes: totalManual ? Math.round(Number(totalDraft) * 2 ** 30) : null,
        model_limit_bytes: modelManual ? Math.round(Number(modelDraft) * 2 ** 30) : null,
        prefix_limit_bytes: prefixManual ? Math.round(Number(prefixDraft) * 2 ** 30) : null,
      });
      if (!current()) return;
      const job = await jobsApi.getJob(accepted.operation_id);
      if (!current()) return;
      addJob(job);
      toast.success(tr('内存与缓存规划已提交', 'Memory and cache plan submitted'));
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { if (current()) setSaving(false); }
  }
  function control(label: string, manual: boolean, setManual: (value: boolean) => void, draft: string, setDraft: (value: string) => void, min: number) {
    return <div className="server-row-actions memory-budget-control">
      <select aria-label={`${label} ${tr('模式', 'mode')}`} disabled={!available || saving || pending} value={manual ? 'manual' : 'automatic'}
        onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="automatic">{tr('自动', 'Automatic')}</option><option value="manual">{tr('手动', 'Manual')}</option>
      </select>
      {manual && <div className="server-input-unit memory-budget-input"><input aria-label={`${label} ${tr('预算上限', 'budget limit')}`} className="server-number-input"
        disabled={saving || pending} type="number" min={min} step="0.5" value={draft}
        onChange={(event) => setDraft(event.target.value)} /><span>GiB</span></div>}
    </div>;
  }
  return <>
    <SettingRow title={totalLabel}
      detail={separate ? tr('显存权重、活跃 KV 与显存热前缀共用此预算；前缀冷层在 SSD，流式专家与 KV 的 RAM 占用单独统计。', 'VRAM weights, live KV and hot prefixes share this budget. Cold prefixes reside on SSD; RAM-streamed experts and KV are accounted separately.') : tr('权重、活跃 KV 和前缀 RAM 共用此预算；接近上限时，先将前缀缓存移至 SSD，再减少常驻专家。', 'Weights, live KV and prefix RAM share this budget; near the limit, move prefix caches to SSD before reducing resident experts.')}
      trailing={<>{control(totalLabel, totalManual, setTotalManual, totalDraft, setTotalDraft, 0.1)}
        {!totalManual && effectiveTotal != null && <small className="memory-budget-available">{(effectiveTotal / 2 ** 30).toFixed(1)} GiB</small>}</>} />
    <SettingRow title={modelLabel}
      detail={separate ? tr('所有模型权重的显存上限；放不下的 MoE 专家在 RAM 中按需传入显存，SSD 专家流式加载尚未支持。', 'VRAM ceiling across model weights; overflowing MoE experts stream from RAM to VRAM. SSD expert streaming is not yet supported.') : tr('所有模型权重的总上限；放不下的 MoE 专家自动转为 SSD 缓存。', 'Total weight ceiling across models; overflowing MoE experts use SSD-backed caching.')}
      trailing={control(modelLabel, modelManual, setModelManual, modelDraft, setModelDraft, 0.1)} />
    <SettingRow title={prefixLabel}
      detail={separate ? tr('所有模型可复用前缀的显存上限，不含活跃 KV；冷前缀直接存入 SSD，不保留 RAM 缓存层。', 'VRAM limit for reusable prefixes, excluding live KV. Cold prefixes go directly to SSD without a RAM cache tier.') : tr('所有模型可复用前缀的 RAM 上限，不含活跃 KV。', 'RAM limit for reusable prefixes across models; excludes active KV.')}
      trailing={control(prefixLabel, prefixManual, setPrefixManual, prefixDraft, setPrefixDraft, 0)} />
    <div className="memory-budget-actions"><small>{tr('当前模型驻留', 'Current weight residency')}: {residency}</small>
      <button disabled={!available || saving || pending || !valid} onClick={() => void apply()} type="button">
        {saving || pending ? tr('应用中…', 'Applying…') : tr('应用预算', 'Apply budgets')}
      </button>
    </div>
  </>;
}
