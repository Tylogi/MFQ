import { useEffect, useRef, useState, type ReactNode } from 'react';
import { runtimeApi, type RuntimeMemoryPolicy } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { ApiError } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useJobStore } from '../../stores/jobStore';
import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { hasSeparateVram, residentMemoryCapacity } from '../runtime/memoryArchitecture';

export function MemoryBudgetControls({ residency, children }: { residency: string; children?: ReactNode }) {
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
  const [error, setError] = useState('');
  const applied = useRef<RuntimeMemoryPolicy | null>(null);
  const [operationId, setOperationId] = useState('');
  const jobs = useJobStore((state) => state.jobs);
  const pending = jobs.some((job) => job.kind === 'runtime.memory.configure' && ['queued', 'running', 'cancelling'].includes(job.status));
  const operation = jobs.find(job => job.id === operationId);
  useEffect(() => {
    if (operation && ['failed', 'cancelled', 'interrupted'].includes(operation.status)) {
      restore();
      setError(operation.error ? errorMessage(new ApiError(400, { error: operation.error }), tr) : tr('预算未能应用，已恢复原设置。', 'The budget could not be applied. The previous settings were restored.'));
    }
  }, [operation?.id, operation?.status]);
  const separate = hasSeparateVram(runtime);
  const totalLabel = separate ? tr('总显存常驻预算', 'Total resident VRAM budget') : tr('总常驻内存预算', 'Total resident memory budget');
  const modelLabel = separate ? tr('模型显存驻留', 'Model VRAM residency') : tr('模型总驻留', 'Total model residency');
  const prefixLabel = separate ? tr('前缀显存配额', 'Prefix VRAM allowance') : tr('前缀 RAM 配额', 'Prefix RAM allowance');
  const effectiveTotal = residentMemoryCapacity(runtime) ?? totalDetected;
  useEffect(() => {
    if (pending) return;
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      applied.current = policy;
      setModelManual(policy.model_limit_bytes != null);
      setTotalManual(policy.total_limit_bytes != null);
      setTotalDetected(policy.effective_total_limit_bytes ?? null);
      if (policy.total_limit_bytes != null) setTotalDraft(String(policy.total_limit_bytes / 2 ** 30));
      else if (policy.effective_total_limit_bytes) setTotalDraft((policy.effective_total_limit_bytes / 2 ** 30).toFixed(1));
      setPrefixManual(policy.prefix_limit_bytes != null);
      if (policy.model_limit_bytes != null) setModelDraft(String(policy.model_limit_bytes / 2 ** 30));
      if (policy.prefix_limit_bytes != null) setPrefixDraft(String(policy.prefix_limit_bytes / 2 ** 30));
      setAvailable(true);
    }).catch(() => {});
    return () => { disposed = true; };
  }, [pending]);
  function restore() {
    const policy = applied.current;
    if (!policy) return;
    setTotalManual(policy.total_limit_bytes != null); setModelManual(policy.model_limit_bytes != null); setPrefixManual(policy.prefix_limit_bytes != null);
    if (policy.total_limit_bytes != null) setTotalDraft(String(policy.total_limit_bytes / 2 ** 30));
    if (policy.model_limit_bytes != null) setModelDraft(String(policy.model_limit_bytes / 2 ** 30));
    if (policy.prefix_limit_bytes != null) setPrefixDraft(String(policy.prefix_limit_bytes / 2 ** 30));
  }
  const validBudget = (text: string, zero: boolean) => text.trim() !== '' && Number.isSafeInteger(Math.round(Number(text) * 2 ** 30)) && (zero ? Number(text) >= 0 : Math.round(Number(text) * 2 ** 30) > 0);
  const valid = (!totalManual || validBudget(totalDraft, false)) && (!modelManual || validBudget(modelDraft, false)) && (!prefixManual || validBudget(prefixDraft, true));
  async function apply() {
    if (!valid || saving || pending) return;
    const current = connectionScope();
    setSaving(true); setError('');
    let submitted = false;
    try {
      const accepted = await runtimeApi.configureMemoryPolicy({
        total_limit_bytes: totalManual ? Math.round(Number(totalDraft) * 2 ** 30) : null,
        model_limit_bytes: modelManual ? Math.round(Number(modelDraft) * 2 ** 30) : null,
        prefix_limit_bytes: prefixManual ? Math.round(Number(prefixDraft) * 2 ** 30) : null,
      });
      if (!current()) return;
      submitted = true;
      setOperationId(accepted.operation_id);
      const job = await jobsApi.getJob(accepted.operation_id);
      if (!current()) return;
      addJob(job);
      toast.success(tr('内存与缓存规划已提交', 'Memory and cache plan submitted'));
    } catch (cause) { if (current()) {
      if (!submitted) restore();
      setError(submitted ? tr('预算已提交，但暂时无法读取进度，请刷新后查看。', 'Budgets were submitted, but progress is unavailable. Refresh to check.') : errorMessage(cause, tr));
    } }
    finally { if (current()) setSaving(false); }
  }
  function changeTotalMode(manual: boolean) {
    if (manual && !totalManual && effectiveTotal != null) setTotalDraft((effectiveTotal / 2 ** 30).toFixed(1));
    setTotalManual(manual);
  }
  function control(label: string, manual: boolean, setManual: (value: boolean) => void, draft: string, setDraft: (value: string) => void, min: number) {
    return <div className="server-row-actions memory-budget-control">
      {manual && <div className="server-input-unit memory-budget-input"><input aria-label={`${label} ${tr('预算上限', 'budget limit')}`} className="server-number-input"
        disabled={saving || pending} type="number" min={min} step="0.5" value={draft}
        onChange={(event) => { setDraft(event.target.value); setError(''); }}
        onBlur={() => {
          if (validBudget(draft, min === 0)) return;
          const bytes = label === totalLabel ? applied.current?.total_limit_bytes ?? effectiveTotal : label === modelLabel ? applied.current?.model_limit_bytes : applied.current?.prefix_limit_bytes;
          setDraft(bytes != null ? label === totalLabel && applied.current?.total_limit_bytes == null ? (bytes / 2 ** 30).toFixed(1) : String(bytes / 2 ** 30) : min === 0 ? '0' : '64');
          setError(min === 0 ? tr('配额必须是非负数，已恢复原值。', 'The allowance must be nonnegative. The previous value was restored.') : tr('预算必须大于 0，已恢复原值。', 'The budget must be greater than zero. The previous value was restored.'));
        }} /><span>GiB</span></div>}
      <select aria-label={`${label} ${tr('模式', 'mode')}`} disabled={!available || saving || pending} value={manual ? 'manual' : 'automatic'}
        onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="automatic">{tr('自动', 'Automatic')}</option><option value="manual">{tr('手动', 'Manual')}</option>
      </select>
    </div>;
  }
  return <>
    <SettingRow title={totalLabel}
      detail={separate ? tr('显存权重、活跃 KV 与显存热前缀共用此预算；RAM 专家、KV 和前缀缓存单独统计。', 'VRAM weights, live KV and hot prefixes share this budget. RAM experts, KV and prefix caches are accounted separately.') : tr('权重、活跃 KV 和前缀 RAM 共用此预算；接近上限时，先将前缀缓存移至 SSD，再减少常驻专家。', 'Weights, live KV and prefix RAM share this budget; near the limit, move prefix caches to SSD before reducing resident experts.')}
      trailing={<>{!totalManual && effectiveTotal != null && <small className="memory-budget-available">{(effectiveTotal / 2 ** 30).toFixed(1)} GiB</small>}
        {control(totalLabel, totalManual, changeTotalMode, totalDraft, setTotalDraft, 0.1)}</>} />
    <SettingRow title={modelLabel}
      detail={separate ? tr('所有模型权重的显存上限；放不下的 MoE 专家在 RAM 中按需传入显存，SSD 专家流式加载尚未支持。', 'VRAM ceiling across model weights; overflowing MoE experts stream from RAM to VRAM. SSD expert streaming is not yet supported.') : tr('所有模型权重的总上限；放不下的 MoE 专家自动转为 SSD 缓存。', 'Total weight ceiling across models; overflowing MoE experts use SSD-backed caching.')}
      trailing={control(modelLabel, modelManual, setModelManual, modelDraft, setModelDraft, 0.1)} />
    <SettingRow title={prefixLabel}
      detail={separate ? tr('所有模型可复用前缀的显存上限，不含活跃 KV；RAM 与 SSD 前缀缓存配额单独设置。', 'VRAM limit for reusable prefixes, excluding live KV. RAM and SSD prefix allowances are configured separately.') : tr('所有模型可复用前缀的 RAM 上限，不含活跃 KV。', 'RAM limit for reusable prefixes across models; excludes active KV.')}
      trailing={control(prefixLabel, prefixManual, setPrefixManual, prefixDraft, setPrefixDraft, 0)} />
    {children}
    <div className="memory-budget-actions"><small>{tr('当前模型驻留', 'Current weight residency')}: {residency}</small>
      <button disabled={!available || saving || pending || !valid} onClick={() => void apply()} type="button">
        {saving || pending ? tr('应用中…', 'Applying…') : tr('应用预算', 'Apply budgets')}
      </button>
    </div>
    {(error || !valid) && <p className="server-apply-state" role="alert">{error || tr('总预算和模型预算必须大于 0；前缀配额必须为非负数。', 'Total and model budgets must be greater than zero; the prefix allowance must be nonnegative.')}</p>}
  </>;
}
