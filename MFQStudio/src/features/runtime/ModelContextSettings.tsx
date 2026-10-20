import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { ApiError } from '../../shared/api/client';
import { SettingRow } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';
import { runtimeApi, type RuntimeContextPolicy, type YarnContextInfo, type KvQuantizationSettings, type KvMemorySettings } from '../../shared/api/resources/runtime';
import type { RuntimeInstance } from '../../shared/api/types';
import { CompactSelect } from '../../shared/ui/CompactSelect';
import { QsaKvOffloadPanel } from '../connections/QsaKvOffloadPanel';
import { KvQuantizationPanel } from '../connections/KvQuantizationPanel';
import { ModelContextSummary } from './ModelContextSummary';
import { hasSeparateVram } from './memoryArchitecture';
import { contextCacheEstimate, quantizedCacheProfile } from '../models/cacheData';
import { useModelCacheProfiles } from '../models/useModelCacheProfiles';

export function ModelContextSettings() {
  const { instances, runtime, ready, connectionRevision, reloadingInstances, reloadModelContext, addJob } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr } = useSettings();
  const jobs = useJobStore((state) => state.jobs);
  const [selected, setSelected] = useState('');
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [yarnDrafts, setYarnDrafts] = useState<Record<string, boolean>>({});
  const [streamDrafts, setStreamDrafts] = useState<Record<string, boolean>>({});
  const [budgetDrafts, setBudgetDrafts] = useState<Record<string, string>>({});
  const [ramBudgetDrafts, setRamBudgetDrafts] = useState<Record<string, string>>({});
  const [quantizationDrafts, setQuantizationDrafts] = useState<Record<string, KvQuantizationSettings>>({});
  const [memoryDrafts, setMemoryDrafts] = useState<Record<string, KvMemorySettings>>({});
  const [yarnInfo, setYarnInfo] = useState<Record<string, YarnContextInfo>>({});
  const [warnings, setWarnings] = useState<Record<string, string>>({});
  const [modelErrors, setModelErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState('');
  const [policy, setPolicy] = useState<RuntimeContextPolicy | null>(null);
  const [capDraft, setCapDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [reloadingGlobal, setReloadingGlobal] = useState(false);
  const [globalOperationId, setGlobalOperationId] = useState('');
  const [error, setError] = useState('');
  const seenInstances = useRef<Record<string, RuntimeInstance>>({});
  const handledFailures = useRef(new Set<string>());
  const contextJobs = jobs.filter((job) => job.kind === 'runtime.context.configure');
  const globalJob = contextJobs.find(job => job.id === globalOperationId);
  const globalBusy = reloadingGlobal || contextJobs.some(job => ['queued', 'running', 'cancelling'].includes(job.status));
  const terminalVersion = contextJobs.filter((job) => !['queued', 'running', 'cancelling'].includes(job.status))
    .map((job) => `${job.id}:${job.status}`).sort().join('|');
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  for (const item of loaded) seenInstances.current[item.id] = item;
  const models = [...loaded];
  for (const job of contextJobs.filter((job) => ['queued', 'running', 'cancelling'].includes(job.status))) {
    const previous = seenInstances.current[String(job.payload.instance_id)];
    if (previous && !models.some((item) => item.model === previous.model)) models.push(previous);
  }
  const item = models.find((model) => model.model === selected) ?? models[0];
  const loadedIds = loaded.map((model) => model.id).sort().join('|');
  const { cacheProfiles, cacheErrors } = useModelCacheProfiles(models.map((model) => model.model));
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    void Promise.allSettled(loaded.map(async (model) => {
      const info = await runtimeApi.yarnContextInfo(model.id);
      if (!disposed) setYarnInfo((current) => ({ ...current, [model.model]: info }));
    }));
    return () => { disposed = true; };
  }, [ready, loadedIds, connectionRevision, terminalVersion]);
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    void runtimeApi.contextPolicy().then((value) => {
      if (disposed) return;
      setPolicy(value); setError('');
      const rejected = contextJobs.filter(job => job.status === 'failed' && !handledFailures.current.has(job.id)
        && ['invalid_request', 'qsa_kv_configuration_invalid', 'qsa_kv_offload_unsupported', 'yarn_unsupported', 'resident_budget_exhausted', 'kv_memory_insufficient', 'kv_vram_insufficient', 'kv_dynamic_unavailable', 'kv_memory_unsupported'].includes(job.error?.code || ''));
      const names = new Set(rejected.map(job => seenInstances.current[String(job.payload.instance_id)]?.model || String(job.result?.model || '')));
      function withoutRejected<T>(draft: Record<string, T>): Record<string, T> {
        return Object.fromEntries(Object.entries(draft).filter(([name]) => !names.has(name)));
      }
      if (rejected.length) {
        rejected.forEach(job => handledFailures.current.add(job.id));
        setDrafts(withoutRejected); setYarnDrafts(withoutRejected); setStreamDrafts(withoutRejected); setBudgetDrafts(withoutRejected);
        setRamBudgetDrafts(withoutRejected);
        setQuantizationDrafts(withoutRejected);
        setMemoryDrafts(withoutRejected);
      }
    }).catch((cause) => { if (!disposed) setError(errorMessage(cause, tr)); });
    return () => { disposed = true; };
  }, [ready, connectionRevision, terminalVersion]);
  useEffect(() => {
    setSelected(''); setDrafts({}); setYarnDrafts({}); setStreamDrafts({}); setBudgetDrafts({});
    setRamBudgetDrafts({});
    setQuantizationDrafts({});
    setMemoryDrafts({});
    setYarnInfo({}); setWarnings({}); setModelErrors({}); setPolicy(null); setError('');
    setSubmitting(''); setSaving(false); setCapDraft(null); setSaved(false); setReloadingGlobal(false); setGlobalOperationId('');
    seenInstances.current = {};
    handledFailures.current.clear();
  }, [connectionRevision]);

  const savedCap = policy?.max_context_size == null ? '' : String(policy.max_context_size);
  const cap = capDraft ?? savedCap;
  const capChanged = policy != null && cap !== savedCap;
  const overGlobalCap = policy?.max_context_size != null && loaded.some(model => (model.context_size ?? 0) > policy.max_context_size!);
  const name = item?.model ?? '';
  const job = contextJobs.filter((candidate) => candidate.payload.instance_id === item?.id
    || seenInstances.current[String(candidate.payload.instance_id)]?.model === name || candidate.result?.model === name)
    .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
  const active = job != null && ['queued', 'running', 'cancelling'].includes(job.status);
  const busy = active || submitting === name || item != null && reloadingInstances?.[item.id] != null;
  const capacity = item?.context_capacity ?? (runtime?.instance_id === item?.id ? runtime?.context_capacity : undefined);
  const info = yarnInfo[name];
  const savedYarn = policy?.model_yarn_enabled?.[name] ?? info?.enabled ?? false;
  const yarnEnabled = yarnDrafts[name] ?? savedYarn;
  const nativeCapacity = info?.native_context ?? capacity;
  const maximum = yarnEnabled && info?.supported ? info.maximum_context : nativeCapacity;
  const savedValue = policy?.model_overrides[name] == null ? '' : String(policy.model_overrides[name]);
  const value = drafts[name] ?? savedValue;
  const automaticSize = Math.min(nativeCapacity ?? policy?.fallback_context_size ?? item?.context_size ?? 32768, policy?.max_context_size ?? Infinity);
  const previewContext = value === '' ? automaticSize : Number(value);
  const finalContext = Math.min(previewContext, maximum ?? Infinity);
  const savedQuantization = policy?.model_kv_quantization?.[name] ?? item?.kv_quantization
    ?? { enabled: false, bits: 4, algorithm: 'turboquant' } satisfies KvQuantizationSettings;
  const quantization = quantizationDrafts[name] ?? savedQuantization;
  const savedMemory = policy?.model_kv_memory?.[name] ?? item?.kv_memory
    ?? { placement: 'auto', allocation: 'preallocated' } satisfies KvMemorySettings;
  const memory = memoryDrafts[name] ?? savedMemory;
  const profile = cacheProfiles[name] ? quantizedCacheProfile(cacheProfiles[name]!, quantization) : cacheProfiles[name];
  const estimate = profile ? contextCacheEstimate(profile, previewContext) : null;
  const storage = policy?.model_qsa_kv_offload?.[name] ?? item?.qsa_kv_offload;
  const streamEnabled = !item?.kv_memory_supported && item?.qsa_kv_offload_supported === true && (streamDrafts[name] ?? storage?.enabled ?? false);
  const savedBudget = String((storage?.budget_bytes ?? 2 ** 31) / 2 ** 30);
  const budget = budgetDrafts[name] ?? savedBudget;
  const budgetBytes = budget.trim() === '' ? 0 : Math.round(Number(budget) * 2 ** 30);
  const separateVram = hasSeparateVram(runtime);
  const savedRamBudget = storage?.ram_budget_bytes == null ? '' : String(storage.ram_budget_bytes / 2 ** 30);
  const ramBudget = ramBudgetDrafts[name] ?? savedRamBudget;
  const ramBudgetBytes = ramBudget.trim() === '' ? null : Math.round(Number(ramBudget) * 2 ** 30);
  const validRamBudget = !separateVram || ramBudgetBytes === null || Number.isSafeInteger(ramBudgetBytes) && ramBudgetBytes >= 0 && ramBudgetBytes <= 2 ** 50;
  const validNumber = (text: string, limit = 2147483647) => Number.isSafeInteger(Number(text)) && Number(text) >= 512 && Number(text) <= limit;
  const valid = value === '' || validNumber(value, maximum != null ? Number.MAX_SAFE_INTEGER : 2147483647);
  const validBudget = !item?.qsa_kv_offload_supported || validRamBudget && Number.isSafeInteger(budgetBytes) && budgetBytes > 0 && budgetBytes <= 2 ** 50;
  const changed = value !== savedValue || yarnEnabled !== savedYarn || streamEnabled !== (storage?.enabled ?? false) || budget !== savedBudget
    || separateVram && ramBudget !== savedRamBudget || quantization.enabled !== savedQuantization.enabled || quantization.bits !== savedQuantization.bits
    || memory.placement !== savedMemory.placement || memory.allocation !== savedMemory.allocation;
  const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(2)} GiB`;
  const phases: Record<string, string> = { reloading: tr('准备重载', 'Preparing reload'), weights: tr('加载权重', 'Loading weights'),
    finalizing: tr('准备运行时', 'Preparing runtime'), compiling: tr('编译内核', 'Compiling kernels'), warming: tr('预热', 'Warming up') };

  async function apply() {
    if (!item || !policy || !info || !valid || !validBudget || busy) return;
    const current = connectionScope();
    const targetName = item.model;
    const exceeded = value !== '' && maximum != null && Number(value) > maximum;
    const size = value === '' ? null : exceeded ? maximum! : Number(value);
    const warning = exceeded ? yarnEnabled
      ? tr('超出YaRN扩展上下文上限', 'Exceeds the YaRN extended context limit')
      : tr('不能设置超出最大值的值', 'Cannot set a value above the native context limit') : '';
    setWarnings((draft) => ({ ...draft, [targetName]: warning }));
    setModelErrors((draft) => ({ ...draft, [targetName]: '' }));
    if (exceeded) { setDrafts((draft) => ({ ...draft, [targetName]: String(size) })); toast.warning(warning); }
    setSubmitting(targetName);
    try {
      await reloadModelContext(item.id, size, yarnEnabled, item.qsa_kv_offload_supported && !item.kv_memory_supported
        ? { enabled: streamEnabled, budget_bytes: budgetBytes, ...(separateVram ? { ram_budget_bytes: ramBudgetBytes } : {}) } : undefined,
        item.kv_quantization_supported ? quantization : undefined, ...(item.kv_memory_supported ? [memory] as const : [] as const));
      if (current()) toast.success(tr('整套模型设置已提交，重载后生效。', 'Model settings submitted together; they apply after reload.'));
    } catch (cause) {
      if (current()) {
        if (cause instanceof ApiError && ['invalid_request', 'qsa_kv_configuration_invalid', 'qsa_kv_offload_unsupported', 'yarn_unsupported', 'resident_budget_exhausted', 'kv_memory_insufficient', 'kv_vram_insufficient', 'kv_dynamic_unavailable', 'kv_memory_unsupported'].includes(cause.code)) {
          setDrafts(draft => ({ ...draft, [targetName]: savedValue }));
          setYarnDrafts(draft => ({ ...draft, [targetName]: savedYarn }));
          setStreamDrafts(draft => ({ ...draft, [targetName]: storage?.enabled ?? false }));
          setBudgetDrafts(draft => ({ ...draft, [targetName]: savedBudget }));
          setRamBudgetDrafts(draft => ({ ...draft, [targetName]: savedRamBudget }));
          setQuantizationDrafts(draft => ({ ...draft, [targetName]: savedQuantization }));
          setMemoryDrafts(draft => ({ ...draft, [targetName]: savedMemory }));
        }
        setModelErrors((draft) => ({ ...draft, [targetName]: errorMessage(cause, tr) })); toast.error(errorMessage(cause, tr));
      }
    } finally { if (current()) setSubmitting(''); }
  }

  async function savePolicy() {
    if (!policy || saving || !capChanged || cap !== '' && !validNumber(cap)) return;
    const current = connectionScope();
    setSaving(true);
    try {
      const next = await runtimeApi.configureContextPolicy({ max_context_size: cap === '' ? null : Number(cap) });
      if (current()) { setPolicy(next); setCapDraft(null); setSaved(true); setError(''); toast.success(tr('全局上下文上限已保存，下次加载模型时生效。', 'Global context cap saved. It applies on the next model load.')); }
    } catch (cause) { if (current()) { setCapDraft(null); setSaved(false); setError(errorMessage(cause, tr)); } }
    finally { if (current()) setSaving(false); }
  }

  async function reloadGlobalCap() {
    if (!policy || capChanged || saving || globalBusy || !overGlobalCap) return;
    const current = connectionScope();
    setReloadingGlobal(true); setError('');
    try {
      const operation = await runtimeApi.reloadGlobalContextCap();
      if (!current()) return;
      setGlobalOperationId(operation.id); addJob(operation);
      toast.success(tr('已提交超出全局上限的模型重载。', 'Reload submitted for models above the global cap.'));
    } catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
    finally { if (current()) setReloadingGlobal(false); }
  }

  return <div className="model-context-settings">
    <SettingRow title={tr('全局上下文上限', 'Global context cap')} detail={tr('留空由 MFQ 管理。保存后，下次加载模型时生效。“重载”会立即重载所有当前上限超出已保存全局上限的已加载模型，并清除其活动 KV 缓存。', 'Leave empty for MFQ to manage. Saved changes apply on the next model load. Reload immediately reloads all loaded models above the saved global cap and clears their active KV cache.')}
      trailing={<div className="model-context-global-control"><div className="server-row-actions">
        <div className="server-input-unit"><input className="server-number-input" type="number" min={512} value={cap} disabled={!policy || saving || globalBusy}
          aria-label={tr('全局上下文上限', 'Global context cap')} placeholder={tr('MFQ 管理', 'MFQ managed')}
          onChange={(event) => { setCapDraft(event.target.value); setSaved(false); setError(''); }}
          onBlur={() => { if (cap !== '' && !validNumber(cap)) { setCapDraft(null); setError(tr('请输入不小于 512 的有效上下文整数，或留空由 MFQ 管理。已恢复原值。', 'Enter a valid context integer of at least 512, or leave empty for MFQ to manage. The previous value was restored.')); } }} /><span>tokens</span></div>
        <button type="button" aria-label={tr('保存全局上下文上限', 'Save global context cap')}
          disabled={!policy || saving || globalBusy || !capChanged || (cap !== '' && !validNumber(cap))} onClick={() => void savePolicy()}>
          {saving ? tr('保存中…', 'Saving…') : tr('保存', 'Save')}</button>
        <button type="button" aria-label={tr('重载超出全局上限的模型', 'Reload models above the global cap')}
          disabled={!policy || capChanged || saving || globalBusy || !overGlobalCap} onClick={() => void reloadGlobalCap()}>
          {globalBusy ? tr('重载中…', 'Reloading…') : tr('重载', 'Reload')}</button></div>
        {(capChanged || saved) && <span className="model-context-save-state" role="status">{saving ? tr('正在保存…', 'Saving…')
          : capChanged ? tr('未保存', 'Unsaved') : tr('已保存 · 下次加载生效', 'Saved · applies on next load')}</span>}
      </div>} />
    {globalJob?.status === 'succeeded' && <p role="status">{tr(`已重载 ${Number(globalJob.result?.models_reloaded ?? 0)} 个超出全局上限的模型。`, `Reloaded ${Number(globalJob.result?.models_reloaded ?? 0)} models above the global cap.`)}</p>}
    {(error || globalJob?.error) && <p role="alert">{error || errorMessage(new ApiError(400, { error: globalJob!.error! }), tr)}</p>}
    {item && <div className="model-context-models">
      <div className="model-context-list-heading"><div>
        <strong>{tr('单模型上下文', 'Per-model context')}</strong>
        <p>{tr('选择模型后编辑整套上下文设置。留空由 MFQ 管理；单模型设置优先于全局。保存后重载该模型并清除活动 KV 缓存。',
          'Select a model to edit its context settings together. Leave empty for MFQ to manage; per-model settings override the global cap. Saving reloads that model and clears active KV cache.')}</p>
      </div><span>{models.length} {tr('个模型', 'models')}</span></div>
      <label className="model-context-selector"><span>{tr('模型', 'Model')}</span>
        <CompactSelect aria-label={tr('单模型上下文模型', 'Per-model context model')} value={name}
          onChange={(event) => setSelected(event.target.value)}>
          {models.map((model) => <option value={model.model} key={model.model}>{model.model}</option>)}
        </CompactSelect>
      </label>
      <div className="model-context-entry" role="group" aria-label={tr('单模型上下文', 'Per-model context')}>
        <SettingRow title={tr('最大上下文', 'Maximum context')}
          detail={tr(`当前 ${formatNumber(item.context_size)} tokens${capacity ? ` · 原生上限 ${formatNumber(capacity)}` : ''}`,
            `Current ${formatNumber(item.context_size)} tokens${capacity ? ` · Native limit ${formatNumber(capacity)}` : ''}`)}
          trailing={<div className="server-row-actions model-context-controls">
            <output className="model-context-kv" aria-label={tr(`${name} KV 估算`, `${name} KV estimate`)}
              title={cacheErrors[name] ?? tr('单个上下文的完整 KV 需求，含已有 MTP 层；不含 GDN/卷积状态、前缀副本或临时计算开销。',
                'Full KV requirement for one context, including available MTP layers; excludes GDN/convolution state, prefix copies and compute scratch.')}>
              <span>KV <strong>{estimate ? gib(estimate.total) : '—'}</strong></span>
              <small>{estimate ? tr(`（原始KV ${gib(estimate.raw)} · Indexer ${gib(estimate.indexer)}）`,
                `(Raw KV ${gib(estimate.raw)} · Indexer ${gib(estimate.indexer)})`)
                : cacheErrors[name] || profile === null ? tr('无法估算', 'Unavailable')
                  : profile ? tr('输入有效上下文', 'Enter a valid context') : tr('正在计算…', 'Calculating…')}</small>
            </output>
            <div className="server-input-unit"><input aria-label={tr(`${name} 最大上下文`, `${name} maximum context`)}
              className="server-number-input" min={512} max={maximum ?? 2147483647} type="number" value={value} disabled={busy || !policy}
              placeholder={tr('MFQ 管理', 'MFQ managed')} onChange={(event) => {
                setDrafts((draft) => ({ ...draft, [name]: event.target.value }));
                setWarnings((draft) => ({ ...draft, [name]: '' }));
              }} onBlur={() => { if (!valid) {
                setDrafts(draft => ({ ...draft, [name]: savedValue }));
                setModelErrors(draft => ({ ...draft, [name]: tr('上下文必须为不小于 512 的整数，或留空由 MFQ 管理。已恢复原值。', 'Context must be an integer of at least 512, or empty for MFQ to manage. The previous value was restored.') }));
              } }} /><span>tokens</span></div>
          </div>} />
        <div className="model-context-yarn"><div>
          <span>{tr('YaRN上下文上限扩展', 'YaRN context limit extension')}</span>
          <p>{info?.supported ? tr(`最多 ${info.maximum_factor}× · 扩展上限 ${formatNumber(info.maximum_context)} tokens；按保存值计算倍率。`,
            `Up to ${info.maximum_factor}× · Extended limit ${formatNumber(info.maximum_context)} tokens. Scaling follows the saved value.`)
            : tr('当前模型或计算后端暂不支持 YaRN 扩展。', 'YaRN extension is not currently supported by this model or backend.')}</p>
        </div><input type="checkbox" aria-label={tr(`${name} YaRN上下文上限扩展`, `${name} YaRN context limit extension`)}
          checked={yarnEnabled} disabled={busy || !info?.supported || !policy}
          onChange={(event) => {
            setYarnDrafts((draft) => ({ ...draft, [name]: event.target.checked }));
            setWarnings((draft) => ({ ...draft, [name]: '' }));
          }} /></div>
        <KvQuantizationPanel key={`quantization:${name}`} supported={item.kv_quantization_supported === true} disabled={busy || !policy}
          settings={quantization} onChange={value => setQuantizationDrafts(draft => ({ ...draft, [name]: value }))} />
        {item.kv_memory_supported && <>
          <SettingRow title={tr('KV 存储位置', 'KV placement')}
            detail={tr('原始 KV 按所选档位存储；自动档优先显存，其次内存。SSD 只在实际溢出时占用空间。', 'Raw KV follows this policy. Automatic placement uses VRAM first, then RAM. SSD space is used only when data spills.')}
            trailing={<CompactSelect aria-label={tr('KV 存储位置', 'KV placement')} value={memory.placement} disabled={busy || !policy}
              onChange={event => setMemoryDrafts(draft => ({ ...draft, [name]: { ...memory,
                placement: event.target.value as KvMemorySettings['placement'],
                allocation: ['ram', 'auto_ssd'].includes(event.target.value) ? 'preallocated' : memory.allocation } }))}>
              <option value="auto">{tr('自动（不启用 SSD KV）', 'Automatic (without SSD KV)')}</option>
              <option value="auto_ssd">{tr('自动（启用 SSD KV）', 'Automatic (with SSD KV)')}</option>
              <option value="vram">{tr('全显存', 'VRAM only')}</option>
              <option value="ram">{tr('全内存', 'RAM only')}</option>
            </CompactSelect>} />
          <SettingRow title={tr('KV 分配方式', 'KV allocation')}
            detail={tr('预分配按最大上下文保留容量。实验性动态 KV 保留固定虚拟地址，按需增加显存；目前支持全显存和自动无 SSD 档。', 'Preallocation reserves capacity for the maximum context. Experimental dynamic KV keeps stable virtual addresses and maps VRAM as needed; currently available for VRAM and automatic without SSD.')}
            trailing={<CompactSelect aria-label={tr('KV 分配方式', 'KV allocation')} value={memory.allocation} disabled={busy || !policy}
              onChange={event => setMemoryDrafts(draft => ({ ...draft, [name]: { ...memory, allocation: event.target.value as KvMemorySettings['allocation'] } }))}>
              <option value="preallocated">{tr('预分配（默认）', 'Preallocated (default)')}</option>
              <option value="dynamic" disabled={['ram', 'auto_ssd'].includes(memory.placement)}>{tr('动态（实验性）', 'Dynamic (experimental)')}</option>
            </CompactSelect>} />
        </>}
        {!item.kv_memory_supported && <QsaKvOffloadPanel key={name} model={item} context={finalContext} enabled={streamEnabled} budget={budget} profile={profile}
          separateVram={separateVram} ramBudget={ramBudget}
          onRamBudgetChange={(value) => setRamBudgetDrafts((draft) => ({ ...draft, [name]: value }))}
          onBudgetBlur={() => { if (!Number.isSafeInteger(budgetBytes) || budgetBytes <= 0 || budgetBytes > 2 ** 50) {
            setBudgetDrafts(draft => ({ ...draft, [name]: savedBudget }));
            setModelErrors(draft => ({ ...draft, [name]: tr('请输入大于 0 的有效 KV 常驻预算，已恢复原值。', 'Enter a valid resident KV budget greater than zero. The previous value was restored.') }));
          } }}
          onRamBudgetBlur={() => { if (!validRamBudget) {
            setRamBudgetDrafts(draft => ({ ...draft, [name]: savedRamBudget }));
            setModelErrors(draft => ({ ...draft, [name]: tr('内存流式 KV 预算请输入有效的非负数，或留空使用自动设置，已恢复原值。', 'Enter a valid nonnegative RAM streaming KV budget, or leave empty for automatic sizing. The previous value was restored.') }));
          } }}
          metadataError={cacheErrors[name]} disabled={busy || !policy}
          onEnabledChange={(enabled) => setStreamDrafts((draft) => ({ ...draft, [name]: enabled }))}
          onBudgetChange={(value) => setBudgetDrafts((draft) => ({ ...draft, [name]: value }))} />}
        {warnings[name] && <p role="alert">{warnings[name]} · {formatNumber(maximum)} tokens</p>}
        {modelErrors[name] && <p role="alert">{modelErrors[name]}</p>}
        {!valid && <p role="alert">{tr('请输入有效整数，或留空使用自动设置。', 'Enter a valid integer, or leave empty for automatic context.')}</p>}
        {!validBudget && <p role="alert">{validRamBudget ? tr('请输入大于 0 的有效 KV 常驻预算。', 'Enter a valid resident KV budget greater than zero.')
          : tr('内存流式 KV 预算应为非负数，或留空使用自动设置。', 'RAM streaming KV budget must be nonnegative, or empty for automatic sizing.')}</p>}
        {active && <div className="model-context-progress" role="status">
          <span>{phases[String(job.progress_data?.phase)] ?? tr('等待执行', 'Queued')} · {job.progress_data?.context_size ?? finalContext} · {Math.round(job.progress * 100)}%</span>
          <progress aria-label={tr(`${name} 上下文设置进度`, `${name} context update progress`)} max={1} value={job.progress} />
        </div>}
        {job?.status === 'succeeded' && !changed && <p role="status">{tr('已保存到该模型', 'Saved to this model')} · {formatNumber(typeof job.result?.max_context === 'number' ? job.result.max_context : item.context_size)} tokens</p>}
        {job && ['failed', 'cancelled', 'interrupted'].includes(job.status) && <p role="alert">{job.error ? errorMessage(new ApiError(400, { error: job.error }), tr) : tr('模型设置未完成，请重试。', 'Model settings did not complete. Please retry.')}</p>}
        <div className="model-context-footer">
          <ModelContextSummary model={name} context={valid ? finalContext : NaN} nativeContext={nativeCapacity} yarnEnabled={yarnEnabled}
            profile={profile} streaming={streamEnabled} budget={budgetBytes} separateVram={separateVram} ramBudget={ramBudgetBytes}
            changed={changed} quantization={quantization} memory={item.kv_memory_supported ? memory : undefined} />
          <button className="model-context-save" aria-label={tr('保存到该模型', 'Save to this model')} type="button"
            disabled={busy || !valid || !validBudget || !policy || !info} onClick={() => void apply()}>
            {busy ? tr('保存中…', 'Saving…') : tr('保存到该模型', 'Save to this model')}
          </button>
        </div>
      </div>
    </div>}
  </div>;
}
