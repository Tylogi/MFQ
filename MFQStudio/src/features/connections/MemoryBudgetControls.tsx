/** Provide MemoryBudgetControls interface behavior. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { useJobStore } from '../../stores/jobStore';
import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

/** Edit the shared memory budget and apply it through the server reload workflow. */
export function MemoryBudgetControls({ residency }: { residency: string }) {
  const { t } = useTranslation();
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
      toast.success(localized('connections:memoryBudgetControls.memoryPlanSubmittedLoadedModelsWillBeReloaded'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  function control(label: string, manual: boolean, setManual: (value: boolean) => void, draft: string, setDraft: (value: string) => void, min: number) {
    return <div className="server-row-actions">
      <select aria-label={`${label} ${t('connections:memoryBudgetControls.mode')}`} disabled={!available || saving || pending} value={manual ? 'manual' : 'automatic'}
        onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="automatic">{t('connections:memoryBudgetControls.automatic')}</option><option value="manual">{t('connections:memoryBudgetControls.manual')}</option>
      </select>
      {manual && <div className="server-input-unit"><input aria-label={`${label} ${t('connections:memoryBudgetControls.budgetLimit')}`} className="server-number-input"
        disabled={saving || pending} type="number" min={min} step="0.5" value={draft}
        onChange={(event) => setDraft(event.target.value)} /><span>GiB</span></div>}
    </div>;
  }
  return <>
    <SettingRow title={t('connections:memoryBudgetControls.totalModelResidency')}
      detail={t('connections:memoryBudgetControls.totalWeightCeilingAcrossModelsOverflowingMoeExpertsUseSsdBackedCaching')}
      trailing={control(t('connections:memoryBudgetControls.totalModelResidency'), modelManual, setModelManual, modelDraft, setModelDraft, 0.1)} />
    <SettingRow title={t('connections:memoryBudgetControls.prefixRamAllowance')}
      detail={t('connections:memoryBudgetControls.combinedReusablePrefixHotCacheLimitAcrossModelsExcludingLiveKv')}
      trailing={control(t('connections:memoryBudgetControls.prefixRamAllowance'), prefixManual, setPrefixManual, prefixDraft, setPrefixDraft, 0)} />
    <div className="memory-budget-actions"><small>{t('connections:memoryBudgetControls.currentWeightResidency')}: {residency}</small>
      <button disabled={!available || saving || pending || !valid} onClick={() => void apply()} type="button">
        {saving || pending ? t('connections:memoryBudgetControls.applying') : t('connections:memoryBudgetControls.applyAndReplan')}
      </button>
    </div>
  </>;
}
