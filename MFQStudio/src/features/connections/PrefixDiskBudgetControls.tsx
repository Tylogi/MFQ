import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { SettingRow } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';

export function PrefixDiskBudgetControls({ budget }: { budget: number }) {
  const { addJob } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr } = useSettings();
  const [manual, setManual] = useState(false);
  const [draft, setDraft] = useState('100');
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const pending = useJobStore((state) => state.jobs.some((job) => job.kind === 'runtime.memory.configure'
    && ['queued', 'running', 'cancelling'].includes(job.status)));
  useEffect(() => {
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      setManual(policy.prefix_disk_limit_bytes != null);
      if (policy.prefix_disk_limit_bytes != null) setDraft(String(policy.prefix_disk_limit_bytes / 2 ** 30));
      setReady(true);
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);
  async function apply() {
    const current = connectionScope();
    setSaving(true);
    try {
      const accepted = await runtimeApi.configureMemoryPolicy({ prefix_disk_limit_bytes: manual ? Math.round(Number(draft) * 2 ** 30) : null });
      if (!current()) return;
      const job = await jobsApi.getJob(accepted.operation_id);
      if (!current()) return;
      addJob(job);
      toast.success(tr('SSD 总配额已提交', 'SSD cache budget submitted'));
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { if (current()) setSaving(false); }
  }
  const valid = !manual || draft.trim() !== '' && Number.isFinite(Number(draft)) && Number(draft) >= 0;
  return <SettingRow title={tr('SSD 总配额', 'Total SSD budget')}
    detail={tr('所有模型共用；自动使用剩余磁盘空间与现有缓存总量的一半，超额按 LRU 回收。',
      'Shared across models. Automatic uses half of free space plus existing cache; LRU reclaims overflow.')}
    trailing={<div className="server-row-actions">
      <select aria-label={tr('SSD 配额模式', 'SSD budget mode')} disabled={!ready || saving || pending}
        value={manual ? 'manual' : 'automatic'} onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="automatic">{tr('自动', 'Automatic')}</option><option value="manual">{tr('手动', 'Manual')}</option>
      </select>
      {manual ? <div className="server-input-unit"><input aria-label={tr('SSD 预算上限', 'SSD budget limit')}
        className="server-number-input" type="number" min="0" step="1" disabled={saving || pending}
        value={draft} onChange={(event) => setDraft(event.target.value)} /><span>GiB</span></div>
        : <span className="server-managed-value">{budget > 0 ? `${formatNumber(budget / 2 ** 30, 0)} GiB` : '--'}</span>}
      <button type="button" disabled={!ready || saving || pending || !valid} onClick={() => void apply()}>
        {saving || pending ? tr('应用中…', 'Applying…') : tr('应用', 'Apply')}
      </button>
    </div>} />;
}
