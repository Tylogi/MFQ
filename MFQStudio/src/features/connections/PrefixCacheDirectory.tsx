import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { SettingRow } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { useJobStore } from '../../stores/jobStore';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

export function PrefixCacheDirectory() {
  const { tr } = useSettings();
  const { addJob } = useRuntime();
  const [manual, setManual] = useState(false);
  const [draft, setDraft] = useState('');
  const [actual, setActual] = useState('');
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const pending = useJobStore((state) => state.jobs.some((job) => job.kind === 'runtime.memory.configure' && ['queued', 'running', 'cancelling'].includes(job.status)));
  useEffect(() => {
    if (pending) return;
    let disposed = false;
    void runtimeApi.memoryPolicy().then((policy) => {
      if (disposed) return;
      setManual(policy.prefix_directory != null);
      setDraft(policy.prefix_directory || policy.actual_prefix_directory || '');
      setActual(policy.actual_prefix_directory || '');
      setReady(true);
    }).catch(() => {});
    return () => { disposed = true; };
  }, [pending]);
  async function apply() {
    setSaving(true);
    try {
      const accepted = await runtimeApi.configureMemoryPolicy({ prefix_directory: manual ? draft.trim() : null });
      addJob(await jobsApi.getJob(accepted.operation_id));
      toast.success(tr('缓存目录已提交，将重新加载当前模型', 'Cache directory submitted; loaded models will be reloaded'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  return <SettingRow title={tr('目录', 'Directory')} detail={actual || '--'}
    trailing={<div className="server-row-actions prefix-directory-controls">
      <select aria-label={tr('前缀缓存目录模式', 'Prefix cache directory mode')} disabled={!ready || saving || pending}
        value={manual ? 'manual' : 'managed'} onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="managed">{tr('由 MFQ 管理', 'Managed by MFQ')}</option><option value="manual">{tr('手动位置', 'Manual location')}</option>
      </select>
      {manual && <input aria-label={tr('前缀缓存目录', 'Prefix cache directory')} disabled={saving || pending} value={draft}
        onChange={(event) => setDraft(event.target.value)} placeholder={tr('绝对目录路径', 'Absolute directory path')} />}
      <button disabled={!ready || saving || pending || manual && !draft.trim()} onClick={() => void apply()} type="button">
        {saving || pending ? tr('应用中…', 'Applying…') : tr('应用', 'Apply')}
      </button>
    </div>} />;
}
