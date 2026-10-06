/** Provide PrefixCacheDirectory interface behavior. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { SettingRow } from '../../app/display';
import { useJobStore } from '../../stores/jobStore';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

/** Configure the server cache directory and report the resulting model reload. */
export function PrefixCacheDirectory() {
  const { t } = useTranslation();
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
      toast.success(localized('connections:prefixCacheDirectory.cacheDirectorySubmittedLoadedModelsWillBeReloaded'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  return <SettingRow title={t('connections:prefixCacheDirectory.directory')} detail={actual || '--'}
    trailing={<div className="server-row-actions prefix-directory-controls">
      <select aria-label={t('connections:prefixCacheDirectory.prefixCacheDirectoryMode')} disabled={!ready || saving || pending}
        value={manual ? 'manual' : 'managed'} onChange={(event) => setManual(event.target.value === 'manual')}>
        <option value="managed">{t('connections:prefixCacheDirectory.managedByMfq')}</option><option value="manual">{t('connections:prefixCacheDirectory.manualLocation')}</option>
      </select>
      {manual && <input aria-label={t('connections:prefixCacheDirectory.prefixCacheDirectory')} disabled={saving || pending} value={draft}
        onChange={(event) => setDraft(event.target.value)} placeholder={t('connections:prefixCacheDirectory.absoluteDirectoryPath')} />}
      <button disabled={!ready || saving || pending || manual && !draft.trim()} onClick={() => void apply()} type="button">
        {saving || pending ? t('connections:prefixCacheDirectory.applying') : t('common:apply')}
      </button>
    </div>} />;
}
