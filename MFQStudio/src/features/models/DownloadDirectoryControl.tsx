import { type FormEvent, useEffect, useRef, useState } from 'react';
import { errorMessage } from '../../app/formatters';
import { Icon } from '../../app/display';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { modelsApi, type DownloadDirectoryStatus } from '../../shared/api/resources/models';
import type { ModelDirectoryList } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { ModelMemoryPressure } from './ModelMemoryPressure';

export function DownloadDirectoryControl({ onDirectoryChange }: { onDirectoryChange(path: string | undefined): void }) {
  const { tr } = useSettings();
  const { connectionRevision } = useRuntime();
  const connectionScope = useConnectionScope();
  const [directory, setDirectory] = useState<string | null>(null);
  const [storage, setStorage] = useState<DownloadDirectoryStatus | null>(null);
  const [browser, setBrowser] = useState<ModelDirectoryList | null>(null);
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingReclaim, setSavingReclaim] = useState(false);
  const [error, setError] = useState('');
  const trigger = useRef<HTMLElement | null>(null);
  const request = useRef(0);
  const statusRequest = useRef(0);
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setDirectory(null); setStorage(null); onDirectoryChange(undefined); setOpen(false); setBrowser(null); setLoading(false); setSaving(false); setSavingReclaim(false); setError('');
    async function refresh(initial = false) {
      const revision = ++statusRequest.current;
      try {
        const value = await modelsApi.downloadDirectory();
        if (!disposed && revision === statusRequest.current) {
          setDirectory(value.actual_directory); setStorage(value); onDirectoryChange(value.actual_directory);
        }
      } catch (cause) {
        if (!disposed && revision === statusRequest.current) {
          if (initial) toast.error(errorMessage(cause, tr));
        }
      } finally {
        if (!disposed) timer = setTimeout(() => void refresh(), 5000);
      }
    }
    void refresh(true);
    return () => { disposed = true; clearTimeout(timer); request.current += 1; statusRequest.current += 1; };
  }, [connectionRevision, onDirectoryChange]);

  async function browse(id?: string | null, location?: string | null) {
    const current = connectionScope();
    const previousPath = browser?.current_path || directory || '/';
    const revision = ++request.current;
    setLoading(true); setError('');
    try {
      const value = await modelsApi.modelDirectories(id, location);
      if (!current() || request.current !== revision) return;
      setBrowser(value); setPath(value.current_path || location || '/');
    } catch (cause) {
      if (current() && request.current === revision) { setPath(previousPath); setError(errorMessage(cause, tr)); }
    } finally {
      if (current() && request.current === revision) setLoading(false);
    }
  }
  async function choose() {
    if (!browser?.current_id || saving) return;
    const current = connectionScope();
    setSaving(true); setError('');
    try {
      const value = await modelsApi.configureDownloadDirectory(browser.current_id);
      if (!current()) return;
      statusRequest.current += 1;
      setDirectory(value.actual_directory); setStorage(value); onDirectoryChange(value.actual_directory); setOpen(false);
      toast.success(tr('下载目录已保存', 'Download directory saved'));
    } catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
    finally { if (current()) setSaving(false); }
  }
  async function openDirectory() {
    if (!browser?.current_id) return;
    const current = connectionScope();
    try { await modelsApi.openModelDirectory(browser.current_id); }
    catch (cause) { if (current()) setError(errorMessage(cause, tr)); }
  }
  async function jump(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (path.trim()) await browse(null, path.trim());
  }

  async function configureReclaim(enabled: boolean) {
    const current = connectionScope();
    setSavingReclaim(true);
    statusRequest.current += 1;
    try {
      const value = await modelsApi.configureDownloadReclaim(enabled);
      if (current()) { statusRequest.current += 1; setStorage(value); }
    } catch (cause) { if (current()) toast.error(errorMessage(cause, tr)); }
    finally { if (current()) setSavingReclaim(false); }
  }

  const free = storage?.disk_available_bytes ?? storage?.disk_free_bytes;
  const total = storage?.disk_total_bytes;
  const known = free != null && total != null && Number.isFinite(free) && Number.isFinite(total) && free >= 0 && total > 0;

  return <>
    <div className="download-directory-row">
    <div className="download-directory-control">
      <span>{tr('下载目录', 'Download folder')}</span>
      <code title={directory || undefined}>{directory || '--'}</code>
      <button type="button" disabled={!directory || saving} onClick={event => {
        trigger.current = event.currentTarget; setOpen(true); setPath(directory!); void browse(null, directory);
      }}><Icon name="folder" size={14} />{tr('更改', 'Change')}</button>
    </div>
    <div className="download-storage">
      <div className="download-storage-summary"><span>{tr('SSD 剩余', 'SSD free')}</span>
        <span>{known ? `${(free / 2 ** 30).toFixed(1)} / ${(total / 2 ** 30).toFixed(1)} GiB` : '--'}</span></div>
      <ModelMemoryPressure required={known ? free : null} available={known ? total : null}
        label={tr('SSD 剩余空间', 'SSD free space')} emptyLabel="--" />
      <label className="download-prefix-reclaim" title={tr('将同盘可回收的 SSD Prefix Cache 计入可下载空间；下载时缩减其预算并按 LRU 驱逐缓存。', 'Include reclaimable SSD prefix cache on this drive in available download space. Downloads reduce its budget and evict least recently used blocks.')}>
        <input type="checkbox" checked={storage?.evict_prefix_cache ?? false} disabled={!storage || saving || savingReclaim}
          onChange={event => void configureReclaim(event.target.checked)} />
        {tr('驱逐 SSD Prefix 预算', 'Reclaim SSD prefix budget')}
      </label>
    </div>
    </div>
    <ModelDirectoryDialog title={tr('选择下载文件夹', 'Choose download folder')} catalog={{
      modelBrowser: browser, modelBrowserOpen: open, modelBrowserError: error, modelBrowserLoading: loading,
      modelFilesMode: false, setModelBrowserOpen: setOpen, modelBrowserTriggerRef: trigger, busy: saving || loading,
      jumpToModelDirectory: jump, openModelDirectory: browse, modelDirectoryPath: path, setModelDirectoryPath: setPath,
      registerCurrentModelDirectory: choose, openCurrentDirectory: openDirectory,
    }} />
  </>;
}
