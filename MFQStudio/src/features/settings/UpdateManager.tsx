import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { Icon } from '../../app/display';
import { Switch } from '../../shared/ui/Switch';
import { browserUpdateStatus, cachedBrowserStatus, compareVersions, isReleaseTag, setBrowserAutomatic, studioBuild } from './releases';

import type {
  StudioUpdateProgress,
  StudioUpdateStatus,
} from '../../shared/platform/studio';
import { toast } from '../../stores/toastStore';
import {
  deleteStudioVersion,
  downloadStudioVersion,
  installStudioVersion,
  isStudio,
  openStudioExternal,
  setAutomaticStudioUpdates,
  setAutomaticStudioDownloads,
  studioConfirm,
  studioUpdateProgress,
  studioUpdateStatus,
} from '../../shared/platform/studio';

type Translate = (chinese: string, english: string) => string;

function formatBytes(value: number): string {
  if (!value) return "—";
  const gibibytes = value >= 2 ** 30;
  return `${(value / (gibibytes ? 2 ** 30 : 2 ** 20)).toFixed(gibibytes ? 1 : 0)} ${gibibytes ? "GiB" : "MiB"}`;
}

function olderThan(left: string, right: string): boolean {
  return compareVersions(left, right) < 0;
}

function progressLabel(progress: StudioUpdateProgress, tr: Translate): string {
  if (progress.stage === "preparing") return tr("校验并准备中…", "Verifying & preparing…");
  if (progress.total_bytes <= 0) return tr("下载中…", "Downloading…");
  const percent = Math.min(100, Math.round((progress.received_bytes / progress.total_bytes) * 100));
  return `${percent}% · ${formatBytes(progress.received_bytes)} / ${formatBytes(progress.total_bytes)}`;
}

function progressButtonLabel(progress: StudioUpdateProgress | null, tr: Translate): string {
  if (!progress || progress.total_bytes <= 0) return tr("下载中…", "Downloading…");
  if (progress.stage === "preparing") return tr("准备中…", "Preparing…");
  return `${Math.min(100, Math.round((progress.received_bytes / progress.total_bytes) * 100))}%`;
}

export function useStudioUpdates(onError: (message: string) => void) {
  const [status, setStatus] = useState<StudioUpdateStatus | null>(() => isStudio() ? null : cachedBrowserStatus());
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<StudioUpdateProgress | null>(null);
  const operation = useRef(false);
  const attempted = useRef(new Set<string>());
  const automaticCheck = useRef(true);
  automaticCheck.current = status?.automatic_check ?? true;
  const loadStatus = useCallback((force: boolean) => isStudio() ? studioUpdateStatus(force) : browserUpdateStatus(force), []);

  useEffect(() => {
    let cancelled = false;
    let checking = false;
    const check = (force = false) => {
      if (checking || operation.current) return;
      checking = true;
      void loadStatus(force)
        .then((value) => { if (!cancelled) setStatus(value); })
        .catch((cause) => { if (!cancelled) onError(cause instanceof Error ? cause.message : String(cause)); })
        .finally(() => { checking = false; });
    };
    check();
    const timer = window.setInterval(() => check(), 6 * 60 * 60 * 1000);
    const online = () => {
      if (automaticCheck.current) { attempted.current.clear(); check(true); }
    };
    const visible = () => { if (document.visibilityState === 'visible') check(); };
    window.addEventListener('online', online);
    document.addEventListener('visibilitychange', visible);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.removeEventListener('online', online);
      document.removeEventListener('visibilitychange', visible);
    };
  }, [onError, loadStatus]);

  async function refresh() {
    if (operation.current) return;
    operation.current = true;
    attempted.current.clear();
    setBusy("check");
    try {
      setStatus(await loadStatus(true));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
      operation.current = false;
    }
  }

  async function setAutomatic(enabled: boolean) {
    if (operation.current) return;
    operation.current = true;
    setBusy("preference");
    try {
      setStatus(isStudio() ? await setAutomaticStudioUpdates(enabled) : setBrowserAutomatic(enabled));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
      operation.current = false;
    }
  }

  async function setAutomaticDownload(enabled: boolean) {
    if (operation.current) return;
    operation.current = true;
    setBusy('preference');
    try { setStatus(await setAutomaticStudioDownloads(enabled)); }
    catch (cause) { onError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(null); operation.current = false; }
  }

  async function download(tag: string) {
    if (operation.current) return;
    operation.current = true;
    setBusy(`download:${tag}`);
    const poll = window.setInterval(() => {
      void studioUpdateProgress().then(setProgress).catch(() => undefined);
    }, 400);
    try {
      setProgress(await studioUpdateProgress());
      await downloadStudioVersion(tag);
      setStatus(await studioUpdateStatus(false));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      window.clearInterval(poll);
      setProgress(null);
      setBusy(null);
      operation.current = false;
    }
  }

  async function install(version: string, tr: Translate) {
    if (operation.current) return;
    const confirmed = await studioConfirm(tr(
      `切换到 MFQ Studio ${version}？应用会保留当前版本并重启，本地 MFQ Server 也会随之重启。`,
      `Switch to MFQ Studio ${version}? The current version will be retained, and the app and managed local MFQ Server will restart.`,
    ));
    if (!confirmed) return;
    if (operation.current) return;
    operation.current = true;
    setBusy(`install:${version}`);
    try {
      await installStudioVersion(version);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
      operation.current = false;
    }
  }

  async function remove(version: string, tr: Translate) {
    if (operation.current) return;
    const confirmed = await studioConfirm(tr(
      `移除已缓存的 MFQ Studio ${version}？`,
      `Remove the cached MFQ Studio ${version}?`,
    ));
    if (!confirmed) return;
    if (operation.current) return;
    operation.current = true;
    setBusy(`delete:${version}`);
    try {
      setStatus(await deleteStudioVersion(version));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
      operation.current = false;
    }
  }

  useEffect(() => {
    const latest = status?.latest;
    if (!isStudio() || busy || !status?.current_release || !status.automatic_check || !status.automatic_download ||
      !status.platform_supported || !status.update_available || !latest?.asset || latest.prerelease || !isReleaseTag(latest.tag) ||
      status.installed_versions.some(item => item.version === latest.version && item.ready) || attempted.current.has(latest.tag)) return;
    attempted.current.add(latest.tag);
    void download(latest.tag);
  }, [status, busy]);

  return { busy, download, install, progress, refresh, remove, setAutomatic, setAutomaticDownload, status };
}

type StudioUpdates = ReturnType<typeof useStudioUpdates>;

const unavailableStudioUpdates: StudioUpdates = {
  busy: null,
  progress: null,
  status: null,
  download: async () => undefined,
  install: async () => undefined,
  refresh: async () => undefined,
  remove: async () => undefined,
  setAutomatic: async () => undefined,
  setAutomaticDownload: async () => undefined,
};

const StudioUpdatesContext = createContext<StudioUpdates>(unavailableStudioUpdates);

export function StudioUpdateProvider({ children }: { children: ReactNode }) {
  const reportError = useCallback((message: string) => {
    toast.error(message);
  }, []);
  const updates = useStudioUpdates(reportError);
  return (
    <StudioUpdatesContext.Provider value={updates}>
      {children}
    </StudioUpdatesContext.Provider>
  );
}

export function useStudioUpdateContext(): StudioUpdates {
  return useContext(StudioUpdatesContext);
}

function ReleaseNotes({ notes, tr }: { notes: string; tr: Translate }) {
  const html = useMemo(() => DOMPurify.sanitize(marked.parse(notes, { async: false }), {
    FORBID_TAGS: ['img', 'style', 'iframe', 'form', 'input', 'button'],
    FORBID_ATTR: ['style', 'id', 'name'],
  }), [notes]);
  return notes.trim() ? <div className="release-notes" onClick={event => {
    const link = (event.target as HTMLElement).closest('a');
    if (!link) return;
    event.preventDefault();
    void openStudioExternal(link.href).catch(cause => toast.error(String(cause)));
  }} dangerouslySetInnerHTML={{ __html: html }} /> : <p className="version-note">{tr('该版本未提供说明。', 'No notes were published for this version.')}</p>;
}

export function UpdateManager({
  busy, download, install, progress, refresh, remove, setAutomatic, setAutomaticDownload, status, tr,
}: StudioUpdates & { tr: Translate }) {
  const installed = useMemo(
    () => new Set(status?.installed_versions.filter(item => item.ready && !item.prerelease).map(item => item.version) ?? []),
    [status],
  );
  const releases = useMemo(
    () => (status?.releases || []).filter(item => !item.prerelease && isReleaseTag(item.tag)),
    [status],
  );
  const currentVersion = status?.current_version || studioBuild.version;
  const currentRelease = status?.current_release ?? studioBuild.release;
  const local = (status?.installed_versions || []).filter(item => !item.current && !item.prerelease);
  const experimental = (status?.installed_versions || []).filter(item => !item.current && item.prerelease);
  const open = (url: string) => void openStudioExternal(url).catch(cause => toast.error(String(cause)));
  function localRow(version: NonNullable<StudioUpdateStatus['installed_versions']>[number]) {
    return <div className="local-version" key={version.version}>
      <div><strong>v{version.version}</strong><small>{version.ready
        ? tr('已缓存', 'Cached') : tr('缓存不完整', 'Incomplete cache')}{version.byte_size > 0 ? ' · ' + formatBytes(version.byte_size) : ''}</small></div>
      <button disabled={busy !== null || !version.ready || !status?.platform_supported} onClick={() => void install(version.version, tr)} type="button">{olderThan(version.version, currentVersion) ? tr('回退', 'Roll back') : tr('切换', 'Switch')}</button>
      <button aria-label={tr('移除 v' + version.version + ' 缓存', 'Remove v' + version.version + ' cache')} className="version-remove" disabled={busy !== null} onClick={() => void remove(version.version, tr)} type="button"><Icon name="trash" size={15} /></button>
      {version.notes && <details className="local-version-notes"><summary>{tr('版本说明', 'Version notes')}</summary><ReleaseNotes notes={version.notes} tr={tr} /></details>}
    </div>;
  }
  return <section className="update-manager">
    <div className="update-current-row">
      <div><span>{tr('当前版本', 'Current version')}</span><strong title={currentVersion}>v{currentVersion}</strong>
        <small>{currentRelease ? 'Release' : tr('开发构建 · 实验版本', 'Development build · Experimental')}</small></div>
      <button disabled={busy !== null} onClick={() => void refresh()} type="button"><Icon name="refresh" size={15} />{busy === 'check' ? tr('检查中', 'Checking') : tr('检查更新', 'Check now')}</button>
    </div>
    <div className="version-preferences">
      <div><div><strong>{tr('自动检查更新', 'Automatically check for updates')}</strong><small>{tr('联网后检查 Release，断网时保留已有版本信息。', 'Check Releases when online and retain cached information while offline.')}</small></div>
        <Switch label={tr('自动检查更新', 'Automatically check for updates')} checked={status?.automatic_check ?? true} disabled={!status || busy !== null} onCheckedChange={checked => void setAutomatic(checked)} /></div>
      {isStudio() && <div><div><strong>{tr('自动下载 Release 更新', 'Automatically download Release updates')}</strong><small>{currentRelease
        ? tr('后台下载并校验；安装前确认，随后重启应用与本地服务。', 'Download and verify in the background; confirm before restarting the app and local service.')
        : tr('开发构建不参与自动下载；可手动选择 Release。', 'Development builds do not auto-download updates; choose a Release manually.')}</small></div>
        <Switch label={tr('自动下载 Release 更新', 'Automatically download Release updates')} checked={status?.automatic_download ?? false} disabled={!status || busy !== null || !currentRelease || !status.platform_supported || !status.automatic_check} onCheckedChange={checked => void setAutomaticDownload(checked)} /></div>}
      <p>{status?.checked_at_epoch_seconds
        ? tr('上次检查 ', 'Last checked ') + new Date(status.checked_at_epoch_seconds * 1000).toLocaleString()
        : tr('尚未检查', 'Not checked yet')}</p>
    </div>
    {status?.error && <div className="update-error" role="status"><strong>{tr('未能获取在线版本', 'Could not retrieve online releases')}</strong><span>{tr('可继续查看缓存，联网后会重试。', 'Cached releases remain available; retry when online.')}</span><details><summary>{tr('详情', 'Details')}</summary>{status.error}</details></div>}
    {!isStudio() && <p className="version-note">{tr('浏览器可查看版本与说明；安装、切换和回退请在桌面应用中操作。', 'Browse releases and notes here; install, switch, or roll back in the desktop app.')}</p>}
    <div className="update-manager-grid">
      <section className="versions-release-panel">
        <div className="version-section-heading"><h2>Release</h2><span>{releases.length} {tr('个版本', 'versions')}</span></div>
        <div className="release-version-list">
          {releases.map((release, index) => {
            const ready = installed.has(release.version);
            const current = currentRelease && release.version === currentVersion;
            const active = busy === 'download:' + release.tag || busy === 'install:' + release.version;
            const releaseProgress = progress?.tag === release.tag ? progress : null;
            return <article className="release-version" key={release.tag}>
              <div className="release-version-heading"><strong>v{release.version}</strong>
                {index === 0 && <span className="version-badge">{tr('最新', 'Latest')}</span>}
                {current && <span className="version-badge">{tr('当前', 'Current')}</span>}
                <time>{release.published_at ? new Date(release.published_at).toLocaleDateString() : '—'}</time>
                {isStudio() && !current && (ready
                  ? <button disabled={busy !== null || !status?.platform_supported} onClick={() => void install(release.version, tr)} type="button">{active ? tr('准备中', 'Preparing') : olderThan(release.version, currentVersion) ? tr('回退', 'Roll back') : tr('安装并重启', 'Install & restart')}</button>
                  : release.asset ? <button disabled={busy !== null || !status?.platform_supported} onClick={() => void download(release.tag)} type="button"><Icon name="download" size={14} />{active ? progressButtonLabel(releaseProgress, tr) : tr('下载', 'Download')}</button> : <span className="version-note">{tr('暂无本平台安装包', 'No installer for this platform')}</span>)}
                <button aria-label={tr('查看 v' + release.version + ' Release', 'View v' + release.version + ' Release')} className="version-link" onClick={() => open(release.page_url)} type="button"><Icon name="link" size={15} /></button>
              </div>
              <p className="release-version-name">{release.name}{release.asset ? ' · ' + formatBytes(release.asset.byte_size) : ''}</p>
              {releaseProgress && <div className="version-download-progress"><span>{progressLabel(releaseProgress, tr)}</span><progress aria-label={tr('更新下载进度', 'Update download progress')} max={Math.max(1, releaseProgress.total_bytes)} value={releaseProgress.received_bytes} /></div>}
              <details className="release-notes-disclosure" open={index === 0}><summary>{tr('版本说明', 'Version notes')}</summary><ReleaseNotes notes={release.notes} tr={tr} /></details>
            </article>;
          })}
          {!releases.length && <div className="update-empty">{status ? tr('暂无 Release 信息，联网后检查更新。', 'No Release information yet. Check for updates when online.') : tr('正在读取版本信息…', 'Loading release information…')}</div>}
        </div>
      </section>
      <section className="versions-local-panel">
        <div className="version-section-heading"><h2>{tr('本地版本', 'Local versions')}</h2></div>
        <div className="installed-version-list">
          <div className="local-version current"><div><strong title={currentVersion}>v{currentVersion}</strong><small>{tr('正在运行', 'Running')} · {currentRelease ? 'Release' : tr('实验版本', 'Experimental')}</small></div><Icon name="check" size={16} /></div>
          {local.map(localRow)}
          {isStudio() && !local.length && <p className="version-note">{tr('下载后可切换；安装时保留当前版本供回退。', 'Downloaded versions can be switched to; the current version is retained on installation for rollback.')}</p>}
        </div>
        <details className="experimental-versions"><summary>{tr('实验版本', 'Experimental versions')}</summary>
          <p className="version-note">{tr('非 Release 版本需额外编译，不参与自动更新。', 'Non-Release versions require a separate build and do not participate in automatic updates.')}</p>
          {experimental.map(localRow)}
          <button className="release-page-link" onClick={() => open('https://github.com/Tylogi/TyloQuant')} type="button">{tr('查看源码', 'View source')}<Icon name="link" size={14} /></button>
        </details>
      </section>
    </div>
    <button className="release-page-link" onClick={() => open(status?.releases_page || 'https://github.com/Tylogi/TyloQuant/releases')} type="button">{tr('查看全部 Release', 'View all Releases')}<Icon name="link" size={14} /></button>
  </section>;
}
