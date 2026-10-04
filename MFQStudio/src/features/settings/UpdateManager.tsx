import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

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
  const parse = (value: string) => value.split(".").map((item) => Number(item));
  const a = parse(left);
  const b = parse(right);
  for (let index = 0; index < 3; index += 1) {
    if ((a[index] || 0) !== (b[index] || 0)) return (a[index] || 0) < (b[index] || 0);
  }
  return false;
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
  const [status, setStatus] = useState<StudioUpdateStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<StudioUpdateProgress | null>(null);

  useEffect(() => {
    if (!isStudio()) return;
    let cancelled = false;
    const check = () => {
      void studioUpdateStatus(false)
        .then((value) => { if (!cancelled) setStatus(value); })
        .catch((cause) => { if (!cancelled) onError(cause instanceof Error ? cause.message : String(cause)); });
    };
    check();
    const timer = window.setInterval(check, 6 * 60 * 60 * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [onError]);

  async function refresh() {
    setBusy("check");
    try {
      setStatus(await studioUpdateStatus(true));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }

  async function setAutomatic(enabled: boolean) {
    setBusy("preference");
    try {
      setStatus(await setAutomaticStudioUpdates(enabled));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }

  async function download(tag: string) {
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
    }
  }

  async function install(version: string, tr: Translate) {
    const confirmed = await studioConfirm(tr(
      `切换到 MFQ Studio ${version}？应用会保留当前版本并重启，本地 MFQ Server 也会随之重启。`,
      `Switch to MFQ Studio ${version}? The current version will be retained, and the app and managed local MFQ Server will restart.`,
    ));
    if (!confirmed) return;
    setBusy(`install:${version}`);
    try {
      await installStudioVersion(version);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
    }
  }

  async function remove(version: string, tr: Translate) {
    const confirmed = await studioConfirm(tr(
      `移除已缓存的 MFQ Studio ${version}？`,
      `Remove the cached MFQ Studio ${version}?`,
    ));
    if (!confirmed) return;
    setBusy(`delete:${version}`);
    try {
      setStatus(await deleteStudioVersion(version));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }

  return { busy, download, install, progress, refresh, remove, setAutomatic, status };
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
};

const StudioUpdatesContext = createContext<StudioUpdates>(unavailableStudioUpdates);

/** Maintain a single shared state for update checks, download progress, and cached versions throughout the app lifecycle. */
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

/** Read shared Studio update state; callers must be within StudioUpdateProvider. */
export function useStudioUpdateContext(): StudioUpdates {
  return useContext(StudioUpdatesContext);
}

export function UpdateAvailableBanner({ onOpen, status, tr }: {
  onOpen(): void;
  status: StudioUpdateStatus | null;
  tr: Translate;
}) {
  if (!status?.update_available || !status.latest) return null;
  return (
    <button className="update-available-banner" onClick={onOpen} type="button">
      <span>↑</span>
      <div><strong>{tr(`MFQ Studio ${status.latest.version} 可用`, `MFQ Studio ${status.latest.version} is available`)}</strong><small>{tr("查看更新与版本管理", "View update and version options")}</small></div>
    </button>
  );
}

export function UpdateManager({
  busy,
  download,
  install,
  progress,
  refresh,
  remove,
  setAutomatic,
  status,
  tr,
}: ReturnType<typeof useStudioUpdates> & { tr: Translate }) {
  const installed = useMemo(
    () => new Set(status?.installed_versions.filter((item) => item.ready).map((item) => item.version) ?? []),
    [status],
  );
  if (!isStudio()) return null;
  return (
    <section className="update-manager">
      <div className="update-manager-heading">
        <div><h3>{tr("应用更新", "Application updates")}</h3><p>{tr("自动检查 GitHub Release；安装前校验摘要并保留当前版本。", "Check GitHub Releases automatically; verify downloads and retain the current version before installation.")}</p></div>
        <button disabled={busy !== null} onClick={() => void refresh()} type="button">{busy === "check" ? tr("检查中", "Checking") : tr("检查更新", "Check now")}</button>
      </div>
      {status ? (
        <>
          <div className="update-current-row">
            <div><span>{tr("当前版本", "Current version")}</span><strong>MFQ Studio {status.current_version}</strong><small>{status.checked_at_epoch_seconds ? tr(`上次检查 ${new Date(status.checked_at_epoch_seconds * 1000).toLocaleString()}`, `Last checked ${new Date(status.checked_at_epoch_seconds * 1000).toLocaleString()}`) : tr("尚未检查", "Not checked yet")}</small></div>
            <label><span>{tr("自动检查并提醒", "Automatically check and notify")}</span><input checked={status.automatic_check} disabled={busy !== null} onChange={(event) => void setAutomatic(event.target.checked)} type="checkbox" /></label>
          </div>
          {status.error && <p className="update-error">{status.error}</p>}
          {!status.platform_supported && <p className="update-error">{tr("当前平台只能查看 Release，暂不支持应用内安装。", "This platform can browse releases but does not support in-app installation yet.")}</p>}
          <div className="update-manager-grid">
            <div>
              <h4>{tr("可用版本", "Available releases")}</h4>
              <div className="release-version-list">
                {status.releases.map((release) => {
                  const ready = installed.has(release.version);
                  const current = release.version === status.current_version;
                  const active = busy === `download:${release.tag}` || busy === `install:${release.version}`;
                  const releaseProgress = progress?.tag === release.tag ? progress : null;
                  return <div className={status.latest?.tag === release.tag ? "latest" : ""} key={release.tag}><div><strong>{release.version}{status.latest?.tag === release.tag ? ` · ${tr("最新", "Latest")}` : ""}</strong><small>{releaseProgress ? progressLabel(releaseProgress, tr) : `${release.name} · ${formatBytes(release.asset.byte_size)}`}</small>{releaseProgress && <progress aria-label={tr("更新下载进度", "Update download progress")} max={Math.max(1, releaseProgress.total_bytes)} value={releaseProgress.received_bytes} />}</div>{current ? <span>{tr("当前", "Current")}</span> : ready ? <button disabled={busy !== null || !status.platform_supported} onClick={() => void install(release.version, tr)} type="button">{active ? tr("准备中", "Preparing") : olderThan(release.version, status.current_version) ? tr("回退", "Roll back") : tr("安装并重启", "Install & restart")}</button> : <button disabled={busy !== null || !status.platform_supported} onClick={() => void download(release.tag)} type="button">{active ? progressButtonLabel(releaseProgress, tr) : tr("下载", "Download")}</button>}</div>;
                })}
                {!status.releases.length && <div className="update-empty">{tr("没有缓存的 Release 信息。", "No release metadata is cached.")}</div>}
              </div>
            </div>
            <div>
              <h4>{tr("本地版本", "Local versions")}</h4>
              <div className="installed-version-list">
                {status.installed_versions.map((version) => <div key={`${version.version}:${version.current}`}><div><strong>{version.version}</strong><small>{version.current ? tr("正在运行", "Running") : version.ready ? tr(`已校验 · ${formatBytes(version.byte_size)}`, `Verified · ${formatBytes(version.byte_size)}`) : tr("缓存不完整", "Incomplete cache")}</small></div>{version.current ? <span>{tr("当前", "Current")}</span> : <><button disabled={busy !== null || !version.ready} onClick={() => void install(version.version, tr)} type="button">{olderThan(version.version, status.current_version) ? tr("回退", "Roll back") : tr("切换", "Switch")}</button><button aria-label={tr("移除缓存", "Remove cached version")} className="version-remove" disabled={busy !== null} onClick={() => void remove(version.version, tr)} type="button">×</button></>}</div>)}
              </div>
            </div>
          </div>
          <button className="release-page-link" onClick={() => void openStudioExternal(status.releases_page).catch(() => undefined)} type="button">{tr("在 GitHub 查看全部 Release", "View all releases on GitHub")}</button>
        </>
      ) : <div className="update-empty">{tr("正在读取版本状态…", "Loading version status…")}</div>}
    </section>
  );
}
