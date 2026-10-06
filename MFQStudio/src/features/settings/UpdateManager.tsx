/** Manage desktop update checks, cached versions, and installation actions. */
import { i18n } from '../../i18n';
import type { TFunction } from 'i18next';
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

type Translate = TFunction;

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

function progressLabel(progress: StudioUpdateProgress, t: Translate): string {
  if (progress.stage === "preparing") return t('settings:updateManager.verifyingPreparing');
  if (progress.total_bytes <= 0) return t('settings:updateManager.downloading');
  const percent = Math.min(100, Math.round((progress.received_bytes / progress.total_bytes) * 100));
  return `${percent}% · ${formatBytes(progress.received_bytes)} / ${formatBytes(progress.total_bytes)}`;
}

function progressButtonLabel(progress: StudioUpdateProgress | null, t: Translate): string {
  if (!progress || progress.total_bytes <= 0) return t('settings:updateManager.downloading');
  if (progress.stage === "preparing") return t('settings:updateManager.preparing');
  return `${Math.min(100, Math.round((progress.received_bytes / progress.total_bytes) * 100))}%`;
}

/** Track desktop update status and expose download, installation, and removal actions. */
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

  async function install(version: string, t: Translate) {
    const confirmed = await studioConfirm(t('settings:updateManager.switchToMfqStudioTheCurrentVersionWillBeRetainedAndThe', { version: version }));
    if (!confirmed) return;
    setBusy(`install:${version}`);
    try {
      await installStudioVersion(version);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
    }
  }

  async function remove(version: string, t: Translate) {
    const confirmed = await studioConfirm(t('settings:updateManager.removeTheCachedMfqStudio', { version: version }));
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

/** Open update settings when a newer desktop release is available. */
export function UpdateAvailableBanner({ onOpen, status, t }: {
  onOpen(): void;
  status: StudioUpdateStatus | null;
  t: Translate;
}) {
  if (!status?.update_available || !status.latest) return null;
  return (
    <button className="update-available-banner" onClick={onOpen} type="button">
      <span>↑</span>
      <div><strong>{t('settings:updateManager.mfqStudioIsAvailable', { version: status.latest.version })}</strong><small>{t('settings:updateManager.viewUpdateAndVersionOptions')}</small></div>
    </button>
  );
}

/** Manage desktop update checks, cached versions, and installation actions. */
export function UpdateManager({
  busy,
  download,
  install,
  progress,
  refresh,
  remove,
  setAutomatic,
  status,
  t,
}: ReturnType<typeof useStudioUpdates> & { t: Translate }) {
  const installed = useMemo(
    () => new Set(status?.installed_versions.filter((item) => item.ready).map((item) => item.version) ?? []),
    [status],
  );
  if (!isStudio()) return null;
  return (
    <section className="update-manager">
      <div className="update-manager-heading">
        <div><h3>{t('settings:updateManager.applicationUpdates')}</h3><p>{t('settings:updateManager.checkGithubReleasesAutomaticallyVerifyDownloadsAndRetainTheCurrentVersionBefore')}</p></div>
        <button disabled={busy !== null} onClick={() => void refresh()} type="button">{busy === "check" ? t('settings:updateManager.checking') : t('settings:updateManager.checkNow')}</button>
      </div>
      {status ? (
        <>
          <div className="update-current-row">
            <div><span>{t('settings:updateManager.currentVersion')}</span><strong>MFQ Studio {status.current_version}</strong><small>{status.checked_at_epoch_seconds ? t('settings:updateManager.lastChecked', { date: new Date(status.checked_at_epoch_seconds * 1000).toLocaleString(i18n.resolvedLanguage) }) : t('settings:updateManager.notCheckedYet')}</small></div>
            <label><span>{t('settings:updateManager.automaticallyCheckAndNotify')}</span><input checked={status.automatic_check} disabled={busy !== null} onChange={(event) => void setAutomatic(event.target.checked)} type="checkbox" /></label>
          </div>
          {status.error && <p className="update-error">{status.error}</p>}
          {!status.platform_supported && <p className="update-error">{t('settings:updateManager.thisPlatformCanBrowseReleasesButDoesNotSupportInAppInstallation')}</p>}
          <div className="update-manager-grid">
            <div>
              <h4>{t('settings:updateManager.availableReleases')}</h4>
              <div className="release-version-list">
                {status.releases.map((release) => {
                  const ready = installed.has(release.version);
                  const current = release.version === status.current_version;
                  const active = busy === `download:${release.tag}` || busy === `install:${release.version}`;
                  const releaseProgress = progress?.tag === release.tag ? progress : null;
                  return <div className={status.latest?.tag === release.tag ? "latest" : ""} key={release.tag}><div><strong>{release.version}{status.latest?.tag === release.tag ? ` · ${t('settings:updateManager.latest')}` : ""}</strong><small>{releaseProgress ? progressLabel(releaseProgress, t) : `${release.name} · ${formatBytes(release.asset.byte_size)}`}</small>{releaseProgress && <progress aria-label={t('settings:updateManager.updateDownloadProgress')} max={Math.max(1, releaseProgress.total_bytes)} value={releaseProgress.received_bytes} />}</div>{current ? <span>{t('settings:updateManager.current')}</span> : ready ? <button disabled={busy !== null || !status.platform_supported} onClick={() => void install(release.version, t)} type="button">{active ? t('settings:updateManager.preparing2') : olderThan(release.version, status.current_version) ? t('settings:updateManager.rollBack') : t('settings:updateManager.installRestart')}</button> : <button disabled={busy !== null || !status.platform_supported} onClick={() => void download(release.tag)} type="button">{active ? progressButtonLabel(releaseProgress, t) : t('settings:updateManager.download')}</button>}</div>;
                })}
                {!status.releases.length && <div className="update-empty">{t('settings:updateManager.noReleaseMetadataIsCached')}</div>}
              </div>
            </div>
            <div>
              <h4>{t('settings:updateManager.localVersions')}</h4>
              <div className="installed-version-list">
                {status.installed_versions.map((version) => <div key={`${version.version}:${version.current}`}><div><strong>{version.version}</strong><small>{version.current ? t('settings:updateManager.running') : version.ready ? t('settings:updateManager.verified', { size: formatBytes(version.byte_size) }) : t('settings:updateManager.incompleteCache')}</small></div>{version.current ? <span>{t('settings:updateManager.current')}</span> : <><button disabled={busy !== null || !version.ready} onClick={() => void install(version.version, t)} type="button">{olderThan(version.version, status.current_version) ? t('settings:updateManager.rollBack') : t('settings:updateManager.switch')}</button><button aria-label={t('settings:updateManager.removeCachedVersion')} className="version-remove" disabled={busy !== null} onClick={() => void remove(version.version, t)} type="button">×</button></>}</div>)}
              </div>
            </div>
          </div>
          <button className="release-page-link" onClick={() => void openStudioExternal(status.releases_page).catch(() => undefined)} type="button">{t('settings:updateManager.viewAllReleasesOnGithub')}</button>
        </>
      ) : <div className="update-empty">{t('settings:updateManager.loadingVersionStatus')}</div>}
    </section>
  );
}
