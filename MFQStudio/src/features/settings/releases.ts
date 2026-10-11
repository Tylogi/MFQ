import { version } from '../../../package.json';
import type { StudioRelease, StudioUpdateStatus } from '../../shared/platform/studio';

declare const __MFQ_STUDIO_BUILD__: { version: string; release: boolean };
export const studioBuild = typeof __MFQ_STUDIO_BUILD__ === 'undefined'
  ? { version: `${version}+dev.local`, release: false } : __MFQ_STUDIO_BUILD__;
export const releasesPage = 'https://github.com/Tylogi/TyloQuant/releases';
export const releasesUrl = 'https://api.github.com/repos/Tylogi/TyloQuant/releases?per_page=100';
const cacheKey = 'mfq-studio-release-cache';
const preferenceKey = 'mfq-studio-release-check';
const cacheSeconds = 6 * 60 * 60;

export function compareVersions(left: string, right: string): number {
  const parts = (value: string) => value.match(/\d+\.\d+\.\d+/)?.[0].split('.').map(Number) || [0, 0, 0];
  const a = parts(left), b = parts(right);
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

export function isReleaseTag(tag: string): boolean {
  return /^v?\d+\.\d+\.\d+$/i.test(tag) || /^studio-v\d+\.\d+\.\d+-multimodal-\d{8}$/.test(tag);
}

export function releaseCatalog(payload: unknown): StudioRelease[] {
  if (!Array.isArray(payload)) throw new Error('Invalid GitHub Release response');
  const values: StudioRelease[] = [];
  for (const item of payload) {
    if (!item || item.draft || item.prerelease || typeof item.tag_name !== 'string' || !isReleaseTag(item.tag_name)) continue;
    const version = item.tag_name.match(/\d+\.\d+\.\d+/)?.[0];
    if (!version) continue;
    values.push({ version, tag: item.tag_name, name: typeof item.name === 'string' ? item.name : item.tag_name,
      notes: typeof item.body === 'string' ? item.body : '', published_at: item.published_at,
      page_url: `${releasesPage}/tag/${encodeURIComponent(item.tag_name)}`, prerelease: false, asset: null });
  }
  return values.sort((a, b) => compareVersions(b.version, a.version))
    .filter((item, index, all) => !index || all[index - 1].version !== item.version);
}

export function cachedBrowserStatus(): StudioUpdateStatus {
  let cache: { checked: number; releases: StudioRelease[] } | null = null;
  let automatic = true;
  try {
    cache = JSON.parse(localStorage.getItem(cacheKey) || 'null');
    automatic = localStorage.getItem(preferenceKey) !== 'false';
  } catch {}
  const releases = Array.isArray(cache?.releases) ? cache.releases.filter(item => item && typeof item.tag === 'string' && typeof item.notes === 'string' && !item.prerelease && isReleaseTag(item.tag)) : [];
  const latest = releases[0] || null;
  return { current_version: studioBuild.version, current_release: studioBuild.release,
    automatic_check: automatic, automatic_download: false, checked_at_epoch_seconds: cache?.checked || null,
    update_available: Boolean(latest && compareVersions(latest.version, studioBuild.version) > 0),
    platform_supported: false, releases_page: releasesPage, latest, releases,
    installed_versions: [{ version: studioBuild.version, tag: studioBuild.version, current: true,
      ready: true, byte_size: 0, installed_at_epoch_seconds: 0, prerelease: !studioBuild.release }] };
}

export async function browserUpdateStatus(force: boolean): Promise<StudioUpdateStatus> {
  const cached = cachedBrowserStatus();
  if (!force && (!cached.automatic_check || Date.now() / 1000 - (cached.checked_at_epoch_seconds || 0) < cacheSeconds)) return cached;
  try {
    const response = await fetch(releasesUrl, { signal: AbortSignal.timeout(20_000), headers: { Accept: 'application/vnd.github+json' } });
    if (!response.ok) throw new Error(`GitHub Release: HTTP ${response.status}`);
    const releases = releaseCatalog(await response.json());
    const checked = Math.floor(Date.now() / 1000);
    try { localStorage.setItem(cacheKey, JSON.stringify({ checked, releases })); } catch {}
    return { ...cached, releases, latest: releases[0] || null, checked_at_epoch_seconds: checked,
      update_available: Boolean(releases[0] && compareVersions(releases[0].version, studioBuild.version) > 0) };
  } catch (cause) {
    return { ...cached, error: cause instanceof Error ? cause.message : String(cause) };
  }
}

export function setBrowserAutomatic(enabled: boolean): StudioUpdateStatus {
  try { localStorage.setItem(preferenceKey, String(enabled)); } catch {}
  return { ...cachedBrowserStatus(), automatic_check: enabled };
}
