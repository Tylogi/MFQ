/** Wrap Tauri desktop commands and browser fallbacks to isolate platform runtime differences. */
export type StudioRuntimeMode = 'local' | 'remote';

/** Desktop runtime connection mode and local server port configuration. */
export interface StudioConfig {
  mode: StudioRuntimeMode;
  remote_url: string;
  local_service_port: number;
}

/** Platform-provided connection status and managed server process information. */
export interface StudioStatus {
  config: StudioConfig;
  service_url: string;
  reachable: boolean;
  managed_pid: number | null;
}

/** Installable GitHub release asset that exactly matches the current platform. */
export interface StudioReleaseAsset {
  name: string;
  byte_size: number;
  sha256?: string | null;
  download_url: string;
}

/** Installable Studio version and its verified release metadata. */
export interface StudioRelease {
  version: string;
  tag: string;
  name: string;
  notes: string;
  published_at?: string | null;
  page_url: string;
  prerelease: boolean;
  asset: StudioReleaseAsset;
}

/** Locally cached or currently running Studio version. */
export interface InstalledStudioVersion {
  version: string;
  tag: string;
  current: boolean;
  ready: boolean;
  byte_size: number;
  installed_at_epoch_seconds: number;
}

/** Unified snapshot of desktop update checks, cached versions, and platform capabilities. */
export interface StudioUpdateStatus {
  current_version: string;
  automatic_check: boolean;
  checked_at_epoch_seconds?: number | null;
  update_available: boolean;
  platform_supported: boolean;
  releases_page: string;
  latest?: StudioRelease | null;
  releases: StudioRelease[];
  installed_versions: InstalledStudioVersion[];
  error?: string | null;
}

/** Progress of the current update download or installation preparation stage. */
export interface StudioUpdateProgress {
  tag: string;
  stage: 'downloading' | 'preparing' | string;
  received_bytes: number;
  total_bytes: number;
}

interface TauriInternals {
  /** Invoke a command registered by the desktop shell and return its deserialized result. */
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: TauriInternals;
  }
}

function internals(): TauriInternals | null {
  return window.__TAURI_INTERNALS__ ?? null;
}

/** Determine whether the current page is running in the Studio desktop shell with command bridging. */
export function isStudio(): boolean {
  return internals() !== null;
}

/** Query desktop service status; return null in browsers, which have no locally managed service. */
export async function studioStatus(): Promise<StudioStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioStatus>('studio_status') : null;
}

/** Save desktop connection settings and return the updated status; reject calls in browsers. */
export async function configureStudio(config: StudioConfig): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_configure', { config });
}

/** Ask the desktop shell to start the local service and return its status; reject calls in browsers. */
export async function startLocalStudio(): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_start_local');
}

/** Open the native desktop model-directory picker, returning null if cancelled. */
export async function selectLocalModelDirectory(): Promise<string[] | null> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<string[] | null>('studio_select_model_directory');
}

/** Use the current platform's confirmation dialog, falling back to window.confirm in browsers. */
export async function studioConfirm(message: string): Promise<boolean> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<boolean>('studio_confirm', { message })
    : window.confirm(message);
}

/** Read the token from desktop credential storage; return an empty string in browsers. */
export async function studioCredential(): Promise<string> {
  const tauri = internals();
  return tauri ? tauri.invoke<string>('studio_credential_get') : '';
}

/** Write the token to desktop credential storage; do not persist it in browsers. */
export async function saveStudioCredential(token: string): Promise<void> {
  const tauri = internals();
  if (!tauri) return;
  await tauri.invoke<void>('studio_credential_set', { token });
}

function trustedExternalUrl(value: string): string {
  const parsed = new URL(value);
  const trustedHosts = new Set([
    'github.com',
    'www.github.com',
    'huggingface.co',
    'www.huggingface.co',
    'modelscope.cn',
    'www.modelscope.cn',
  ]);
  if (
    parsed.protocol !== 'https:' ||
    parsed.username ||
    parsed.password ||
    (parsed.port && parsed.port !== '443') ||
    !trustedHosts.has(parsed.hostname.toLowerCase())
  ) {
    throw new Error('External links are limited to trusted MFQ model and release hosts');
  }
  return parsed.toString();
}

/** Open external links through the desktop shell only when their hosts are on the trusted allowlist. */
export async function openStudioExternal(url: string): Promise<void> {
  const trusted = trustedExternalUrl(url);
  const tauri = internals();
  if (tauri) {
    await tauri.invoke<void>('studio_open_external', { url: trusted });
    return;
  }
  window.open(trusted, '_blank', 'noopener,noreferrer');
}

/** Read update status, bypassing the six-hour cache when force is true. */
export async function studioUpdateStatus(force = false): Promise<StudioUpdateStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioUpdateStatus>('studio_update_status', { force }) : null;
}

/** Read the progress of an ongoing update download. */
export async function studioUpdateProgress(): Promise<StudioUpdateProgress | null> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<StudioUpdateProgress | null>('studio_update_progress')
    : null;
}

/** Change the automatic background update check preference and return the latest status. */
export async function setAutomaticStudioUpdates(
  enabled: boolean,
): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_set_automatic', { enabled });
}

/** Download, verify, and cache the specified release. */
export async function downloadStudioVersion(tag: string): Promise<InstalledStudioVersion> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<InstalledStudioVersion>('studio_update_download', { tag });
}

/** Switch to the specified verified version and restart the desktop app. */
export async function installStudioVersion(version: string): Promise<void> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  await tauri.invoke<void>('studio_update_install', { version });
}

/** Delete the specified locally cached version and return the latest status. */
export async function deleteStudioVersion(version: string): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_delete', { version });
}
