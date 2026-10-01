/** 封装 Tauri 桌面命令与浏览器回退，隔离平台运行时差异。 */
export type StudioRuntimeMode = 'local' | 'remote';

/** 桌面运行时连接方式及本地服务端口配置。 */
export interface StudioConfig {
  mode: StudioRuntimeMode;
  remote_url: string;
  local_service_port: number;
}

/** 平台返回的连接状态与受管理服务进程信息。 */
export interface StudioStatus {
  config: StudioConfig;
  service_url: string;
  reachable: boolean;
  managed_pid: number | null;
}

/** GitHub Release 中与当前平台严格匹配的可安装资产。 */
export interface StudioReleaseAsset {
  name: string;
  byte_size: number;
  sha256?: string | null;
  download_url: string;
}

/** Studio 可安装版本及其已验证发布元数据。 */
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

/** 本地缓存或当前运行中的 Studio 版本。 */
export interface InstalledStudioVersion {
  version: string;
  tag: string;
  current: boolean;
  ready: boolean;
  byte_size: number;
  installed_at_epoch_seconds: number;
}

/** 桌面更新检查、缓存版本和平台能力的统一快照。 */
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

/** 当前更新下载或安装准备阶段的进度。 */
export interface StudioUpdateProgress {
  tag: string;
  stage: 'downloading' | 'preparing' | string;
  received_bytes: number;
  total_bytes: number;
}

interface TauriInternals {
  /** 调用桌面壳注册的命令并返回反序列化结果。 */
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

/** 判断当前页面是否运行在具有命令桥接的 Studio 桌面壳中。 */
export function isStudio(): boolean {
  return internals() !== null;
}

/** 查询桌面服务状态；浏览器环境没有本地管理服务，返回 null。 */
export async function studioStatus(): Promise<StudioStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioStatus>('studio_status') : null;
}

/** 保存桌面连接配置并返回新状态；浏览器环境拒绝调用。 */
export async function configureStudio(config: StudioConfig): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_configure', { config });
}

/** 请求桌面壳启动本地服务，返回服务状态；浏览器环境拒绝调用。 */
export async function startLocalStudio(): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioStatus>('studio_start_local');
}

/** 打开桌面原生模型目录选择器，取消时返回 null。 */
export async function selectLocalModelDirectory(): Promise<string[] | null> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<string[] | null>('studio_select_model_directory');
}

/** 使用当前平台的确认对话框，浏览器环境回退到 window.confirm。 */
export async function studioConfirm(message: string): Promise<boolean> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<boolean>('studio_confirm', { message })
    : window.confirm(message);
}

/** 从桌面凭据存储读取令牌；浏览器环境返回空字符串。 */
export async function studioCredential(): Promise<string> {
  const tauri = internals();
  return tauri ? tauri.invoke<string>('studio_credential_get') : '';
}

/** 将令牌写入桌面凭据存储；浏览器环境不执行持久化。 */
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

/** 只允许桌面壳通过受信任主机白名单打开外部链接。 */
export async function openStudioExternal(url: string): Promise<void> {
  const trusted = trustedExternalUrl(url);
  const tauri = internals();
  if (tauri) {
    await tauri.invoke<void>('studio_open_external', { url: trusted });
    return;
  }
  window.open(trusted, '_blank', 'noopener,noreferrer');
}

/** 读取更新状态；force 为真时绕过六小时缓存。 */
export async function studioUpdateStatus(force = false): Promise<StudioUpdateStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioUpdateStatus>('studio_update_status', { force }) : null;
}

/** 读取正在进行的更新下载进度。 */
export async function studioUpdateProgress(): Promise<StudioUpdateProgress | null> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<StudioUpdateProgress | null>('studio_update_progress')
    : null;
}

/** 修改后台自动检查偏好并返回最新状态。 */
export async function setAutomaticStudioUpdates(
  enabled: boolean,
): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_set_automatic', { enabled });
}

/** 下载、校验并缓存指定 Release。 */
export async function downloadStudioVersion(tag: string): Promise<InstalledStudioVersion> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<InstalledStudioVersion>('studio_update_download', { tag });
}

/** 切换到指定已校验版本并重启桌面应用。 */
export async function installStudioVersion(version: string): Promise<void> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  await tauri.invoke<void>('studio_update_install', { version });
}

/** 删除指定本地版本缓存并返回最新状态。 */
export async function deleteStudioVersion(version: string): Promise<StudioUpdateStatus> {
  const tauri = internals();
  if (!tauri) throw new Error('MFQ Studio runtime is unavailable');
  return tauri.invoke<StudioUpdateStatus>('studio_update_delete', { version });
}
