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
