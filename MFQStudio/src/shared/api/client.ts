/** 统一 HTTP 请求、服务地址与内存鉴权，供各资源 API 复用。 */
import type { ApiErrorBody } from './types';
/** 保留 HTTP 状态与服务错误码，供业务决定展示和恢复策略。 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = body.error.code;
    this.retryable = body.error.retryable;
  }
}

let apiBaseUrl = '';
let apiToken = '';

/** 读取当前内存中的凭据，仅供请求层构造鉴权头。 */
export function getApiToken(): string {
  return apiToken;
}

/** 切换服务地址，移除末尾斜杠；凭据由单独入口设置。 */
export function setApiBaseUrl(value: string): void {
  apiBaseUrl = value.trim().replace(/\/+$/, '');
}

/** 更新内存凭据，后续 HTTP 与实时连接使用新值，不写入浏览器存储。 */
export function setApiToken(value: string): void {
  apiToken = value.trim();
}

/** 读取规范化的服务地址，空值表示当前页面同源服务。 */
export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

/** 将资源路径拼接到当前服务地址。 */
export function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

/** 构造浏览器 WebSocket 音频入口，并附带服务要求的连接凭据。 */
export function runtimeRealtimeUrl(): string {
  const base = apiBaseUrl || window.location.origin;
  const url = new URL('/api/v1/runtime/realtime?mode=audio', base);
  if (apiToken) url.searchParams.set('access_token', apiToken);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

/** 读取服务错误体；非 JSON 或格式无效时回退为 HTTP 状态错误。 */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  try {
    return new ApiError(response.status, (await response.json()) as ApiErrorBody);
  } catch {
    return new ApiError(response.status, {
      error: {
        code: `http_${response.status}`,
        message: response.statusText || 'Request failed',
        retryable: false,
        details: {},
      },
    });
  }
}

/** 发起可取消的 JSON 请求，保留调用方 Headers 语义并统一处理失败状态。 */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = authorizedHeaders(init?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
  });
  if (!response.ok) throw await errorFromResponse(response);
  return (await response.json()) as T;
}

/** 基于任意合法 HeadersInit 创建请求头，并注入当前服务凭据。 */
export function authorizedHeaders(headers?: HeadersInit): Headers {
  const result = new Headers(headers);
  if (apiToken) result.set('Authorization', `Bearer ${apiToken}`);
  return result;
}
