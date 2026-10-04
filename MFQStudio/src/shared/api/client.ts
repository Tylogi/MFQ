/** Centralize HTTP requests, service URLs, and in-memory authentication for reuse by resource APIs. */
import type { ApiErrorBody } from './types';
/** Preserve HTTP status and server error codes so business logic can choose display and recovery strategies. */
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
const BROWSER_SERVICE_KEY = 'mfq.studio.service-url';

export function browserServiceUrl(): string {
  return localStorage.getItem(BROWSER_SERVICE_KEY) ?? '';
}

export function setBrowserServiceUrl(value: string): void {
  if (value) localStorage.setItem(BROWSER_SERVICE_KEY, value);
  else localStorage.removeItem(BROWSER_SERVICE_KEY);
}

/** Read the current in-memory credential for constructing authentication headers in the request layer. */
export function getApiToken(): string {
  return apiToken;
}

/** Change the service URL and remove trailing slashes; credentials are set through a separate entry point. */
export function setApiBaseUrl(value: string): void {
  apiBaseUrl = value.trim().replace(/\/+$/, '');
}

/** Update the in-memory credential for subsequent HTTP and real-time connections without writing it to browser storage. */
export function setApiToken(value: string): void {
  apiToken = value.trim();
}

/** Read the normalized service URL; an empty value means the current page's same-origin service. */
export function getApiBaseUrl(): string {
  return apiBaseUrl;
}

/** Append a resource path to the current service URL. */
export function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

/** Build the browser WebSocket audio endpoint and include the connection credential required by the service. */
export function runtimeRealtimeUrl(): string {
  const base = apiBaseUrl || window.location.origin;
  const url = new URL('/api/v1/runtime/realtime?mode=audio', base);
  if (apiToken) url.searchParams.set('access_token', apiToken);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

/** Read the server error body, falling back to an HTTP status error when it is not valid JSON. */
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

/** Make a cancellable JSON request, preserve caller-supplied Headers semantics, and handle failure statuses consistently. */
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

/** Create request headers from any valid HeadersInit and inject the current service credential. */
export function authorizedHeaders(headers?: HeadersInit): Headers {
  const result = new Headers(headers);
  if (apiToken) result.set('Authorization', `Bearer ${apiToken}`);
  return result;
}
