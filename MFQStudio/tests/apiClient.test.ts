/** 验证传输层保留标准 HeadersInit 并仅在内存中注入当前服务凭据。 */
import { afterEach, expect, it, vi } from 'vitest';
import { request, setApiBaseUrl, setApiToken } from '../src/shared/api/client';

afterEach(() => {
  setApiBaseUrl('');
  setApiToken('');
});

it('支持 Headers 实例和 AbortSignal，保留业务头并添加鉴权', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true })));
  vi.stubGlobal('fetch', fetchMock);
  setApiBaseUrl('https://test.invalid/');
  setApiToken('test-token');
  const headers = new Headers({ 'X-Request-ID': 'request-a' });
  const controller = new AbortController();
  await expect(request('/api/v1/status', { headers, signal: controller.signal })).resolves.toEqual({
    ok: true,
  });
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe('https://test.invalid/api/v1/status');
  expect(init.signal).toBe(controller.signal);
  expect(init.headers.get('X-Request-ID')).toBe('request-a');
  expect(init.headers.get('Authorization')).toBe('Bearer test-token');
  expect(headers.has('Authorization')).toBe(false);
  expect(localStorage.length).toBe(0);
});
