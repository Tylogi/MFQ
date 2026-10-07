/** 验证传输层保留标准 HeadersInit 并仅在内存中注入当前服务凭据。 */
import { afterEach, expect, it, vi } from 'vitest';
import { browserServiceUrl, request, setApiBaseUrl, setApiToken, setBrowserServiceUrl } from '../src/shared/api/client';
import { jobsApi } from '../src/shared/api/resources/jobs';

afterEach(() => {
  vi.restoreAllMocks();
  setBrowserServiceUrl('');
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

it('keeps browser connections usable when storage cannot be read or written', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('Blocked', 'SecurityError'); });
  expect(browserServiceUrl()).toBe('');
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('Full', 'QuotaExceededError'); });
  expect(() => setBrowserServiceUrl('http://127.0.0.1:8091')).not.toThrow();
  expect(browserServiceUrl()).toBe('http://127.0.0.1:8091');
  vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new DOMException('Blocked', 'SecurityError'); });
  expect(() => setBrowserServiceUrl('')).not.toThrow();
  expect(browserServiceUrl()).toBe('');
});

it('ignores invalid stored service URLs and credential-bearing URLs', () => {
  for (const value of ['not a URL', 'file:///private/model', 'https://user:password@example.invalid']) {
    localStorage.setItem('mfq.studio.service-url', value);
    expect(browserServiceUrl()).toBe('');
  }
  localStorage.setItem('mfq.studio.service-url', 'https://server.invalid');
  expect(browserServiceUrl()).toBe('https://server.invalid');
});

it('requests the latest bounded job-event window without changing forward SSE replay', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: [
    { sequence: 1, level: 'info', message: null, data: {}, created_at: '2026-10-06T00:00:00Z' },
    { sequence: 1005, level: 'info', message: 'Latest output', data: { phase: 'ready' }, created_at: '2026-10-06T00:00:01Z' },
  ] })));
  vi.stubGlobal('fetch', fetchMock);
  setApiBaseUrl('https://test.invalid');
  const controller = new AbortController();
  expect(await jobsApi.jobEvents('job-a', controller.signal)).toEqual([
    { sequence: 1005, level: 'info', message: 'Latest output', fields: { phase: 'ready' }, created_at: '2026-10-06T00:00:01Z' },
  ]);
  expect(fetchMock.mock.calls[0][0]).toBe('https://test.invalid/api/v1/jobs/job-a/events?limit=1000&tail=true');
  expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
});
