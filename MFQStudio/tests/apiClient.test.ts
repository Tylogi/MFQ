/** Verify the transport preserves standard HeadersInit and injects the current service credential only in memory. */
import { afterEach, expect, it, vi } from 'vitest';
import { request, setApiBaseUrl, setApiToken } from '../src/shared/api/client';

afterEach(() => {
  setApiBaseUrl('');
  setApiToken('');
});

it('verifies apiClient test behavior 1', async () => {
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
