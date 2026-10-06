/** Verify the instance-reload request path, context parameters, authentication, and error propagation. */
import { afterEach, expect, it, vi } from 'vitest';
import { runtimeApi } from '../src/shared/api/resources/runtime';
import { setApiBaseUrl, setApiToken } from '../src/shared/api/client';

afterEach(() => {
  setApiBaseUrl('');
  setApiToken('');
});

it('verifies runtimeApi test behavior 1', async () => {
  const response = { instance_id: 'instance-a', context_size: 8192 };
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response)));
  vi.stubGlobal('fetch', fetchMock);
  setApiBaseUrl('https://test.invalid');
  setApiToken('test-token');
  await expect(runtimeApi.reloadRuntime(8192, 'instance-a')).resolves.toEqual(response);
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe('https://test.invalid/api/v1/runtime/reload');
  expect(init.method).toBe('POST');
  expect(JSON.parse(init.body)).toEqual({ context_size: 8192, instance_id: 'instance-a' });
  expect(init.headers.get('Authorization')).toBe('Bearer test-token');
});

it('verifies runtimeApi test behavior 2', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response('invalid context', { status: 400 }));
  vi.stubGlobal('fetch', fetchMock);
  await expect(runtimeApi.reloadRuntime(0)).rejects.toThrow();
  expect(fetchMock).toHaveBeenCalledOnce();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ context_size: 0 });
});

it('encodes history cursors and forwards cancellation without changing incremental log defaults', async () => {
  const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ data: [] })));
  vi.stubGlobal('fetch', fetchMock);
  const signal = new AbortController().signal;
  await runtimeApi.runtimeLogPage(50, { before: 123, order: 'desc' }, signal);
  expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/runtime/logs?limit=50&order=desc&before=123');
  expect(fetchMock.mock.calls[0][1].signal).toBe(signal);
  await runtimeApi.runtimeRequestPage(50, { after: 321, order: 'asc' }, signal);
  expect(fetchMock.mock.calls[1][0]).toBe('/api/v1/runtime/requests?limit=50&order=asc&after=321');
  await runtimeApi.runtimeLogs(100, 10);
  expect(fetchMock.mock.calls[2][0]).toBe('/api/v1/runtime/logs?limit=100&after=10');
});


it.each(['logs', 'requests'] as const)('streams %s with authorization, cursor and cancellation', async (channel) => {
  const entry = { sequence: 43 };
  const fetchMock = vi.fn().mockResolvedValue(new Response(
    `: keep-alive\n\nid: 43\nevent: ${channel}\ndata: ${JSON.stringify(entry)}\n\n`,
    { headers: { 'Content-Type': 'text/event-stream' } },
  ));
  vi.stubGlobal('fetch', fetchMock);
  setApiToken('stream-token');
  const controller = new AbortController();
  const onEvent = vi.fn();
  const stream = channel === 'logs' ? runtimeApi.streamRuntimeLogs : runtimeApi.streamRuntimeRequests;
  await stream(42, onEvent, controller.signal);
  expect(fetchMock.mock.calls[0][0]).toBe(`/api/v1/runtime/${channel}/stream?after=42`);
  expect(fetchMock.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer stream-token');
  expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
  expect(onEvent).toHaveBeenCalledExactlyOnceWith(entry);
});
