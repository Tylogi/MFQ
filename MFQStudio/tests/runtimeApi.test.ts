/** 验证重载实例的请求路径、上下文参数、鉴权与错误传播。 */
import { afterEach, expect, it, vi } from 'vitest';
import { runtimeApi } from '../src/shared/api/resources/runtime';
import { setApiBaseUrl, setApiToken } from '../src/shared/api/client';

afterEach(() => {
  setApiBaseUrl('');
  setApiToken('');
});

it('重载使用指定实例与上下文，并保留服务响应', async () => {
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

it('默认实例不发送伪造标识，服务拒绝时上抛错误且不重试', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response('invalid context', { status: 400 }));
  vi.stubGlobal('fetch', fetchMock);
  await expect(runtimeApi.reloadRuntime(0)).rejects.toThrow();
  expect(fetchMock).toHaveBeenCalledOnce();
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ context_size: 0 });
});
