/** 验证聊天工具选择跨路由保留，并在切换服务时拒绝旧工具请求回写。 */
import { act, renderHook, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { McpToolResource } from '../../shared/api/types';
import { ChatToolsProvider, useChatTools } from './ChatToolsProvider';
import type { ReactNode } from 'react';

const runtime = vi.hoisted(() => ({ ready: true, connectionRevision: 1 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));

beforeEach(() => {
  runtime.ready = true;
  runtime.connectionRevision = 1;
  vi.restoreAllMocks();
});

it('服务切换时清空工具选择，丢弃旧连接迟到的工具列表', async () => {
  let resolveOld!: (value: { data: McpToolResource[] }) => void;
  const oldTools = new Promise<{ data: McpToolResource[] }>((resolve) => {
    resolveOld = resolve;
  });
  const fresh = { qualified_name: 'new/tool' } as McpToolResource;
  vi.spyOn(connectionsApi, 'mcpTools')
    .mockImplementationOnce(() => oldTools as ReturnType<typeof connectionsApi.mcpTools>)
    .mockResolvedValue({ data: [fresh] } as Awaited<ReturnType<typeof connectionsApi.mcpTools>>);
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter initialEntries={['/chat']}>
      <ChatToolsProvider>{children}</ChatToolsProvider>
    </MemoryRouter>
  );
  const { result, rerender } = renderHook(() => useChatTools(), { wrapper });
  act(() => result.current.setSelectedTools(['old/tool']));

  runtime.connectionRevision = 2;
  rerender();
  await waitFor(() => expect(result.current.mcpTools).toEqual([fresh]));
  expect(result.current.selectedTools).toEqual([]);

  await act(async () => resolveOld({ data: [{ qualified_name: 'old/tool' } as McpToolResource] }));
  expect(result.current.mcpTools).toEqual([fresh]);
});
