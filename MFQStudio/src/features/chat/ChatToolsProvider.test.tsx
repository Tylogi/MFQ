/** Verify that chat-tool selections persist across routes and stale tool requests cannot write back after a service switch. */
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

it('verifies ChatToolsProvider test behavior 1', async () => {
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
