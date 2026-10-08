import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ToolsRoutingPanel } from './ToolsRoutingPanel';
import { setApiBaseUrl } from '../../shared/api/client';

const api = vi.hoisted(() => ({ servers: vi.fn(), tools: vi.fn(), nodes: vi.fn(), create: vi.fn(), error: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ready: true, connectionRevision: 0 }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/connections', () => ({ connectionsApi: {
  mcpServers: api.servers, mcpTools: api.tools, remoteNodes: api.nodes, createMcpServer: api.create,
} }));
vi.mock('../../stores/toastStore', () => ({ toast: { error: api.error } }));
beforeEach(() => {
  vi.clearAllMocks();
  setApiBaseUrl('');
  api.servers.mockResolvedValue([]);
  api.tools.mockResolvedValue({ data: [], errors: {} });
  api.nodes.mockResolvedValue([]);
  api.create.mockResolvedValue({ id: 'server-a' });
});
afterEach(() => { setApiBaseUrl(''); vi.restoreAllMocks(); });

async function submit() {
  await waitFor(() => expect(api.servers).toHaveBeenCalledOnce());
  fireEvent.change(screen.getByRole('textbox', { name: 'Server name' }), { target: { value: 'Test server' } });
  const url = screen.getByRole('textbox', { name: 'Streamable HTTP URL' });
  fireEvent.change(url, { target: { value: 'https://tools.invalid/mcp' } });
  fireEvent.submit(url.closest('form')!);
}

it('refreshes and notifies tool consumers after a current-service mutation', async () => {
  const event = vi.spyOn(window, 'dispatchEvent');
  render(<ToolsRoutingPanel />);
  await submit();
  await waitFor(() => expect(api.servers).toHaveBeenCalledTimes(2));
  expect(event.mock.calls.some(([value]) => value.type === 'mfq:tools-changed')).toBe(true);
  expect(screen.getByRole('textbox', { name: 'Server name' })).toHaveValue('');
});

it('does not refresh a different server after an old mutation response', async () => {
  let resolve!: (value: { id: string }) => void;
  api.create.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const event = vi.spyOn(window, 'dispatchEvent');
  render(<ToolsRoutingPanel />);
  await submit();
  setApiBaseUrl('https://second.invalid');
  await act(async () => { resolve({ id: 'server-a' }); });
  expect(api.servers).toHaveBeenCalledOnce();
  expect(api.nodes).toHaveBeenCalledOnce();
  expect(event.mock.calls.some(([value]) => value.type === 'mfq:tools-changed')).toBe(false);
});

it('does not dispatch stale tool updates when leaving during the post-write refresh', async () => {
  let resolve!: (value: never[]) => void;
  api.servers.mockResolvedValueOnce([]).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const event = vi.spyOn(window, 'dispatchEvent');
  const view = render(<ToolsRoutingPanel />);
  await submit();
  await waitFor(() => expect(api.servers).toHaveBeenCalledTimes(2));
  view.unmount();
  await act(async () => { resolve([]); });
  expect(event.mock.calls.some(([value]) => value.type === 'mfq:tools-changed')).toBe(false);
});

it('does not show an old mutation error after leaving the page', async () => {
  let reject!: (cause: Error) => void;
  api.create.mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  const view = render(<ToolsRoutingPanel />);
  await submit();
  view.unmount();
  await act(async () => { reject(new Error('Old server failed')); });
  expect(api.error).not.toHaveBeenCalled();
});
