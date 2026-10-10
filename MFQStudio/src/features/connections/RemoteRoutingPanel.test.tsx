import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { RemoteNode } from '../../shared/api/types';
import { RemoteRoutingPanel } from './RemoteRoutingPanel';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ready: true, connectionRevision: 1 }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

const node: RemoteNode = { id: 'remote', name: 'Remote worker', url: 'https://worker.example',
  enabled: true, healthy: true, models: ['model'], active_requests: 0, metrics: {},
  created_at: '2026-10-10T00:00:00Z', updated_at: '2026-10-10T00:00:00Z' };

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(connectionsApi, 'remoteNodes').mockResolvedValue([]);
});

it('only discovers remote inference nodes and has no tool registry', async () => {
  render(<RemoteRoutingPanel />);
  await waitFor(() => expect(connectionsApi.remoteNodes).toHaveBeenCalledExactlyOnceWith(true));
  expect(screen.getByText('Remote routing')).toBeInTheDocument();
  expect(screen.queryByText('MCP')).not.toBeInTheDocument();
  expect(Object.keys(connectionsApi).sort()).toEqual(['createRemoteNode', 'deleteRemoteNode', 'remoteNodes']);
});

it('still registers a node and refreshes its health', async () => {
  const create = vi.spyOn(connectionsApi, 'createRemoteNode').mockResolvedValue(node);
  render(<RemoteRoutingPanel />);
  await waitFor(() => expect(connectionsApi.remoteNodes).toHaveBeenCalledOnce());
  fireEvent.change(screen.getByRole('textbox', { name: 'Node name' }), { target: { value: node.name } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Node URL' }), { target: { value: node.url } });
  fireEvent.click(screen.getByRole('button', { name: 'Add' }));
  await waitFor(() => expect(create).toHaveBeenCalledExactlyOnceWith({ name: node.name, url: node.url, api_key_env: null, enabled: true }));
  await waitFor(() => expect(connectionsApi.remoteNodes).toHaveBeenCalledTimes(2));
  expect(screen.getByRole('textbox', { name: 'Node name' })).toHaveValue('');
});

it('still removes registered remote nodes', async () => {
  vi.mocked(connectionsApi.remoteNodes).mockResolvedValueOnce([node]);
  const remove = vi.spyOn(connectionsApi, 'deleteRemoteNode').mockResolvedValue(undefined);
  render(<RemoteRoutingPanel />);
  expect(await screen.findByText(node.name)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
  await waitFor(() => expect(remove).toHaveBeenCalledExactlyOnceWith(node.id));
  await waitFor(() => expect(screen.queryByText(node.name)).not.toBeInTheDocument());
});
