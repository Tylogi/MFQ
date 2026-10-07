import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { CachePage } from './CachePage';
import type { RuntimeStatus } from '../../shared/api/contracts/runtime';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, connectionRevision: 0,
  refreshRuntime: vi.fn().mockResolvedValue(true), clearRuntimeCache: vi.fn().mockResolvedValue({}), confirm: vi.fn(),
  success: vi.fn(), error: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { clearRuntimeCache: state.clearRuntimeCache } }));
vi.mock('../../studio', () => ({ studioConfirm: state.confirm }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: state.success, error: state.error } }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./ResourceMonitorPanel', () => ({ ResourceMonitorPanel: () => <h2>Resource monitoring</h2> }));
vi.mock('./PrefixCacheInventoryPanel', () => ({ PrefixCacheInventoryPanel: () => <h2>Cache management</h2> }));
beforeEach(() => {
  vi.clearAllMocks();
  state.runtime = null;
  state.connectionRevision = 0;
  state.confirm.mockResolvedValue(true);
});

it('resources contains monitoring and cache but no service configuration', () => {
  state.runtime = null;
  render(<CachePage />);
  expect(screen.getByRole('heading', { name: 'Resource monitoring' })).toBeInTheDocument();
  expect(screen.queryByText('Runtime profiles')).not.toBeInTheDocument();
  expect(screen.queryByText('Tool servers and model-visible tools')).not.toBeInTheDocument();
});

it('shows persisted shared cache totals without a loaded model', () => {
  state.runtime = { runtime_state: 'idle', prefix_cache_total_disk_bytes: 2 ** 30,
    prefix_cache_total_disk_max_bytes: 10 * 2 ** 30, prefix_cache_total_disk_blocks: 17,
    prefix_cache_total_hot_bytes: 0, prefix_cache_total_hot_max_bytes: 2 * 2 ** 30 };
  render(<CachePage />);
  expect(screen.getByText('Total SSD prefix cache')).toBeInTheDocument();
  expect(screen.getByText('Total RAM hot cache')).toBeInTheDocument();
  expect(screen.getByText('17')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Clear this model’s prefix cache' })).not.toBeInTheDocument();
});

it('locks cache cleanup while confirmation is pending and only clears its model', async () => {
  state.runtime = { instance_id: 'model-a', prefix_cache_max_bytes: 1024, prefix_cache_snapshots: 2 };
  let confirm!: (value: boolean) => void;
  state.confirm.mockImplementation(() => new Promise<boolean>((resolve) => { confirm = resolve; }));
  render(<CachePage />);
  const button = screen.getByRole('button', { name: 'Clear this model’s prefix cache' });
  fireEvent.click(button);
  fireEvent.click(button);
  expect(state.confirm).toHaveBeenCalledTimes(1);
  expect(button).toBeDisabled();
  expect(state.clearRuntimeCache).not.toHaveBeenCalled();
  await act(async () => { confirm(true); });
  await waitFor(() => expect(state.success).toHaveBeenCalledTimes(1));
  expect(state.clearRuntimeCache).toHaveBeenCalledExactlyOnceWith('model-a');
  expect(state.refreshRuntime).toHaveBeenCalledExactlyOnceWith(false);
});

it('does not clear a different service after a delayed confirmation', async () => {
  state.runtime = { instance_id: 'model-a', prefix_cache_max_bytes: 1024, prefix_cache_snapshots: 2 };
  let confirm!: (value: boolean) => void;
  state.confirm.mockImplementation(() => new Promise<boolean>((resolve) => { confirm = resolve; }));
  const view = render(<CachePage />);
  fireEvent.click(screen.getByRole('button', { name: 'Clear this model’s prefix cache' }));
  state.connectionRevision = 1;
  view.rerender(<CachePage />);
  await act(async () => { confirm(true); });
  expect(state.clearRuntimeCache).not.toHaveBeenCalled();
  expect(state.success).not.toHaveBeenCalled();
});

it('unlocks the new server’s cache controls while an old confirmation is still pending', async () => {
  state.runtime = { instance_id: 'model-a', prefix_cache_max_bytes: 1024, prefix_cache_snapshots: 2 };
  let confirm!: (value: boolean) => void;
  state.confirm.mockImplementationOnce(() => new Promise<boolean>((resolve) => { confirm = resolve; }));
  const view = render(<CachePage />);
  fireEvent.click(screen.getByRole('button', { name: 'Clear this model’s prefix cache' }));
  state.connectionRevision = 1;
  state.runtime = { instance_id: 'model-b', prefix_cache_max_bytes: 1024, prefix_cache_snapshots: 1 };
  view.rerender(<CachePage />);
  expect(screen.getByRole('button', { name: 'Clear this model’s prefix cache' })).toBeEnabled();
  await act(async () => { confirm(true); });
  expect(state.clearRuntimeCache).not.toHaveBeenCalled();
});

it('does not report stale cleanup errors after leaving resources', async () => {
  state.runtime = { instance_id: 'model-a', prefix_cache_max_bytes: 1024, prefix_cache_snapshots: 2 };
  let reject!: (cause: Error) => void;
  state.clearRuntimeCache.mockImplementationOnce(() => new Promise((_, fail) => { reject = fail; }));
  const view = render(<CachePage />);
  fireEvent.click(screen.getByRole('button', { name: 'Clear this model’s prefix cache' }));
  await waitFor(() => expect(state.clearRuntimeCache).toHaveBeenCalledOnce());
  view.unmount();
  await act(async () => { reject(new Error('Old service disconnected')); });
  expect(state.error).not.toHaveBeenCalled();
  expect(state.refreshRuntime).not.toHaveBeenCalled();
});
