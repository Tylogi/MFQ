import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ResourceAllocationPanel } from './ResourceAllocationPanel';
import type { RuntimeInstance, RuntimeStatus } from '../../shared/api/types';

const state = vi.hoisted(() => ({ ready: true, connectionRevision: 0, refreshRuntime: vi.fn().mockResolvedValue(true),
  runtime: null as RuntimeStatus | null, instances: [] as RuntimeInstance[] }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
afterEach(() => { vi.clearAllMocks(); vi.restoreAllMocks(); vi.useRealTimers(); });

it('uses the live shared memory budget and headroom rather than total physical memory', async () => {
  state.runtime = { runtime_memory_budget_bytes: 16 * 2 ** 30, runtime_memory_effective_budget_bytes: 8 * 2 ** 30,
    runtime_memory_committed_bytes: 5 * 2 ** 30, runtime_memory_headroom_bytes: 2 * 2 ** 30, runtime_memory_budget_mode: 'automatic' };
  render(<ResourceAllocationPanel />);
  await act(async () => {});
  expect(screen.getByText('8 GiB')).toBeInTheDocument();
  expect(screen.getByText('5 GiB')).toBeInTheDocument();
  expect(screen.getByText('2 GiB')).toBeInTheDocument();
  expect(screen.queryByText('16 GiB')).not.toBeInTheDocument();
  expect(screen.getByText('Automatic · current usable budget')).toBeInTheDocument();
});

it('refreshes every five seconds and stops on leaving the resources page', async () => {
  vi.useFakeTimers();
  const view = render(<ResourceAllocationPanel />);
  await act(async () => {});
  expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
  view.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
});

it('leaves missing budgets unknown and reports the true per-model sizes', async () => {
  state.runtime = null;
  state.instances = [{ id: 'a', model: 'Qwen3.8', state: 'ready', devices: ['metal'], active_sessions: 2, queued_requests: 1,
    memory: { resident_weight_bytes: 2 ** 30, kv_bytes: 512, prefix_cache_bytes: 128, prefix_cache_limit_bytes: 1024,
      context_count: 2, prefix_cache_blocks: 1, ssd_experts: false, ssd_expert_bytes: 0, ssd_ple: true, ssd_ple_bytes: 2 ** 30 } }];
  render(<ResourceAllocationPanel />);
  await act(async () => {});
  expect(screen.getByText('2 active sessions · 1 queued requests')).toBeInTheDocument();
  expect(screen.getByText(/Active KV 384 B/)).toBeInTheDocument();
  expect(screen.getAllByText('--')).toHaveLength(3);
  state.instances = [];
});

it('skips hidden refreshes and coalesces visibility changes while the shared runtime refresh is pending', async () => {
  vi.useFakeTimers();
  const visibility = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
  let resolve!: (success: boolean) => void;
  state.refreshRuntime.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  const view = render(<ResourceAllocationPanel />);
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(state.refreshRuntime).not.toHaveBeenCalled();
  visibility.mockReturnValue(false);
  fireEvent(document, new Event('visibilitychange'));
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(state.refreshRuntime).toHaveBeenCalledOnce();
  await act(async () => { resolve(true); });
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
  view.unmount();
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
});

it('keeps unloading allocations consistent with the committed budget and marks their lifecycle', async () => {
  state.runtime = { runtime_memory_effective_budget_bytes: 8 * 2 ** 30,
    runtime_memory_committed_bytes: 2 * 2 ** 30, runtime_memory_headroom_bytes: 6 * 2 ** 30 };
  state.instances = [{ id: 'unloading-a', model: 'Qwen3.8', state: 'unloading', devices: ['metal'],
    active_sessions: 0, queued_requests: 0,
    memory: { resident_weight_bytes: 2 * 2 ** 30, kv_bytes: 0, context_count: null, prefix_cache_blocks: null,
      ssd_experts: false, ssd_expert_bytes: 0, ssd_ple: false, ssd_ple_bytes: 0 } }];
  const { container } = render(<ResourceAllocationPanel />);
  await act(async () => {});
  expect(container.querySelector('[data-tier="weights"]')).toHaveTextContent('2 GiB / 8 GiB');
  expect(screen.getByText('0 active sessions · 0 queued requests · 1 unloading')).toBeInTheDocument();
  expect(screen.getByText('Loaded models').parentElement).toHaveTextContent('1');
  state.instances = [];
  state.runtime = null;
});
