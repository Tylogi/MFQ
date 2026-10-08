import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ResourceMonitorPanel } from './ResourceMonitorPanel';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeResources } from '../../shared/api/types';

const state = vi.hoisted(() => ({ ready: true, connectionRevision: 0 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

const snapshot: RuntimeResources = { sampled_at: 100, interval_seconds: 2,
  cpu_name: 'Apple M5 Max', cpu_cores: 18, cpu_utilization_percent: 36,
  gpus: [{ name: 'Apple M5 Max', core_count: 40, utilization_percent: 0 }], disks: [{ name: 'disk0',
    read_bytes_per_second: 2 ** 30, write_bytes_per_second: 0, busy_percent: null, bandwidth_utilization_percent: null }],
  memory_bandwidth_bytes_per_second: null, memory_bandwidth_limit_bytes_per_second: 614e9,
  memory_bandwidth_utilization_percent: null, weights: [{ instance_id: '1', model: 'Qwen3.8',
    expert_read_bytes_per_second: 0, ple_read_bytes_per_second: 2 ** 20, engram_read_bytes_per_second: null }] };

beforeEach(() => { state.ready = true; state.connectionRevision = 0; });
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it('shows measured system and weight traffic without inventing bandwidth utilization', async () => {
  vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue(snapshot);
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByRole('meter', { name: 'CPU: Apple M5 Max 18C' })).toBeInTheDocument();
  expect(screen.getByRole('meter', { name: 'GPU: Apple M5 Max 40C' })).toBeInTheDocument();
  expect(screen.getByText('36%')).toBeInTheDocument();
  expect(screen.getByText('0%')).toBeInTheDocument();
  expect(screen.getByText('1 GiB/s')).toBeInTheDocument();
  expect(screen.getByText('1 MiB/s')).toBeInTheDocument();
  expect(screen.getByText('Specified limit 571.83 GiB/s')).toBeInTheDocument();
  expect(screen.getByText('Not reported', { exact: true })).toBeInTheDocument();
  expect(screen.queryByText('Engram')).not.toBeInTheDocument();
});

it('uses detected device names and does not invent missing GPU core counts', async () => {
  vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue({ ...snapshot,
    cpu_name: 'AMD Ryzen 9 9950X', cpu_cores: 16,
    gpus: [{ name: 'NVIDIA GeForce RTX 5090', utilization_percent: 24 }] });
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByRole('meter', { name: 'CPU: AMD Ryzen 9 9950X 16C' })).toBeInTheDocument();
  expect(screen.getByRole('meter', { name: 'GPU: NVIDIA GeForce RTX 5090' })).toBeInTheDocument();
  expect(screen.queryByText(/Apple M5 Max/)).not.toBeInTheDocument();
});

it('marks unknown device models as unreported', async () => {
  vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue({ ...snapshot,
    cpu_name: null, cpu_cores: null, gpus: [{ name: 'GPU', utilization_percent: null }] });
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByRole('meter', { name: 'CPU: Not reported' })).toBeInTheDocument();
  expect(screen.getByRole('meter', { name: 'GPU: Not reported' })).toBeInTheDocument();
});

it('keeps polling while mounted, stops on navigation and resamples on return', async () => {
  vi.useFakeTimers();
  const query = vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue(snapshot);
  const first = render(<ResourceMonitorPanel />);
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(query).toHaveBeenCalledTimes(2);
  first.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(query).toHaveBeenCalledTimes(2);
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(query).toHaveBeenCalledTimes(3);
});

it('clears stale measurements on failure instead of presenting them as live', async () => {
  vi.useFakeTimers();
  vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValueOnce(snapshot).mockRejectedValue(new Error('offline'));
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByText('36%')).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(screen.queryByText('36%')).not.toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('offline');
});

it('does not poll hidden pages and immediately refreshes on returning', async () => {
  vi.useFakeTimers();
  const visibility = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
  const query = vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue(snapshot);
  const view = render(<ResourceMonitorPanel />);
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(query).not.toHaveBeenCalled();
  visibility.mockReturnValue(false);
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => {});
  expect(query).toHaveBeenCalledOnce();
  expect(screen.getByText('36%')).toBeInTheDocument();
  const signal = query.mock.calls[0][0];
  view.unmount();
  expect(signal?.aborted).toBe(true);
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(query).toHaveBeenCalledOnce();
});

it('keeps sampling serial even when visibility changes during a pending request', async () => {
  vi.useFakeTimers();
  let resolve!: (value: RuntimeResources) => void;
  const query = vi.spyOn(runtimeApi, 'runtimeResources').mockImplementationOnce(() => new Promise((done) => { resolve = done; }))
    .mockResolvedValue(snapshot);
  render(<ResourceMonitorPanel />);
  fireEvent(document, new Event('visibilitychange'));
  fireEvent(document, new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(query).toHaveBeenCalledOnce();
  await act(async () => { resolve(snapshot); });
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(query).toHaveBeenCalledTimes(2);
});

it('aborts and ignores a previous server’s late resource response', async () => {
  let resolve!: (value: RuntimeResources) => void;
  const query = vi.spyOn(runtimeApi, 'runtimeResources').mockImplementationOnce(() => new Promise((done) => { resolve = done; }))
    .mockResolvedValue({ ...snapshot, cpu_utilization_percent: 12 });
  const view = render(<ResourceMonitorPanel />);
  const previousSignal = query.mock.calls[0][0];
  state.connectionRevision = 1;
  view.rerender(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(previousSignal?.aborted).toBe(true);
  expect(screen.getByText('12%')).toBeInTheDocument();
  await act(async () => { resolve(snapshot); });
  expect(screen.queryByText('36%')).not.toBeInTheDocument();
  expect(screen.getByText('12%')).toBeInTheDocument();
});
