import { act, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ResourceMonitorPanel } from './ResourceMonitorPanel';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeResources } from '../../shared/api/types';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ready: true, connectionRevision: 0 }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

const snapshot: RuntimeResources = { sampled_at: 100, interval_seconds: 2, cpu_utilization_percent: 36,
  gpus: [{ name: 'Apple M5 Max', utilization_percent: 0 }], disks: [{ name: 'disk0',
    read_bytes_per_second: 2 ** 30, write_bytes_per_second: 0, busy_percent: null, bandwidth_utilization_percent: null }],
  memory_bandwidth_bytes_per_second: null, memory_bandwidth_limit_bytes_per_second: 614e9,
  memory_bandwidth_utilization_percent: null, weights: [{ instance_id: '1', model: 'Qwen3.8',
    expert_read_bytes_per_second: 0, ple_read_bytes_per_second: 2 ** 20, engram_read_bytes_per_second: null }] };

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it('shows measured system and weight traffic without inventing bandwidth utilization', async () => {
  vi.spyOn(runtimeApi, 'runtimeResources').mockResolvedValue(snapshot);
  render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByText('36%')).toBeInTheDocument();
  expect(screen.getByText('0%')).toBeInTheDocument();
  expect(screen.getByText('1 GiB/s')).toBeInTheDocument();
  expect(screen.getByText('1 MiB/s')).toBeInTheDocument();
  expect(screen.getByText('Specified limit 571.83 GiB/s')).toBeInTheDocument();
  expect(screen.getByText('Not reported', { exact: true })).toBeInTheDocument();
  expect(screen.queryByText('Engram')).not.toBeInTheDocument();
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
