import { act, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { RuntimeResources } from '../../shared/api/types';
import { ResourceMonitorPanel } from './ResourceMonitorPanel';

const mocks = vi.hoisted(() => ({ resources: vi.fn(), ready: true, revision: 0 }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ready: mocks.ready, connectionRevision: mocks.revision }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { runtimeResources: mocks.resources } }));
afterEach(() => { vi.clearAllMocks(); vi.useRealTimers(); });

function sample(read: number | null, written: number | null): RuntimeResources {
  return { sampled_at: 1, interval_seconds: 2, cpu_utilization_percent: null, gpus: [], disks: [],
    memory_bandwidth_bytes_per_second: null, memory_bandwidth_limit_bytes_per_second: null,
    memory_bandwidth_utilization_percent: null,
    weights: [{ instance_id: 'a', model: 'Qwen3.8-Flash-Next', expert_read_bytes_per_second: 0,
      ple_read_bytes_per_second: 2 ** 20, engram_read_bytes_per_second: null,
      kv_read_bytes_per_second: read, kv_write_bytes_per_second: written }] };
}

it('polls per-model streamed KV read and write rates alongside weight reads', async () => {
  vi.useFakeTimers();
  mocks.resources.mockResolvedValue(sample(2 ** 20, 2 * 2 ** 20));
  const view = render(<ResourceMonitorPanel />);
  await act(async () => {});
  expect(screen.getByText('Weight and KV traffic')).toBeInTheDocument();
  expect(screen.getByText('Streamed KV read').parentElement).toHaveTextContent('1 MiB/s');
  expect(screen.getByText('Streamed KV write').parentElement).toHaveTextContent('2 MiB/s');
  mocks.resources.mockResolvedValue(sample(0, null));
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(screen.getByText('Streamed KV read').parentElement).toHaveTextContent('0 MiB/s');
  expect(screen.getByText('Streamed KV write').parentElement).toHaveTextContent('--');
  view.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
  expect(mocks.resources).toHaveBeenCalledTimes(2);
});
