import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useServingStatistics } from './useServingStatistics';
import type { ServingStatistics } from '../../shared/api/contracts/runtime';

const api = vi.hoisted(() => vi.fn());
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { servingStatistics: api } }));

function stats(scope: 'session' | 'alltime', requests: number): ServingStatistics {
  return { scope, started_at: 1, activity: [], totals: { completed_requests: requests,
    prompt_tokens: 0, prefill_tokens: 0, completion_tokens: 0, cached_tokens: 0,
    prefill_tps: null, decode_tps: null, ttft_ms: null, cache_hit_rate: null } };
}

beforeEach(() => { vi.useFakeTimers(); api.mockReset(); api.mockResolvedValue(stats('session', 1)); });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

it('polls at 500ms, pauses while hidden, refreshes on return and stops on unmount', async () => {
  const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  const view = renderHook(() => useServingStatistics('session', 0));
  await act(async () => {});
  expect(api).toHaveBeenCalledTimes(1);
  await act(async () => { await vi.advanceTimersByTimeAsync(500); });
  expect(api).toHaveBeenCalledTimes(2);
  hidden.mockReturnValue(true);
  document.dispatchEvent(new Event('visibilitychange'));
  await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
  expect(api).toHaveBeenCalledTimes(2);
  hidden.mockReturnValue(false);
  await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
  expect(api).toHaveBeenCalledTimes(3);
  view.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
  expect(api).toHaveBeenCalledTimes(3);
});

it('ignores an old scope response and coalesces focus while a request is pending', async () => {
  let resolve!: (value: ServingStatistics) => void;
  api.mockImplementationOnce(() => new Promise(done => { resolve = done; }));
  const view = renderHook(({ scope }: { scope: 'session' | 'alltime' }) => useServingStatistics(scope, 0),
    { initialProps: { scope: 'session' } });
  window.dispatchEvent(new Event('focus'));
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(api).toHaveBeenCalledTimes(1);
  api.mockResolvedValueOnce(stats('alltime', 22));
  view.rerender({ scope: 'alltime' });
  await act(async () => {});
  expect(view.result.current.data?.totals.completed_requests).toBe(22);
  await act(async () => { resolve(stats('session', 999)); });
  expect(view.result.current.data?.totals.completed_requests).toBe(22);
});

it('preserves the last valid statistics on a temporary polling failure', async () => {
  const view = renderHook(() => useServingStatistics('session', 0));
  await act(async () => {});
  api.mockRejectedValueOnce(new Error('offline'));
  await act(async () => { await vi.advanceTimersByTimeAsync(500); });
  expect(view.result.current.data?.totals.completed_requests).toBe(1);
  expect(view.result.current.error).toBe('offline');
});
