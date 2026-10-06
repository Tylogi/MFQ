/** Verify job-log subscription merging and isolation of stale requests when switching jobs. */
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import type { JobEventResource, RuntimeLogEntry } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useJobEventLog } from './useJobEventLog';

afterEach(() => vi.restoreAllMocks());

it('verifies useJobEventLog test behavior 1', async () => {
  const callbacks: Array<(event: JobEventResource) => void> = [];
  vi.spyOn(jobsApi, 'jobEvents').mockResolvedValue([
    { sequence: 2, level: 'info', message: 'history', fields: {}, created_at: '' },
  ] as RuntimeLogEntry[]);
  vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation(async (_id, callback) => {
    callbacks.push(callback);
  });
  const { result } = renderHook(() => useJobEventLog('job-1'));
  await waitFor(() => expect(result.current.jobLogs).toHaveLength(1));
  act(() => {
    callbacks[0]({ job_id: 'job-1', sequence: 1, type: 'log', level: 'info',
      message: 'live', data: {}, created_at: '' });
    callbacks[0]({ job_id: 'job-1', sequence: 2, type: 'log', level: 'info',
      message: 'duplicate', data: {}, created_at: '' });
  });
  expect(result.current.jobLogs.map((entry) => entry.message)).toEqual(['live', 'history']);
});

it('verifies useJobEventLog test behavior 2', async () => {
  let resolveOld!: (entries: RuntimeLogEntry[]) => void;
  vi.spyOn(jobsApi, 'jobEvents').mockImplementation((id) =>
    id === 'job-1'
      ? new Promise<RuntimeLogEntry[]>((resolve) => { resolveOld = resolve; })
      : Promise.resolve([]),
  );
  const signals: AbortSignal[] = [];
  vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation(async (_id, _callback, signal) => {
    signals.push(signal);
  });
  const { result, rerender } = renderHook(({ id }) => useJobEventLog(id), {
    initialProps: { id: 'job-1' },
  });
  rerender({ id: 'job-2' });
  expect(signals[0].aborted).toBe(true);
  await act(async () => {
    resolveOld([{ sequence: 1, level: 'info', message: 'stale', fields: {}, created_at: '' }]);
  });
  expect(result.current.jobLogs).toEqual([]);
});
