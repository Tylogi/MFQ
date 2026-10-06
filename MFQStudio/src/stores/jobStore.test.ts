/**
 * Unit tests for background job state management (jobStore).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { jobsApi } from '../shared/api/resources/jobs';
import { useJobStore } from './jobStore';
import type { JobResource, JobEventResource } from '../shared/api/types';

const mockJob1: JobResource = {
  id: 'job-1',
  kind: 'model.load',
  status: 'running',
  progress: 10,
  cancel_requested: false,
  payload: { model: 'test-model' },
  created_at: '2026-09-23T10:00:00Z',
  updated_at: '2026-09-23T10:00:00Z',
};

const mockJob2: JobResource = {
  id: 'job-2',
  kind: 'quantize.gguf',
  status: 'succeeded',
  progress: 100,
  cancel_requested: false,
  payload: { model: 'test-model-2' },
  created_at: '2026-09-23T09:00:00Z',
  updated_at: '2026-09-23T09:30:00Z',
};

describe('describes jobStore test behavior 1', () => {
  beforeEach(() => {
    useJobStore.getState().clearJobStreams();
    useJobStore.getState().setJobs([]);
    vi.restoreAllMocks();
  });

  it('verifies jobStore test behavior 2', () => {
    const state = useJobStore.getState();
    expect(state.jobs).toEqual([]);
    expect(state.activeJobIds).toEqual([]);
  });

  it('verifies jobStore test behavior 3', () => {
    useJobStore.getState().setJobs([{ ...mockJob1, progress: 0.7, updated_at: '2026-09-23T10:10:00Z' }]);
    let receive!: (event: JobEventResource) => void;
    const stream = vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, onEvent) => {
      receive = onEvent;
      return new Promise<void>(() => {});
    });
    const stop = useJobStore.getState().streamJobEvents('job-1');
    receive({ job_id: 'job-1', sequence: 8, type: 'progress', level: 'info', data: {}, progress: 0.2, created_at: '2026-09-23T10:05:00Z' });
    useJobStore.getState().setJobs([{ ...mockJob1, progress: 0.1 }]);
    expect(useJobStore.getState().jobs[0].progress).toBe(0.7);
    stop();
    useJobStore.getState().streamJobEvents('job-1');
    expect(stream.mock.calls[1][3]).toBe(8);
    receive({ job_id: 'job-1', sequence: 9, type: 'progress', level: 'info', data: {}, progress: 0.8, created_at: '2026-09-23T10:11:00Z' });
    expect(useJobStore.getState().jobs[0].progress).toBe(0.8);
  });

  it('verifies jobStore test behavior 4', () => {
    useJobStore.getState().setJobs([mockJob1, mockJob2]);
    const state = useJobStore.getState();
    expect(state.jobs).toHaveLength(2);
    expect(state.activeJobIds).toEqual(['job-1']);
  });

  it('verifies jobStore test behavior 5', () => {
    const input = [{ ...mockJob1 }];
    useJobStore.getState().setJobs(input);
    const previous = useJobStore.getState();
    expect(previous.jobs).not.toBe(input);
    expect(previous.jobs[0]).not.toBe(input[0]);

    input[0].progress = 30;
    input.push(mockJob2);
    expect(previous.jobs).toHaveLength(1);
    expect(previous.jobs[0].progress).toBe(10);

    const patch = { progress: 50 };
    useJobStore.getState().updateJob('job-1', patch);
    expect(useJobStore.getState().jobs).not.toBe(previous.jobs);
    expect(useJobStore.getState().jobs[0]).not.toBe(previous.jobs[0]);
    expect(previous.jobs[0].progress).toBe(10);
    expect(input[0].progress).toBe(30);
    expect(patch.progress).toBe(50);
  });

  it('verifies jobStore test behavior 6', () => {
    useJobStore.getState().setJobs([mockJob2]);
    useJobStore.getState().addJob(mockJob1);
    const state = useJobStore.getState();
    expect(state.jobs[0].id).toBe('job-1');
    expect(state.activeJobIds).toEqual(['job-1']);
  });

  it('verifies jobStore test behavior 7', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const updatedJob: JobResource = { ...mockJob1, status: 'succeeded', progress: 100 };
    useJobStore.getState().addJob(updatedJob);
    const state = useJobStore.getState();
    expect(state.jobs).toHaveLength(1);
    expect(state.jobs[0].status).toBe('succeeded');
    expect(state.activeJobIds).toEqual([]);
  });

  it('verifies jobStore test behavior 8', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const prevActive = useJobStore.getState().activeJobIds;

    useJobStore
      .getState()
      .updateJob('job-1', { progress: 45.5, updated_at: '2026-09-23T10:05:00Z' });
    const state = useJobStore.getState();
    expect(state.jobs[0].progress).toBe(45.5);
    expect(state.jobs[0].updated_at).toBe('2026-09-23T10:05:00Z');
    // Preserve the reference when active-list elements are unchanged.
    expect(state.activeJobIds).toBe(prevActive);
  });

  it('verifies jobStore test behavior 9', () => {
    useJobStore.getState().setJobs([mockJob1]);
    expect(useJobStore.getState().activeJobIds).toEqual(['job-1']);

    useJobStore.getState().updateJob('job-1', { status: 'succeeded' });
    expect(useJobStore.getState().activeJobIds).toEqual([]);
  });

  it('verifies jobStore test behavior 10', async () => {
    useJobStore.getState().setJobs([mockJob1]);

    let eventCallback: ((event: JobEventResource) => void) | undefined;
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, onEvent) => {
      eventCallback = onEvent;
      return Promise.resolve();
    });

    const onTerminal = vi.fn();
    const cleanup = useJobStore.getState().streamJobEvents('job-1', { onTerminal });

    expect(jobsApi.streamJobEvents).toHaveBeenCalledWith(
      'job-1',
      expect.any(Function),
      expect.any(AbortSignal),
      0,
    );
    expect(eventCallback).toBeDefined();

    // Emit a progress event.
    eventCallback!({
      job_id: 'job-1',
      sequence: 1,
      type: 'progress',
      level: 'info',
      data: {},
      progress: 60,
      created_at: '2026-09-23T10:10:00Z',
    });

    expect(useJobStore.getState().jobs[0].progress).toBe(60);
    expect(onTerminal).not.toHaveBeenCalled();

    // Emit a terminal-state event.
    eventCallback!({
      job_id: 'job-1',
      sequence: 2,
      type: 'state',
      level: 'info',
      data: { status: 'succeeded' },
      progress: 100,
      created_at: '2026-09-23T10:15:00Z',
    });

    expect(useJobStore.getState().jobs[0].status).toBe('succeeded');
    expect(onTerminal).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'job-1', status: 'succeeded' }),
    );

    cleanup();
  });

  it('verifies jobStore test behavior 11', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, _onEvent, signal) => {
      signals.push(signal);
      return new Promise<void>(() => {});
    });

    const first = useJobStore.getState().streamJobEvents('job-1');
    const duplicate = useJobStore.getState().streamJobEvents('job-1');
    duplicate();
    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(false);

    first();
    useJobStore.getState().streamJobEvents('job-1');
    first();
    duplicate();
    expect(signals).toHaveLength(2);
    expect(signals[1].aborted).toBe(false);
  });

  it('verifies jobStore test behavior 12', async () => {
    useJobStore.getState().setJobs([mockJob1]);
    const callbacks: Array<(event: JobEventResource) => void> = [];
    const rejectors: Array<(cause: Error) => void> = [];
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, onEvent, signal) => {
      callbacks.push(onEvent);
      signals.push(signal);
      return new Promise<void>((_resolve, reject) => {
        rejectors.push(reject);
      });
    });
    const onError = vi.fn();
    const oldCleanup = useJobStore.getState().streamJobEvents('job-1', { onError });
    oldCleanup();
    useJobStore.getState().streamJobEvents('job-1', { onError });
    callbacks[0]({
      job_id: 'job-1',
      sequence: 1,
      type: 'progress',
      level: 'info',
      data: {},
      progress: 80,
      created_at: '2026-09-23T10:10:00Z',
    });
    rejectors[0](new Error('old stream failed'));
    await Promise.resolve();
    expect(useJobStore.getState().jobs[0].progress).toBe(10);
    expect(onError).not.toHaveBeenCalled();
    expect(signals[1].aborted).toBe(false);
    expect(jobsApi.streamJobEvents).toHaveBeenCalledTimes(2);
    useJobStore.getState().streamJobEvents('job-1');
    expect(jobsApi.streamJobEvents).toHaveBeenCalledTimes(2);
  });

  it('verifies jobStore test behavior 13', () => {
    useJobStore.getState().setJobs([mockJob1]);
    let oldEvent: ((event: JobEventResource) => void) | undefined;
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, onEvent, signal) => {
      oldEvent ??= onEvent;
      signals.push(signal);
      return new Promise<void>(() => {});
    });
    let cleanup = () => {};
    const onEvent = vi.fn(() => {
      cleanup();
      useJobStore.getState().streamJobEvents('job-1');
    });
    cleanup = useJobStore.getState().streamJobEvents('job-1', { onEvent });
    oldEvent!({
      job_id: 'job-1',
      sequence: 1,
      type: 'progress',
      level: 'info',
      data: {},
      progress: 70,
      created_at: '2026-09-23T10:10:00Z',
    });
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(useJobStore.getState().jobs[0].progress).toBe(10);
    expect(signals[1].aborted).toBe(false);
  });

  it('verifies jobStore test behavior 14', async () => {
    useJobStore.getState().setJobs([mockJob1]);
    let resolveOld: (() => void) | undefined;
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents')
      .mockImplementationOnce((_id, _onEvent, signal) => {
        signals.push(signal);
        return new Promise<void>((resolve) => {
          resolveOld = resolve;
        });
      })
      .mockImplementationOnce((_id, _onEvent, signal) => {
        signals.push(signal);
        return new Promise<void>(() => {});
      });
    const onError = vi.fn();
    const cleanup = useJobStore.getState().streamJobEvents('job-1', { onError });
    cleanup();
    useJobStore.getState().streamJobEvents('job-1', { onError });
    resolveOld!();
    await Promise.resolve();
    expect(onError).not.toHaveBeenCalled();
    expect(signals[1].aborted).toBe(false);
  });

  it('verifies jobStore test behavior 15', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const callbacks: Array<(event: JobEventResource) => void> = [];
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, onEvent, signal) => {
      callbacks.push(onEvent);
      signals.push(signal);
      return new Promise<void>(() => {});
    });
    const onTerminal = vi.fn(() => {
      useJobStore.getState().streamJobEvents('job-1');
    });
    const cleanup = useJobStore.getState().streamJobEvents('job-1', { onTerminal });
    callbacks[0]({
      job_id: 'job-1',
      sequence: 1,
      type: 'state',
      level: 'info',
      data: { status: 'succeeded' },
      progress: 100,
      created_at: '2026-09-23T10:15:00Z',
    });
    cleanup();
    expect(onTerminal).toHaveBeenCalledTimes(1);
    expect(signals).toHaveLength(2);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it('verifies jobStore test behavior 16', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const signals: AbortSignal[] = [];
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((_id, _onEvent, signal) => {
      signals.push(signal);
      return new Promise<void>(() => {});
    });
    const expired = new AbortController();
    expired.abort();
    useJobStore.getState().streamJobEvents('job-1', { signal: expired.signal });
    expect(signals).toHaveLength(0);

    const owner = new AbortController();
    useJobStore.getState().streamJobEvents('job-1', { signal: owner.signal });
    owner.abort();
    useJobStore.getState().streamJobEvents('job-1');
    expect(signals).toHaveLength(2);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it('verifies jobStore test behavior 17', () => {
    useJobStore.getState().setJobs([mockJob1, { ...mockJob1, id: 'job-3' }]);
    const signals = new Map<string, AbortSignal[]>();
    vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation((id, _onEvent, signal) => {
      signals.set(id, [...(signals.get(id) ?? []), signal]);
      return new Promise<void>(() => {});
    });
    const direct = useJobStore.getState().streamJobEvents('job-1');
    const batch = useJobStore.getState().watchActiveJobs();
    const duplicate = useJobStore.getState().watchActiveJobs();
    duplicate();
    expect(signals.get('job-3')?.[0].aborted).toBe(false);
    batch();
    expect(signals.get('job-1')?.[0].aborted).toBe(false);
    expect(signals.get('job-3')?.[0].aborted).toBe(true);

    useJobStore.getState().streamJobEvents('job-3');
    batch();
    expect(signals.get('job-3')?.[1].aborted).toBe(false);
    direct();
  });

  it('verifies jobStore test behavior 18', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const streamSpy = vi.spyOn(jobsApi, 'streamJobEvents').mockResolvedValue(undefined);

    useJobStore.getState().watchActiveJobs();
    expect(streamSpy).toHaveBeenCalledTimes(1);

    // Repeated calls do not start duplicate streams.
    useJobStore.getState().watchActiveJobs();
    expect(streamSpy).toHaveBeenCalledTimes(1);

    useJobStore.getState().clearJobStreams();
  });

  it('verifies jobStore test behavior 19', async () => {
    useJobStore.getState().setJobs([mockJob1]);
    const callbacks: Array<(event: JobEventResource) => void> = [];
    const streamSpy = vi.spyOn(jobsApi, 'streamJobEvents');
    streamSpy.mockImplementationOnce(() => Promise.reject(new Error('stream offline')));
    streamSpy.mockImplementationOnce((_id, onEvent) => {
      callbacks.push(onEvent);
      return new Promise<void>(() => {});
    });
    const onError = vi.fn();
    const onEvent = vi.fn();

    useJobStore.getState().watchActiveJobs({ onError, onEvent });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith('job-1', expect.any(Error)));
    useJobStore.getState().watchActiveJobs({ onError, onEvent });
    expect(streamSpy).toHaveBeenCalledTimes(2);
    expect(onEvent).not.toHaveBeenCalled();

    callbacks[0]({
      job_id: 'job-1',
      sequence: 1,
      type: 'progress',
      level: 'info',
      data: {},
      progress: 50,
      created_at: '2026-09-23T10:10:00Z',
    });
    expect(onEvent).toHaveBeenCalledWith('job-1');
  });

  it('verifies jobStore test behavior 20', async () => {
    useJobStore.getState().setJobs([mockJob1]);
    const streamSpy = vi.spyOn(jobsApi, 'streamJobEvents').mockResolvedValue(undefined);
    const onError = vi.fn();
    useJobStore.getState().watchActiveJobs({ onError });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith('job-1', expect.any(Error)));
    useJobStore.getState().watchActiveJobs({ onError });
    expect(streamSpy).toHaveBeenCalledTimes(2);
  });
});
