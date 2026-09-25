/**
 * 后台任务状态管理 (jobStore) 单元测试。
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

describe('jobStore 后台任务状态管理', () => {
  beforeEach(() => {
    useJobStore.getState().clearJobStreams();
    useJobStore.getState().setJobs([]);
    vi.restoreAllMocks();
  });

  it('初始状态为空列表与空活跃标识', () => {
    const state = useJobStore.getState();
    expect(state.jobs).toEqual([]);
    expect(state.activeJobIds).toEqual([]);
  });

  it('setJobs 设置任务列表并正确提取活跃任务标识', () => {
    useJobStore.getState().setJobs([mockJob1, mockJob2]);
    const state = useJobStore.getState();
    expect(state.jobs).toHaveLength(2);
    expect(state.activeJobIds).toEqual(['job-1']);
  });

  it('setJobs 保存独立数组及任务快照，updateJob 不修改旧快照或调用方对象', () => {
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

  it('addJob 新增任务并插入列表前列', () => {
    useJobStore.getState().setJobs([mockJob2]);
    useJobStore.getState().addJob(mockJob1);
    const state = useJobStore.getState();
    expect(state.jobs[0].id).toBe('job-1');
    expect(state.activeJobIds).toEqual(['job-1']);
  });

  it('addJob 遇到已有标识时进行替换更新', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const updatedJob: JobResource = { ...mockJob1, status: 'succeeded', progress: 100 };
    useJobStore.getState().addJob(updatedJob);
    const state = useJobStore.getState();
    expect(state.jobs).toHaveLength(1);
    expect(state.jobs[0].status).toBe('succeeded');
    expect(state.activeJobIds).toEqual([]);
  });

  it('updateJob 仅更新指定属性且保持活跃数组引用稳定性', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const prevActive = useJobStore.getState().activeJobIds;

    useJobStore
      .getState()
      .updateJob('job-1', { progress: 45.5, updated_at: '2026-09-23T10:05:00Z' });
    const state = useJobStore.getState();
    expect(state.jobs[0].progress).toBe(45.5);
    expect(state.jobs[0].updated_at).toBe('2026-09-23T10:05:00Z');
    // 活跃列表元素未变化时引用保持一致
    expect(state.activeJobIds).toBe(prevActive);
  });

  it('updateJob 变更状态为终态时同步剔除活跃列表', () => {
    useJobStore.getState().setJobs([mockJob1]);
    expect(useJobStore.getState().activeJobIds).toEqual(['job-1']);

    useJobStore.getState().updateJob('job-1', { status: 'succeeded' });
    expect(useJobStore.getState().activeJobIds).toEqual([]);
  });

  it('streamJobEvents 处理高频 SSE 事件与终态回调', async () => {
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
    );
    expect(eventCallback).toBeDefined();

    // 触发进度事件
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

    // 触发终态事件
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

  it('重复订阅的 cleanup 不拥有连接，也不能取消随后建立的新连接', () => {
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

  it('取消后的旧事件及异步结束不能覆盖新订阅', async () => {
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

  it('事件回调同步取消并重订阅时，旧事件不得更新新流的任务', () => {
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

  it('旧流正常结束后只反馈自身故障，不干扰重新订阅', async () => {
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

  it('终态回调重订阅时旧 cleanup 不会关掉新流', () => {
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

  it('取消信号仅关闭所属连接，旧信号不影响重新订阅', () => {
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

  it('批量监听只清理本次创建的流，旧 cleanup 不影响替代连接', () => {
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

  it('watchActiveJobs 自动为活跃任务开启监听并可在清除时取消', () => {
    useJobStore.getState().setJobs([mockJob1]);
    const streamSpy = vi.spyOn(jobsApi, 'streamJobEvents').mockResolvedValue(undefined);

    useJobStore.getState().watchActiveJobs();
    expect(streamSpy).toHaveBeenCalledTimes(1);

    // 重复调用不重复开启
    useJobStore.getState().watchActiveJobs();
    expect(streamSpy).toHaveBeenCalledTimes(1);

    useJobStore.getState().clearJobStreams();
  });

  it('任务流断开后可重新订阅，并按任务标识确认恢复', async () => {
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

  it('流意外正常关闭也会释放订阅并反馈故障', async () => {
    useJobStore.getState().setJobs([mockJob1]);
    const streamSpy = vi.spyOn(jobsApi, 'streamJobEvents').mockResolvedValue(undefined);
    const onError = vi.fn();
    useJobStore.getState().watchActiveJobs({ onError });
    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith('job-1', expect.any(Error)));
    useJobStore.getState().watchActiveJobs({ onError });
    expect(streamSpy).toHaveBeenCalledTimes(2);
  });
});
