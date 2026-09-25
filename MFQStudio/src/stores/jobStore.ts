/**
 * MFQ Studio 后台任务状态管理，隔离高频任务事件与进度更新，避免根级上下文频繁重渲染。
 */

import { create } from 'zustand';
import { jobsApi } from '../shared/api/resources/jobs';
import type { JobResource, JobEventResource } from '../shared/api/types';

const ACTIVE_STATUSES: ReadonlySet<JobResource['status']> = new Set([
  'queued',
  'running',
  'cancelling',
]);

const TERMINAL_STATUSES: ReadonlySet<JobResource['status']> = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
]);

/** 过滤活跃状态的任务标识。 */
function extractActiveJobIds(jobs: JobResource[]): string[] {
  return jobs.filter((job) => ACTIVE_STATUSES.has(job.status)).map((job) => job.id);
}

/** 比较两个字符串数组元素是否完全一致。 */
function haveSameElements(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export interface WatchActiveJobsOptions {
  /** 当任务转入终态时触发，通常用于通知全局运行时刷新模型与实例。 */
  onTerminal?: (jobId: string, status: JobResource['status']) => void;
  /** 当 SSE 事件流发生异常且未主动中断时触发。 */
  onError?: (jobId: string, cause: unknown) => void;
  /** 某个任务收到事件时触发，用于确认失败的订阅已恢复。 */
  onEvent?: (jobId: string) => void;
  /** 外部生命周期取消信号。 */
  signal?: AbortSignal;
}

export interface StreamJobEventsOptions {
  /** 任务进入终态后的回调。 */
  onTerminal?: (job: JobResource) => void;
  /** 流监听异常回调。 */
  onError?: (cause: unknown) => void;
  /** 事件到达后的回调。 */
  onEvent?: () => void;
  /** 取消信号。 */
  signal?: AbortSignal;
}

export interface JobState {
  /** 当前所有后台任务记录。 */
  jobs: JobResource[];
  /** 处于就绪、运行中或取消中的活跃任务标识列表。 */
  activeJobIds: string[];
  /**
   * 批量重置或更新任务列表，并同步刷新活跃任务标识。
   *
   * @param jobs 新的任务列表数据
   */
  setJobs: (jobs: JobResource[]) => void;
  /**
   * 将新建任务合并到任务列表头部，若已存在相同标识则更新，并同步活跃任务标识。
   *
   * @param job 新建或待并入的任务对象
   */
  addJob: (job: JobResource) => void;
  /**
   * 局部更新单个任务字段，例如进度或状态变更；若活跃任务列表未变则保留原数组引用。
   *
   * @param id 任务唯一标识
   * @param patch 待合并的任务局部属性
   */
  updateJob: (id: string, patch: Partial<JobResource>) => void;
  /**
   * 启动单个任务的 SSE 事件流监听，进度和状态直接更新到 store，隔离高频重渲染。
   *
   * @param id 任务唯一标识
   * @param options 监听参数，包含终态通知与异常回调
   * @returns 停止当前任务事件流的清理函数
   */
  streamJobEvents: (id: string, options?: StreamJobEventsOptions) => () => void;
  /**
   * 监听当前所有活跃状态的任务事件流，已建立连接的任务不重复建立。
   *
   * @param options 监听参数与生命周期信号
   * @returns 停止本次批量监听的清理函数
   */
  watchActiveJobs: (options?: WatchActiveJobsOptions) => () => void;
  /**
   * 中断并清空所有进行中的任务 SSE 连接。
   */
  clearJobStreams: () => void;
}

/** 模块内部持久保存的任务 SSE 控制器映射，按任务 ID 隔离。 */
const activeStreams = new Map<string, AbortController>();

/** 仅结束仍由当前控制器持有的任务流，避免旧清理函数影响重订阅。 */
function stopOwnedStream(id: string, controller: AbortController): void {
  if (activeStreams.get(id) !== controller) return;
  activeStreams.delete(id);
  controller.abort();
}

/** 后台任务全局状态 Store，提供细粒度的任务订阅与事件隔离。 */
export const useJobStore = create<JobState>()((set, get) => ({
  jobs: [],
  activeJobIds: [],

  setJobs: (jobs) => {
    const nextJobs = jobs.map((job) => ({ ...job }));
    const activeJobIds = extractActiveJobIds(nextJobs);
    set({ jobs: nextJobs, activeJobIds });
  },

  addJob: (job) => {
    set((state) => {
      const jobs = [{ ...job }, ...state.jobs.filter((item) => item.id !== job.id)];
      const activeJobIds = extractActiveJobIds(jobs);
      return { jobs, activeJobIds };
    });
  },

  updateJob: (id, patch) => {
    set((state) => {
      let changed = false;
      const jobs = state.jobs.map((job) => {
        if (job.id !== id) return job;
        changed = true;
        return { ...job, ...patch };
      });
      if (!changed) return state;
      const nextActive = extractActiveJobIds(jobs);
      return {
        jobs,
        activeJobIds: haveSameElements(state.activeJobIds, nextActive)
          ? state.activeJobIds
          : nextActive,
      };
    });
  },

  streamJobEvents: (id, options) => {
    if (activeStreams.has(id) || options?.signal?.aborted) return () => {};
    const controller = new AbortController();
    activeStreams.set(id, controller);
    const stop = () => stopOwnedStream(id, controller);

    if (options?.signal) {
      options.signal.addEventListener('abort', stop, { once: true, signal: controller.signal });
      if (options.signal.aborted) {
        stop();
        return stop;
      }
    }

    void jobsApi
      .streamJobEvents(
        id,
        (event: JobEventResource) => {
          if (controller.signal.aborted || activeStreams.get(id) !== controller) return;
          options?.onEvent?.();
          if (controller.signal.aborted || activeStreams.get(id) !== controller) return;
          const status =
            event.type === 'state' && typeof event.data.status === 'string'
              ? (event.data.status as JobResource['status'])
              : null;
          get().updateJob(id, {
            ...(status ? { status } : {}),
            ...(typeof event.progress === 'number' ? { progress: event.progress } : {}),
            updated_at: event.created_at,
          });
          if (status && TERMINAL_STATUSES.has(status)) {
            stop();
            const finishedJob = get().jobs.find((j) => j.id === id);
            if (finishedJob) options?.onTerminal?.(finishedJob);
          }
        },
        controller.signal,
      )
      .then(() => {
        if (!controller.signal.aborted && activeStreams.get(id) === controller) {
          stop();
          options?.onError?.(new Error('Task event stream closed unexpectedly'));
        }
      })
      .catch((cause) => {
        if (!controller.signal.aborted && activeStreams.get(id) === controller) {
          stop();
          options?.onError?.(cause);
        }
      });

    return stop;
  },

  watchActiveJobs: (options) => {
    const currentActive = get().activeJobIds;
    const ownedStreams = new Map<string, AbortController>();
    // 启动新增活跃任务的监听
    for (const id of currentActive) {
      if (activeStreams.has(id)) continue;
      get().streamJobEvents(id, {
        signal: options?.signal,
        onTerminal: (job) => options?.onTerminal?.(job.id, job.status),
        onEvent: () => options?.onEvent?.(id),
        onError: (cause) => options?.onError?.(id, cause),
      });
      const controller = activeStreams.get(id);
      if (controller) ownedStreams.set(id, controller);
    }

    // 停止已不再活跃的任务连接
    for (const [id, controller] of activeStreams.entries()) {
      if (!currentActive.includes(id)) {
        stopOwnedStream(id, controller);
      }
    }

    return () => {
      for (const [id, controller] of ownedStreams) stopOwnedStream(id, controller);
      ownedStreams.clear();
    };
  },

  clearJobStreams: () => {
    for (const [id, controller] of activeStreams) stopOwnedStream(id, controller);
  },
}));
