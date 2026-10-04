/**
 * MFQ Studio background job state management, isolating high-frequency job events and progress updates to avoid frequent root-context rerenders.
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

/** Filter IDs for jobs in active states. */
function extractActiveJobIds(jobs: JobResource[]): string[] {
  return jobs.filter((job) => ACTIVE_STATUSES.has(job.status)).map((job) => job.id);
}

/** Check whether two string arrays contain exactly the same elements. */
function haveSameElements(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export interface WatchActiveJobsOptions {
  /** Called when a job reaches a terminal state, typically to refresh global runtime models and instances. */
  onTerminal?: (jobId: string, status: JobResource['status']) => void;
  /** Called when an SSE stream fails without being explicitly aborted. */
  onError?: (jobId: string, cause: unknown) => void;
  /** Called when a job receives an event, confirming that a failed subscription has recovered. */
  onEvent?: (jobId: string) => void;
  /** External lifecycle cancellation signal. */
  signal?: AbortSignal;
}

export interface StreamJobEventsOptions {
  /** Callback after a job reaches a terminal state. */
  onTerminal?: (job: JobResource) => void;
  /** Callback for stream-listening errors. */
  onError?: (cause: unknown) => void;
  /** Callback invoked when an event arrives. */
  onEvent?: () => void;
  /** Cancellation signal. */
  signal?: AbortSignal;
}

export interface JobState {
  /** All current background job records. */
  jobs: JobResource[];
  /** IDs of jobs that are queued, running, or cancelling. */
  activeJobIds: string[];
  /**
   * Reset or update the job list in bulk and refresh active job IDs.
   *
   * @param jobs The new job list.
   */
  setJobs: (jobs: JobResource[]) => void;
  /**
   * Merge a new job at the start of the list, updating an existing entry with the same ID and synchronizing active job IDs.
   *
   * @param job The new job or job to merge.
   */
  addJob: (job: JobResource) => void;
  /**
   * Partially update one job field, such as progress or status; preserve the active-list array reference if it is unchanged.
   *
   * @param id Unique job identifier.
   * @param patch Partial job properties to merge.
   */
  updateJob: (id: string, patch: Partial<JobResource>) => void;
  /**
   * Start an SSE event stream for one job, updating progress and status directly in the store to isolate high-frequency rerenders.
   *
   * @param id Unique job identifier.
   * @param options Listener options, including terminal-state and error callbacks.
   * @returns A cleanup function that stops this job's event stream.
   */
  streamJobEvents: (id: string, options?: StreamJobEventsOptions) => () => void;
  /**
   * Listen to event streams for all currently active jobs without reconnecting jobs that already have a stream.
   *
   * @param options Listener options and lifecycle signal.
   * @returns A cleanup function that stops this batch of listeners.
   */
  watchActiveJobs: (options?: WatchActiveJobsOptions) => () => void;
  /**
   * Abort and clear all ongoing job SSE connections.
   */
  clearJobStreams: () => void;
}

/** Persistent module-level map of job SSE controllers, keyed by job ID. */
const activeStreams = new Map<string, AbortController>();
const eventSequences = new Map<string, number>();

/** End only streams still owned by the current controller so stale cleanup cannot affect resubscriptions. */
function stopOwnedStream(id: string, controller: AbortController): void {
  if (activeStreams.get(id) !== controller) return;
  activeStreams.delete(id);
  controller.abort();
}

/** Global background-job store providing fine-grained job subscriptions and event isolation. */
export const useJobStore = create<JobState>()((set, get) => ({
  jobs: [],
  activeJobIds: [],

  setJobs: (jobs) => {
    if (!jobs.length) eventSequences.clear();
    const current = new Map(get().jobs.map((job) => [job.id, job]));
    const nextJobs = jobs.map((job) => {
      const previous = current.get(job.id);
      return { ...(previous && Date.parse(previous.updated_at) > Date.parse(job.updated_at) ? previous : job) };
    });
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
        if (patch.updated_at && Date.parse(job.updated_at) > Date.parse(patch.updated_at)) return job;
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
          if (event.sequence <= (eventSequences.get(id) ?? 0)) return;
          eventSequences.set(id, event.sequence);
          options?.onEvent?.();
          if (controller.signal.aborted || activeStreams.get(id) !== controller) return;
          const status =
            event.type === 'state' && typeof event.data.status === 'string'
              ? (event.data.status as JobResource['status'])
              : null;
          get().updateJob(id, {
            ...(status ? { status } : {}),
            ...(typeof event.progress === 'number' ? { progress: event.progress } : {}),
            ...(event.type === 'progress' ? { progress_data: event.data } : {}),
            updated_at: event.created_at,
          });
          if (status && TERMINAL_STATUSES.has(status)) {
            stop();
            eventSequences.delete(id);
            const finishedJob = get().jobs.find((j) => j.id === id);
            if (finishedJob) options?.onTerminal?.(finishedJob);
          }
        },
        controller.signal,
        eventSequences.get(id) ?? 0,
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
    for (const id of eventSequences.keys()) if (!currentActive.includes(id)) eventSequences.delete(id);
    const ownedStreams = new Map<string, AbortController>();
    // Start listeners for newly active jobs.
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

    // Stop connections for jobs that are no longer active.
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
