/** Wrap resource requests for the jobs domain without storing component state. */
import type { RuntimeLogEntry, JobResource, JobEventResource, JobKindResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';
import { readEventStream } from '../eventStream';

export const jobsApi = {
  getJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}`);
  },
  /** Get recent background jobs. */
  async jobs(limit = 100): Promise<JobResource[]> {
    return (await request<{ data: JobResource[] }>(`/api/v1/jobs?limit=${limit}`)).data;
  },

  /** Get job types and dynamic form field definitions. */
  async jobKinds(): Promise<JobKindResource[]> {
    return (await request<{ data: JobKindResource[] }>('/api/v1/jobs/kinds')).data;
  },

  /** Submit a background job and return its initial record. */
  createJob(kind: string, payload: Record<string, unknown>): Promise<JobResource> {
    return request('/api/v1/jobs', {
      method: 'POST',
      body: JSON.stringify({ kind, payload }),
    });
  },

  /** Request cancellation of the specified background job. */
  cancelJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/cancel`, { method: 'POST' });
  },

  /** Create a new execution attempt from an existing job. */
  retryJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/retry`, { method: 'POST' });
  },

  /** Remove the specified job history record. */
  async deleteJob(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Clear completed job records without deleting their associated model artifacts. */
  async clearCompletedJobs(): Promise<void> {
    const response = await fetch(apiUrl('/api/v1/jobs/completed'), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Read job history events and convert them into log entries for display. */
  async jobEvents(id: string): Promise<RuntimeLogEntry[]> {
    const response = await request<{
      data: Array<{
        sequence: number;
        level: RuntimeLogEntry['level'];
        message?: string | null;
        data: Record<string, unknown>;
        created_at: string;
      }>;
    }>(`/api/v1/jobs/${id}/events?limit=1000`);
    return response.data
      .filter((event) => event.message)
      .map((event) => ({
        sequence: event.sequence,
        level: event.level,
        message: event.message || '',
        fields: event.data,
        created_at: event.created_at,
      }));
  },

  /** Consume job SSE events continuously; the caller cancels the signal to end the subscription. */
  async streamJobEvents(
    id: string,
    onEvent: (event: JobEventResource) => void,
    signal: AbortSignal,
    after = 0,
  ): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}/events/stream${after > 0 ? `?after=${after}` : ''}`), {
      headers: authorizedHeaders({ Accept: 'text/event-stream' }),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    await readEventStream(response, onEvent, signal);
  },
};
