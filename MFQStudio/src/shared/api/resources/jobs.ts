/** 封装 jobs 领域资源请求，不保存组件状态。 */
import type { RuntimeLogEntry, JobResource, JobEventResource, JobKindResource } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';
import { readEventStream } from '../eventStream';

export const jobsApi = {
  /** 获取最近的后台任务。 */
  async jobs(limit = 100): Promise<JobResource[]> {
    return (await request<{ data: JobResource[] }>(`/api/v1/jobs?limit=${limit}`)).data;
  },

  /** 获取任务类型和动态表单字段定义。 */
  async jobKinds(): Promise<JobKindResource[]> {
    return (await request<{ data: JobKindResource[] }>('/api/v1/jobs/kinds')).data;
  },

  /** 提交后台任务并返回初始任务记录。 */
  createJob(kind: string, payload: Record<string, unknown>): Promise<JobResource> {
    return request('/api/v1/jobs', {
      method: 'POST',
      body: JSON.stringify({ kind, payload }),
    });
  },

  /** 请求取消指定后台任务。 */
  cancelJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/cancel`, { method: 'POST' });
  },

  /** 根据已有任务重新创建执行尝试。 */
  retryJob(id: string): Promise<JobResource> {
    return request(`/api/v1/jobs/${id}/retry`, { method: 'POST' });
  },

  /** 移除指定任务历史记录。 */
  async deleteJob(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 清理已经完成的任务记录，不删除对应模型产物。 */
  async clearCompletedJobs(): Promise<void> {
    const response = await fetch(apiUrl('/api/v1/jobs/completed'), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 读取任务历史事件并转换为日志展示条目。 */
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

  /** 持续消费任务 SSE；终止订阅时由调用方取消信号。 */
  async streamJobEvents(
    id: string,
    onEvent: (event: JobEventResource) => void,
    signal: AbortSignal,
  ): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/jobs/${id}/events/stream`), {
      headers: authorizedHeaders({ Accept: 'text/event-stream' }),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    await readEventStream(response, onEvent, signal);
  },
};
