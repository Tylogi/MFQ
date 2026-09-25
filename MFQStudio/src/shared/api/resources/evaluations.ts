/** 封装 evaluations 领域资源请求，不保存组件状态。 */
import type { DatasetResource, EvaluationResult, EvaluationComparison } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const evaluationsApi = {
  /** 列出服务注册的数据集。 */
  async datasets(): Promise<DatasetResource[]> {
    return (await request<{ data: DatasetResource[] }>('/api/v1/datasets')).data;
  },

  /** 将现有产物注册为可用于评测的数据集。 */
  createDataset(body: {
    name: string;
    kind: DatasetResource['kind'];
    artifact_uri: string;
    source_uri?: string | null;
    revision?: string | null;
    metadata?: Record<string, unknown>;
  }): Promise<DatasetResource> {
    return request('/api/v1/datasets', { method: 'POST', body: JSON.stringify(body) });
  },

  /** 删除数据集注册记录。 */
  async deleteDataset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/datasets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 获取最近的评测结果。 */
  async evaluations(limit = 200): Promise<EvaluationResult[]> {
    return (await request<{ data: EvaluationResult[] }>(`/api/v1/evaluations?limit=${limit}`)).data;
  },

  /** 比较指定评测结果，返回指标差异及比率。 */
  compareEvaluations(evaluationIds: string[]): Promise<EvaluationComparison> {
    return request('/api/v1/evaluations/compare', {
      method: 'POST',
      body: JSON.stringify({ evaluation_ids: evaluationIds }),
    });
  },
};
