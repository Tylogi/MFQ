/** Wrap resource requests for the evaluations domain without storing component state. */
import type { DatasetResource, EvaluationResult, EvaluationComparison } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const evaluationsApi = {
  /** List datasets registered with the server. */
  async datasets(): Promise<DatasetResource[]> {
    return (await request<{ data: DatasetResource[] }>('/api/v1/datasets')).data;
  },

  /** Register an existing artifact as a dataset available for evaluation. */
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

  /** Delete a dataset registration. */
  async deleteDataset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/datasets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Get recent evaluation results. */
  async evaluations(limit = 200): Promise<EvaluationResult[]> {
    return (await request<{ data: EvaluationResult[] }>(`/api/v1/evaluations?limit=${limit}`)).data;
  },

  /** Compare specified evaluation results and return metric differences and ratios. */
  compareEvaluations(evaluationIds: string[]): Promise<EvaluationComparison> {
    return request('/api/v1/evaluations/compare', {
      method: 'POST',
      body: JSON.stringify({ evaluation_ids: evaluationIds }),
    });
  },
};
