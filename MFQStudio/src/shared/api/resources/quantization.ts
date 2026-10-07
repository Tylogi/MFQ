import { apiUrl, authorizedHeaders, errorFromResponse, request } from '../client';

export interface QuantizationWorkspace {
  import_directory: string;
  export_directory: string;
  candidates: string[];
  candidate_groups?: Record<string, string[]>;
  official_imatrix_url: string | null;
}

export interface QuantizationFiles {
  path: string;
  parent: string | null;
  data: { name: string; path: string; directory: boolean; byte_size: number | null }[];
}

export interface QuantizationSource {
  path: string;
  architecture: string;
  tensors: number;
  parameters: number;
  format: 'hf' | 'mfq';
  full_precision: boolean;
  source_precisions?: string[];
  eligible_candidates?: string[];
  imatrix_supported?: boolean;
}

export interface SourceLoadingPlan {
  backend: string;
  resident_required_bytes: number;
  available_bytes: number;
  resident_allowed: boolean;
  weights_bytes: number;
  workspace_bytes: number;
}

export const quantizationApi = {
  workspace: () => request<QuantizationWorkspace>('/api/v1/quantization/workspace'),
  configure: (import_directory: string, export_directory: string) => request<QuantizationWorkspace>('/api/v1/quantization/workspace', {
    method: 'PUT', body: JSON.stringify({ import_directory, export_directory }),
  }),
  files: (path: string, signal?: AbortSignal) => request<QuantizationFiles>(`/api/v1/quantization/files?path=${encodeURIComponent(path)}`, { signal }),
  source: (path: string) => request<QuantizationSource>('/api/v1/quantization/source', { method: 'POST', body: JSON.stringify({ path }) }),
  loadingPlan: (path: string, purpose: 'imatrix' | 'wt2', context_size: number) => request<SourceLoadingPlan>('/api/v1/quantization/loading-plan', {
    method: 'POST', body: JSON.stringify({ path, purpose, context_size }),
  }),
  async exportRecipe(id: string) {
    const response = await fetch(apiUrl(`/api/v1/quantization/jobs/${id}/recipe`), { headers: authorizedHeaders() });
    if (!response.ok) throw await errorFromResponse(response);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = `mfq-recipe-${id.slice(0, 8)}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};
