/** Wrap resource requests for the models domain without storing component state. */
import type {
  ModelArtifact,
  ModelDirectoryList,
  HubModelSummary,
  HubModelInfo,
  OfficialModelList,
  ArtifactLineage,
} from '../types';
import { request } from '../client';

export const modelsApi = {
  /** List model artifacts; when refresh is true, request a fresh server scan. */
  async modelArtifacts(refresh = false): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>(`/api/v1/models${refresh ? '?refresh=true' : ''}`)
    ).data;
  },

  /** Browse server-side model directories, preferring a directory ID over a full path. */
  modelDirectories(directoryId?: string | null, path?: string | null): Promise<ModelDirectoryList> {
    const query = new URLSearchParams();
    if (directoryId) query.set('directory_id', directoryId);
    else if (path?.trim()) query.set('path', path.trim());
    const suffix = query.size ? `?${query.toString()}` : '';
    return request(`/api/v1/models/directories${suffix}`);
  },

  /** Register model artifacts in a directory and return newly created or updated records. */
  async registerModelDirectory(directoryId: string): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>('/api/v1/models/directories/register', {
        method: 'POST',
        body: JSON.stringify({ directory_id: directoryId }),
      })
    ).data;
  },

  /** Submit a model-loading job with context and idle-unload settings. */
  loadModel(
    model: string,
    contextSize: number,
    prefillChunkSize = 2048,
    policy: { pin?: boolean; idle_ttl_seconds?: number | null } = {},
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/load', {
      method: 'POST',
      body: JSON.stringify({
        model,
        context_size: contextSize,
        prefill_chunk_size: prefillChunkSize,
        pin: policy.pin ?? false,
        idle_ttl_seconds: policy.pin ? null : (policy.idle_ttl_seconds ?? null),
      }),
    });
  },

  /** Submit an instance-unload job; the caller must confirm user intent before forcing unload. */
  unloadModel(
    instanceId: string,
    force = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/unload', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId, force }),
    });
  },

  /** Search for remote model entries in the specified model repository. */
  async searchHubModels(
    provider: HubModelSummary['provider'],
    query: string,
  ): Promise<HubModelSummary[]> {
    const params = new URLSearchParams({ provider, query, limit: '20' });
    return (await request<{ data: HubModelSummary[] }>(`/api/v1/hub/models?${params}`)).data;
  },

  /** Fetch the built-in official catalog and server-evaluated device compatibility recommendations. */
  officialHubModels(refresh = false, signal?: AbortSignal): Promise<OfficialModelList> {
    return request(`/api/v1/hub/official${refresh ? '?refresh=true' : ''}`, { signal });
  },

  /** Resolve an owner/repo reference or a trusted Hugging Face or ModelScope repository URL. */
  resolveHubModel(
    reference: string,
    fallbackProvider: HubModelSummary['provider'],
  ): Promise<HubModelInfo> {
    return request('/api/v1/hub/resolve', {
      method: 'POST',
      body: JSON.stringify({
        reference,
        fallback_provider: fallbackProvider,
      }),
    });
  },

  /** Read version, file list, and size information for a remote model. */
  hubModelInfo(
    provider: HubModelSummary['provider'],
    repoId: string,
    revision?: string,
  ): Promise<HubModelInfo> {
    const [owner, name] = repoId.split('/', 2);
    const suffix = revision ? `?revision=${encodeURIComponent(revision)}` : '';
    return request(
      `/api/v1/hub/models/${provider}/${encodeURIComponent(owner)}/${encodeURIComponent(name)}${suffix}`,
    );
  },

  /** Delete the specified artifact from the server workspace; the caller should confirm first. */
  removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>> {
    return request('/api/v1/artifacts/remove', {
      method: 'POST',
      body: JSON.stringify({ artifact_uri: uri }),
    });
  },

  /** Get job artifacts and their lineage for quantization and dataset selection. */
  async artifactLineage(limit = 200): Promise<ArtifactLineage[]> {
    return (await request<{ data: ArtifactLineage[] }>(`/api/v1/artifacts/lineage?limit=${limit}`))
      .data;
  },
};
