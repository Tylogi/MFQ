/** 封装 models 领域资源请求，不保存组件状态。 */
import type {
  ModelArtifact,
  ModelCacheProfile,
  CheckpointAnalysis,
  ModelDirectoryList,
  HubModelSummary,
  HubModelInfo,
  OfficialModelList,
  ArtifactLineage,
} from '../types';
import { request } from '../client';
import type { KvQuantizationSettings } from './runtime';

export interface ModelLoadContextSettings {
  automatic: boolean;
  ssd_kv: boolean;
  ram_preallocated: boolean;
  vram_preallocated: boolean;
  kv_quantization: KvQuantizationSettings;
}

export interface DownloadDirectoryStatus {
  directory: string | null;
  actual_directory: string;
  default_directory: string;
  disk_total_bytes?: number | null;
  disk_free_bytes?: number | null;
  disk_available_bytes?: number | null;
  download_reserved_bytes?: number;
  reclaimable_prefix_bytes?: number;
  evict_prefix_cache?: boolean;
}

export const modelsApi = {
  downloadDirectory(): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory');
  },
  configureDownloadDirectory(directoryId: string | null): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory', { method: 'PUT', body: JSON.stringify({ directory_id: directoryId }) });
  },
  configureDownloadReclaim(evictPrefixCache: boolean): Promise<DownloadDirectoryStatus> {
    return request('/api/v1/hub/download-directory', { method: 'PUT', body: JSON.stringify({ evict_prefix_cache: evictPrefixCache }) });
  },
  /** 列出模型资产；refresh 为真时要求服务重新扫描。 */
  async modelArtifacts(refresh = false): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>(`/api/v1/models${refresh ? '?refresh=true' : ''}`)
    ).data;
  },

  /** 浏览服务端模型目录，优先使用目录标识，其次使用完整路径。 */
  modelDirectories(directoryId?: string | null, path?: string | null): Promise<ModelDirectoryList> {
    const query = new URLSearchParams();
    if (directoryId) query.set('directory_id', directoryId);
    else if (path?.trim()) query.set('path', path.trim());
    const suffix = query.size ? `?${query.toString()}` : '';
    return request(`/api/v1/models/directories${suffix}`);
  },

  modelArtifactDirectory(modelId: string): Promise<ModelDirectoryList> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/directory`);
  },

  checkpointAnalysis(modelId: string, signal?: AbortSignal): Promise<CheckpointAnalysis> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/analysis`, { signal });
  },

  modelCacheProfile(modelId: string, signal?: AbortSignal): Promise<ModelCacheProfile | null> {
    return request(`/api/v1/models/${encodeURIComponent(modelId)}/cache-profile`, { signal });
  },

  openModelDirectoryInFinder(directoryId: string): Promise<{ opened: boolean }> {
    return request('/api/v1/models/directories/open', {
      method: 'POST', body: JSON.stringify({ directory_id: directoryId }),
    });
  },

  /** 注册目录内的模型资产，返回新增或更新的模型记录。 */
  async registerModelDirectory(directoryId: string): Promise<ModelArtifact[]> {
    return (
      await request<{ data: ModelArtifact[] }>('/api/v1/models/directories/register', {
        method: 'POST',
        body: JSON.stringify({ directory_id: directoryId }),
      })
    ).data;
  },

  /** 提交模型加载任务，配置上下文和空闲卸载策略。 */
  loadModel(
    model: string,
    contextSize: number | null = null,
    prefillChunkSize = 2048,
    policy: { pin?: boolean; idle_ttl_seconds?: number | null; device_ids?: string[]; context_management?: ModelLoadContextSettings } = {},
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/load', {
      method: 'POST',
      body: JSON.stringify({
        model,
        context_size: contextSize,
        prefill_chunk_size: prefillChunkSize,
        pin: policy.pin ?? false,
        ...(policy.context_management ? { context_management: {
          automatic: policy.context_management.automatic, ssd_kv: policy.context_management.ssd_kv,
          ram_preallocated: policy.context_management.ram_preallocated, vram_preallocated: policy.context_management.vram_preallocated,
          kv_quantization: policy.context_management.kv_quantization,
        } } : {}),
        ...(policy.device_ids?.length ? { device_ids: policy.device_ids } : {}),
        idle_ttl_seconds: policy.pin ? null : (policy.idle_ttl_seconds ?? null),
      }),
    });
  },

  /** 提交实例卸载任务；强制卸载需调用方先确认用户意图。 */
  unloadModel(
    instanceId: string,
    force = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/models/unload', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId, force }),
    });
  },

  /** 在指定模型仓库搜索远程模型条目。 */
  async searchHubModels(
    provider: HubModelSummary['provider'],
    query: string,
  ): Promise<HubModelSummary[]> {
    const params = new URLSearchParams({ provider, query, limit: '20' });
    return (await request<{ data: HubModelSummary[] }>(`/api/v1/hub/models?${params}`)).data;
  },

  /** 读取内置官方目录以及服务端评估的设备适配建议。 */
  officialHubModels(refresh = false, signal?: AbortSignal): Promise<OfficialModelList> {
    return request(`/api/v1/hub/official${refresh ? '?refresh=true' : ''}`, { signal });
  },

  /** 解析 owner/repo 或受信任的 Hugging Face、ModelScope 仓库链接。 */
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

  /** 读取远程模型的版本、文件列表与容量信息。 */
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

  /** 删除服务工作区内的指定产物，调用方应先确认。 */
  removeWorkspaceArtifact(uri: string): Promise<Record<string, unknown>> {
    return request('/api/v1/artifacts/remove', {
      method: 'POST',
      body: JSON.stringify({ artifact_uri: uri }),
    });
  },

  /** 获取任务产物及其来源关系，供量化和数据集选择使用。 */
  async artifactLineage(limit = 200): Promise<ArtifactLineage[]> {
    return (await request<{ data: ArtifactLineage[] }>(`/api/v1/artifacts/lineage?limit=${limit}`))
      .data;
  },
};
