/** 封装 runtime 领域资源请求，不保存组件状态。 */
import type {
  RuntimeCapabilities,
  RuntimeStatus,
  RuntimeResources,
  RuntimeListener,
  RuntimeModel,
  RuntimeInstance,
  RuntimeProfile,
  RuntimeMetricSnapshot,
  RuntimeLogEntry,
  RealtimeCapabilities,
  VoiceOutputComponentStatus,
} from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export interface RuntimeMemoryPolicy {
  total_limit_bytes?: number | null;
  effective_total_limit_bytes?: number | null;
  total_capacity_limit_bytes?: number | null;
  model_limit_bytes: number | null;
  prefix_limit_bytes: number | null;
  prefix_disk_limit_bytes?: number | null;
  prefix_directory: string | null;
  actual_prefix_directory: string;
}

export interface PrefixCacheGroup {
  id: string;
  model_name: string | null;
  architecture: string | null;
  codec: string | null;
  context_size: number | null;
  bytes: number;
  blocks: number;
  prefixes: number;
  block_tokens: number;
  max_prefix_tokens: number;
  incomplete_prefixes: number;
  text_blocks: number;
  last_used_at: string;
}
export interface PrefixCacheBlock {
  id: string;
  parent: string;
  tokens: number;
  prefix_tokens: number;
  complete_chain: boolean;
  text_available: boolean;
  bytes: number;
  last_used_at: string;
}
export interface PrefixCacheInventory {
  directory: string;
  total_bytes: number;
  total_blocks: number;
  can_clear: boolean;
  data: PrefixCacheGroup[];
  blocks: PrefixCacheBlock[];
  offset: number;
  limit: number;
}
export interface PrefixCacheText {
  available: boolean;
  text?: string;
  reason?: string;
  total_tokens?: number;
  offset?: number;
  next_offset?: number | null;
}

export const runtimeApi = {
  inferencePolicy(): Promise<{ mtp_enabled: boolean }> {
    return request('/api/v1/runtime/inference-policy');
  },
  configureInferencePolicy(mtpEnabled: boolean): Promise<{ mtp_enabled: boolean }> {
    return request('/api/v1/runtime/inference-policy', {
      method: 'PUT', body: JSON.stringify({ mtp_enabled: mtpEnabled }),
    });
  },
  prefixCacheEntries(namespace?: string, offset = 0, signal?: AbortSignal): Promise<PrefixCacheInventory> {
    const query = new URLSearchParams({ offset: String(offset), limit: '100' });
    if (namespace) query.set('namespace', namespace);
    return request(`/api/v1/runtime/cache/entries?${query}`, { signal });
  },
  prefixCacheText(namespace: string, block: string, offset = 0): Promise<PrefixCacheText> {
    return request(`/api/v1/runtime/cache/entries/${encodeURIComponent(namespace)}/${encodeURIComponent(block)}/text?offset=${offset}`);
  },
  purgePrefixCache(namespace?: string): Promise<{ released_bytes: number; removed_blocks: number; failed_blocks: number }> {
    return request('/api/v1/runtime/cache/purge', { method: 'POST', body: JSON.stringify({ namespace: namespace ?? null }) });
  },
  runtimeResources(signal?: AbortSignal): Promise<RuntimeResources> {
    return request('/api/v1/runtime/resources', { signal });
  },
  modelAliases(): Promise<{ aliases: Record<string, string> }> {
    return request('/api/v1/runtime/model-aliases');
  },
  configureModelAliases(aliases: Record<string, string>): Promise<{ aliases: Record<string, string> }> {
    return request('/api/v1/runtime/model-aliases', { method: 'PUT', body: JSON.stringify({ aliases }) });
  },
  memoryPolicy(): Promise<RuntimeMemoryPolicy> {
    return request('/api/v1/runtime/memory-policy');
  },
  configureMemoryPolicy(policy: Partial<Pick<RuntimeMemoryPolicy, 'total_limit_bytes' | 'model_limit_bytes' | 'prefix_limit_bytes' | 'prefix_disk_limit_bytes' | 'prefix_directory'>>): Promise<{ operation_id: string }> {
    return request('/api/v1/runtime/memory-policy', { method: 'PUT', body: JSON.stringify(policy) });
  },
  runtimeListener(): Promise<RuntimeListener> {
    return request('/api/v1/runtime/listener');
  },

  configureRuntimeListener(port: number): Promise<RuntimeListener> {
    return request('/api/v1/runtime/listener', {
      method: 'PUT', body: JSON.stringify({ port }),
    });
  },
  /** 读取指定实例或默认实例支持的推理能力。 */
  runtimeCapabilities(instanceId?: string | null): Promise<RuntimeCapabilities> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/capabilities${suffix}`);
  },

  /** 读取运行状态、缓存和性能指标。 */
  runtimeStatus(instanceId?: string | null): Promise<RuntimeStatus> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/status${suffix}`);
  },

  /** 列出当前服务提供的模型名称。 */
  async runtimeModels(): Promise<RuntimeModel[]> {
    return (await request<{ data: RuntimeModel[] }>('/api/v1/runtime/models')).data;
  },

  /** 获取所有加载中或已加载的运行实例。 */
  async runtimeInstances(): Promise<RuntimeInstance[]> {
    return (await request<{ data: RuntimeInstance[] }>('/api/v1/runtime/instances')).data;
  },

  /** 获取模型加载配置档案列表。 */
  async runtimeProfiles(): Promise<RuntimeProfile[]> {
    return (await request<{ data: RuntimeProfile[] }>('/api/v1/runtime/profiles')).data;
  },

  /** 保存模型加载参数为可复用配置档案。 */
  createRuntimeProfile(body: {
    name: string;
    load: RuntimeProfile['load'];
  }): Promise<RuntimeProfile> {
    return request('/api/v1/runtime/profiles', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  /** 删除加载配置档案，不卸载现有实例。 */
  async deleteRuntimeProfile(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/runtime/profiles/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** 按配置档案提交加载任务，可显式接受资产版本漂移。 */
  loadRuntimeProfile(
    id: string,
    allowDrift = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request(`/api/v1/runtime/profiles/${id}/load`, {
      method: 'POST',
      body: JSON.stringify({ allow_drift: allowDrift }),
    });
  },

  /** 获取运行指标快照历史，供概览趋势展示。 */
  async runtimeMetrics(limit = 200, signal?: AbortSignal, since?: string): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(`/api/v1/runtime/metrics?limit=${limit}${since ? `&since=${encodeURIComponent(since)}` : ''}`, { signal })
    ).data;
  },

  /** 获取最近的服务日志。 */
  async runtimeLogs(limit = 100, signal?: AbortSignal, after?: number): Promise<RuntimeLogEntry[]> {
    return (await request<{ data: RuntimeLogEntry[] }>(`/api/v1/runtime/logs?limit=${limit}${after != null ? `&after=${after}` : ''}`, { signal })).data;
  },

  /** 读取语音通道可用性与默认音频参数。 */
  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request('/api/v1/runtime/realtime/capabilities');
  },

  /** 查询语音输出组件的下载和激活状态。 */
  voiceOutputComponent(): Promise<VoiceOutputComponentStatus> {
    return request('/api/v1/components/voice-output');
  },

  /** 提交语音输出组件安装任务，不自动触发下载之外的页面操作。 */
  installVoiceOutputComponent(): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/components/voice-output/install', { method: 'POST' });
  },

  /** 激活已经安装的语音输出组件。 */
  activateVoiceOutputComponent(): Promise<{ active: boolean; reason?: string; error?: string }> {
    return request('/api/v1/components/voice-output/activate', { method: 'POST' });
  },

  /** 使用新的上下文容量重载运行实例。 */
  reloadRuntime(contextSize: number, instanceId?: string): Promise<RuntimeStatus> {
    return request('/api/v1/runtime/reload', {
      method: 'POST',
      body: JSON.stringify({ context_size: contextSize, instance_id: instanceId }),
    });
  },

  /** 清理实例前缀缓存并返回释放后的运行状态。 */
  clearRuntimeCache(instanceId?: string): Promise<RuntimeStatus & { released_snapshots: number }> {
    return request('/api/v1/runtime/cache/clear', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId }),
    });
  },

  /** 将缓存回收到目标容量，返回实际释放的字节数。 */
  trimRuntimeCache(
    targetBytes = 0,
    instanceId?: string,
  ): Promise<RuntimeStatus & { released_bytes: number; target_bytes: number }> {
    return request('/api/v1/runtime/cache/trim', {
      method: 'POST',
      body: JSON.stringify({ target_bytes: targetBytes, instance_id: instanceId }),
    });
  },
};
