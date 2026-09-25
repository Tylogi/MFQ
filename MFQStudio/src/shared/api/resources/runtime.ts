/** 封装 runtime 领域资源请求，不保存组件状态。 */
import type {
  RuntimeCapabilities,
  RuntimeStatus,
  RuntimeModel,
  RuntimeInstance,
  RuntimeProfile,
  RuntimeMetricSnapshot,
  RuntimeLogEntry,
  RealtimeCapabilities,
  VoiceOutputComponentStatus,
} from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const runtimeApi = {
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
  async runtimeMetrics(limit = 200): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(`/api/v1/runtime/metrics?limit=${limit}`)
    ).data;
  },

  /** 获取最近的服务日志。 */
  async runtimeLogs(limit = 100): Promise<RuntimeLogEntry[]> {
    return (await request<{ data: RuntimeLogEntry[] }>(`/api/v1/runtime/logs?limit=${limit}`)).data;
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
