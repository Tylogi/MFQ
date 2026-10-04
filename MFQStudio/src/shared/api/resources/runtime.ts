/** Wrap resource requests for the runtime domain without storing component state. */
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
  model_limit_bytes: number | null;
  prefix_limit_bytes: number | null;
  prefix_directory: string | null;
  actual_prefix_directory: string;
}

export const runtimeApi = {
  runtimeResources(): Promise<RuntimeResources> {
    return request('/api/v1/runtime/resources');
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
  configureMemoryPolicy(policy: Partial<Omit<RuntimeMemoryPolicy, 'actual_prefix_directory'>>): Promise<{ operation_id: string }> {
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
  /** Read inference capabilities supported by the specified or default instance. */
  runtimeCapabilities(instanceId?: string | null): Promise<RuntimeCapabilities> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/capabilities${suffix}`);
  },

  /** Read runtime status, cache state, and performance metrics. */
  runtimeStatus(instanceId?: string | null): Promise<RuntimeStatus> {
    const suffix = instanceId ? `?instance_id=${encodeURIComponent(instanceId)}` : '';
    return request(`/api/v1/runtime/status${suffix}`);
  },

  /** List model names provided by the current server. */
  async runtimeModels(): Promise<RuntimeModel[]> {
    return (await request<{ data: RuntimeModel[] }>('/api/v1/runtime/models')).data;
  },

  /** Get all loading or loaded runtime instances. */
  async runtimeInstances(): Promise<RuntimeInstance[]> {
    return (await request<{ data: RuntimeInstance[] }>('/api/v1/runtime/instances')).data;
  },

  /** Get the list of model-loading profiles. */
  async runtimeProfiles(): Promise<RuntimeProfile[]> {
    return (await request<{ data: RuntimeProfile[] }>('/api/v1/runtime/profiles')).data;
  },

  /** Save model-loading parameters as a reusable profile. */
  createRuntimeProfile(body: {
    name: string;
    load: RuntimeProfile['load'];
  }): Promise<RuntimeProfile> {
    return request('/api/v1/runtime/profiles', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  /** Delete a loading profile without unloading existing instances. */
  async deleteRuntimeProfile(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/v1/runtime/profiles/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  /** Submit a loading job from a profile, optionally accepting artifact version drift. */
  loadRuntimeProfile(
    id: string,
    allowDrift = false,
  ): Promise<{ operation_id: string; status: 'accepted' }> {
    return request(`/api/v1/runtime/profiles/${id}/load`, {
      method: 'POST',
      body: JSON.stringify({ allow_drift: allowDrift }),
    });
  },

  /** Get runtime metric snapshot history for overview trends. */
  async runtimeMetrics(limit = 200): Promise<RuntimeMetricSnapshot[]> {
    return (
      await request<{ data: RuntimeMetricSnapshot[] }>(`/api/v1/runtime/metrics?limit=${limit}`)
    ).data;
  },

  /** Get recent server logs. */
  async runtimeLogs(limit = 100): Promise<RuntimeLogEntry[]> {
    return (await request<{ data: RuntimeLogEntry[] }>(`/api/v1/runtime/logs?limit=${limit}`)).data;
  },

  /** Read voice channel availability and default audio parameters. */
  realtimeCapabilities(): Promise<RealtimeCapabilities> {
    return request('/api/v1/runtime/realtime/capabilities');
  },

  /** Check download and activation status for the voice-output component. */
  voiceOutputComponent(): Promise<VoiceOutputComponentStatus> {
    return request('/api/v1/components/voice-output');
  },

  /** Submit a voice-output component installation job without triggering other page actions. */
  installVoiceOutputComponent(): Promise<{ operation_id: string; status: 'accepted' }> {
    return request('/api/v1/components/voice-output/install', { method: 'POST' });
  },

  /** Activate an installed voice-output component. */
  activateVoiceOutputComponent(): Promise<{ active: boolean; reason?: string; error?: string }> {
    return request('/api/v1/components/voice-output/activate', { method: 'POST' });
  },

  /** Reload a runtime instance with a new context capacity. */
  reloadRuntime(contextSize: number, instanceId?: string): Promise<RuntimeStatus> {
    return request('/api/v1/runtime/reload', {
      method: 'POST',
      body: JSON.stringify({ context_size: contextSize, instance_id: instanceId }),
    });
  },

  /** Clear an instance's prefix cache and return its updated runtime status. */
  clearRuntimeCache(instanceId?: string): Promise<RuntimeStatus & { released_snapshots: number }> {
    return request('/api/v1/runtime/cache/clear', {
      method: 'POST',
      body: JSON.stringify({ instance_id: instanceId }),
    });
  },

  /** Reclaim cache to the target capacity and return the number of bytes actually released. */
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
