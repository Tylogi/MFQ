import type { ModelArtifact, ModelCacheProfile, RuntimeStatus } from '../../shared/api/types';
import { availableResidentMemory, hasSeparateVram } from '../runtime/memoryArchitecture';
import { estimateCacheBytes, quantizedCacheProfile } from './cacheData';
import type { LoadContextDefaults } from './loadContextDefaults';

function measured(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value) && value >= 0;
}

export function availableLoadMemory(runtime?: RuntimeStatus | null, deviceIds: string[] = []): number | null {
  if (!hasSeparateVram(runtime)) return availableResidentMemory(runtime);
  const ram = runtime?.host_memory_available_bytes;
  if (!measured(ram)) return null;
  const devices = runtime?.gpu_devices ?? [];
  const selected = deviceIds.length ? devices.filter(device => deviceIds.includes(device.id)) : devices.slice(0, 1);
  if (selected.length && (!deviceIds.length || selected.length === deviceIds.length)) {
    if (selected.some(device => !measured(device.memory_available_bytes))) return null;
    return ram + selected.reduce((sum, device) => sum + device.memory_available_bytes!, 0);
  }
  return !deviceIds.length && measured(runtime?.device_free_bytes) ? ram + runtime.device_free_bytes : null;
}

export function estimateLoadMemory(model: ModelArtifact, profile: ModelCacheProfile, settings: LoadContextDefaults) {
  const weights = model.estimated_resident_weight_bytes;
  if (!measured(weights)) return null;
  const requested = settings.context_size ?? 32768;
  const context = Math.min(requested, model.context_capacity ?? profile.max_context, profile.max_context);
  if (!Number.isSafeInteger(context) || context < 1) return null;
  const quantized = !settings.automatic && settings.kv_quantization.enabled
    && profile.components.some(item => item.name === 'raw_kv' && (item.group === 'QSA' || item.group === 'GQA'))
    ? quantizedCacheProfile(profile, settings.kv_quantization) : profile;
  if (!quantized) return null;
  const caches = estimateCacheBytes(quantized, context);
  return { total: weights + caches, weights, caches, context };
}
