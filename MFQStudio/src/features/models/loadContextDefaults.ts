import { getApiBaseUrl } from '../../shared/api/client';
import type { KvQuantizationSettings } from '../../shared/api/resources/runtime';

export interface LoadContextDefaults {
  context_size: number | null;
  automatic: boolean;
  ssd_kv: boolean;
  ram_preallocated: boolean;
  vram_preallocated: boolean;
  kv_quantization: KvQuantizationSettings;
}

export const DEFAULT_LOAD_CONTEXT: LoadContextDefaults = {
  context_size: null, automatic: true, ssd_kv: false,
  ram_preallocated: true, vram_preallocated: true,
  kv_quantization: { enabled: false, bits: 4, algorithm: 'turboquant' },
};

function key() {
  return `mfq.studio.load-context.v1:${getApiBaseUrl() || window.location.origin}`;
}

export function readLoadContext(): LoadContextDefaults {
  try {
    const value = JSON.parse(localStorage.getItem(key()) || '{}');
    if (!value || typeof value !== 'object' || Array.isArray(value)) return structuredClone(DEFAULT_LOAD_CONTEXT);
    const result = structuredClone(DEFAULT_LOAD_CONTEXT);
    if (Number.isSafeInteger(value.context_size) && value.context_size >= 512 && value.context_size <= 2147483647)
      result.context_size = value.context_size;
    for (const field of ['automatic', 'ssd_kv', 'ram_preallocated', 'vram_preallocated'] as const)
      if (typeof value[field] === 'boolean') result[field] = value[field];
    if (typeof value.kv_quantization?.enabled === 'boolean' && [2, 2.5, 3, 3.5, 4, 6, 8].includes(value.kv_quantization.bits))
      result.kv_quantization = { enabled: value.kv_quantization.enabled, bits: value.kv_quantization.bits, algorithm: 'turboquant' };
    if (!result.ssd_kv) result.ram_preallocated = true;
    return result;
  } catch { return structuredClone(DEFAULT_LOAD_CONTEXT); }
}

export function saveLoadContext(value: LoadContextDefaults) {
  try { localStorage.setItem(key(), JSON.stringify(value)); } catch { /* The current page remains usable without storage. */ }
}
