import type { RuntimeStatus, RuntimeMemoryResources, RuntimeInstance } from '../../shared/api/types';

export function gpuMemoryGroups(runtime: RuntimeStatus | null | undefined, instances: RuntimeInstance[]) {
  const devices = new Map((runtime?.gpu_devices ?? []).filter(device => !device.unified_memory).map(device => [device.id, device]));
  for (const item of instances) for (const memory of item.memory?.devices ?? []) {
    const previous = devices.get(memory.device_id);
    devices.set(memory.device_id, { ...previous, id: memory.device_id, name: previous?.name ?? '',
      memory_total_bytes: memory.memory_total_bytes ?? previous?.memory_total_bytes,
      memory_available_bytes: previous?.memory_available_bytes ?? memory.memory_available_bytes });
  }
  if (!devices.size) devices.set('0', { id: '0', name: '',
    memory_total_bytes: runtime?.device_memory_total_bytes ?? runtime?.device_total_bytes,
    memory_available_bytes: runtime?.device_free_bytes });
  return [...devices.values()].map(device => ({ ...device, instances: instances.flatMap(item => {
    const memory = item.memory?.devices?.find(value => value.device_id === device.id);
    if (memory) return [{ ...item, memory: { ...item.memory!, ...memory,
      resident_weight_bytes: memory.resident_weight_bytes ?? null,
      resident_buffer_bytes: memory.resident_buffer_bytes ?? null,
      prefix_cache_bytes: memory.prefix_cache_bytes ?? null,
      kv_bytes: memory.kv_bytes ?? null } }];
    const assigned = item.devices.filter(value => value.startsWith('cuda:')).map(value => value.slice(5));
    if (item.memory?.devices?.length || assigned.length && !assigned.includes(device.id)) return [];
    if (!assigned.length && device.id !== '0') return [];
    // A legacy worker's aggregate is only attributable when it runs on one GPU.
    return [{ ...item, memory: assigned.length > 1 && item.memory ? { ...item.memory,
      resident_weight_bytes: null, resident_buffer_bytes: null, kv_bytes: null, prefix_cache_bytes: null } : item.memory }];
  }) }));
}

export function residentWeightsAndBuffers(memory: RuntimeMemoryResources | null | undefined,
  separate: boolean): number | null {
  const weights = memory?.resident_weight_bytes;
  if (weights == null) return null;
  return weights + (separate ? memory?.resident_buffer_bytes ?? 0 : 0);
}

/** The server describes physical memory; the backend name is not a topology. */
export function hasSeparateVram(runtime: RuntimeStatus | null | undefined): boolean {
  return runtime?.memory_architecture === 'discrete'
    || (runtime?.memory_architecture == null && runtime?.unified_memory === false);
}

export function residentMemoryCapacity(runtime: RuntimeStatus | null | undefined): number | null | undefined {
  return runtime?.runtime_memory_effective_budget_bytes ?? runtime?.runtime_memory_budget_bytes
    ?? (hasSeparateVram(runtime) ? runtime?.device_memory_total_bytes ?? runtime?.device_total_bytes : runtime?.host_memory_total_bytes);
}

export function availableResidentMemory(runtime: RuntimeStatus | null | undefined): number | null {
  const value = runtime?.runtime_memory_headroom_bytes ?? runtime?.host_memory_available_bytes;
  return value != null && Number.isFinite(value) && value >= 0 ? value : null;
}

/** Add our existing allocations to live free memory; the ceiling still caps the pool.
 * Other processes and unreported allocations are excluded from the tier budget. */
export function memoryPoolUsage(capacityBytes: number | null | undefined,
  weights: (number | null)[], caches: (number | null)[], availableBytes?: number | null) {
  const measured = (value: number | null | undefined): value is number =>
    value != null && Number.isFinite(value) && value >= 0;
  const ceiling = measured(capacityBytes) ? capacityBytes : null;
  const total = (amounts: (number | null)[]) => amounts.every(measured)
    ? amounts.reduce<number>((sum, value) => sum + value!, 0) : null;
  const weightBytes = total(weights), cacheBytes = total(caches);
  const used = weightBytes != null && cacheBytes != null ? weightBytes + cacheBytes : null;
  const usable = used != null && measured(availableBytes) ? used + availableBytes : null;
  const capacity = usable != null ? ceiling != null ? Math.min(ceiling, usable) : usable : ceiling;
  const cacheCapacity = capacity != null && weightBytes != null ? Math.max(0, capacity - weightBytes) : null;
  const remaining = capacity != null && used != null ? Math.max(0, capacity - used) : null;
  const available = remaining != null
    ? measured(availableBytes) ? Math.min(remaining, availableBytes) : remaining : null;
  return { capacity, used, available, cacheCapacity };
}
