import type { RuntimeStatus, RuntimeMemoryResources } from '../../shared/api/types';

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
