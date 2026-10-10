import type { RuntimeStatus } from '../../shared/api/types';

/** The server describes physical memory; the backend name is not a topology. */
export function hasSeparateVram(runtime: RuntimeStatus | null | undefined): boolean {
  return runtime?.memory_architecture === 'discrete'
    || (runtime?.memory_architecture == null && runtime?.unified_memory === false);
}

export function residentMemoryCapacity(runtime: RuntimeStatus | null | undefined): number | null | undefined {
  return runtime?.runtime_memory_effective_budget_bytes ?? runtime?.runtime_memory_budget_bytes
    ?? (hasSeparateVram(runtime) ? runtime?.device_memory_total_bytes : runtime?.host_memory_total_bytes);
}
