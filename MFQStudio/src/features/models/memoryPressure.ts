import type { HubModelVariant, HubSystemProfile } from '../../shared/api/types';

export type WeightRoles = Partial<Record<'dense' | 'experts' | 'embedding', number>>;
export type PressureSystem = Pick<HubSystemProfile, 'memory_pools' | 'physical_memory_bytes' | 'runtime_memory_budget_bytes'>;
export interface PressureRow {
  kind: 'vram' | 'ram' | 'shared';
  required: number | null | undefined;
  available: number | null | undefined;
}

export function variantMemoryPressureRows(variant: HubModelVariant | undefined, system?: PressureSystem, cacheBytes = 0, moe = false) {
  if (!variant) return undefined;
  return memoryPressureRows({ weights: variant.estimated_resident_weight_bytes ?? variant.resident_weight_bytes
      ?? (variant.configuration.required_memory_bytes == null ? null : variant.configuration.required_memory_bytes - cacheBytes),
    roles: variant.estimated_weight_bytes_by_role, system, cacheBytes, moe, available: variant.configuration.available_memory_bytes });
}

export function memoryPressureRows({ weights, roles, system, available, cacheBytes = 0, moe = false }: {
  weights?: number | null;
  roles?: WeightRoles;
  system?: PressureSystem;
  available?: number | null;
  cacheBytes?: number;
  moe?: boolean;
}): PressureRow[] {
  const pools = system?.memory_pools ?? [];
  const vram = pools.find(pool => pool.kind === 'vram');
  if (!vram || pools.some(pool => pool.kind === 'uma')) {
    return [{ kind: 'shared', required: weights == null ? null : weights + cacheBytes,
      available: available ?? system?.runtime_memory_budget_bytes ?? system?.physical_memory_bytes }];
  }
  if (moe || (roles?.experts ?? 0) > 0) {
    return [{ kind: 'vram', required: roles?.dense == null ? null : roles.dense + cacheBytes,
      available: vram.capacity_bytes },
    { kind: 'ram', required: roles?.experts == null || roles?.embedding == null ? null : roles.experts + roles.embedding,
      available: pools.find(pool => pool.kind === 'ram')?.capacity_bytes ?? system?.physical_memory_bytes }];
  }
  return [{ kind: 'vram', required: weights == null ? null : weights + cacheBytes, available: vram.capacity_bytes }];
}
