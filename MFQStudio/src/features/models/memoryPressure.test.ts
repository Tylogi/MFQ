import { expect, it } from 'vitest';
import { memoryPressureRows, type PressureSystem } from './memoryPressure';

const system: PressureSystem = { memory_pools: [
  { kind: 'vram', capacity_bytes: 24 }, { kind: 'vram', capacity_bytes: 32 }, { kind: 'ram', capacity_bytes: 64 },
] };

it('compares dense plus applied KV to one GPU and experts plus embedding to RAM', () => {
  expect(memoryPressureRows({ weights: 83, roles: { dense: 2, experts: 80, embedding: 1 }, system, cacheBytes: 4 }))
    .toEqual([{ kind: 'vram', required: 6, available: 24 }, { kind: 'ram', required: 81, available: 64 }]);
});

it('keeps dense weights and KV on VRAM, with streamed PLE already excluded from weights', () => {
  expect(memoryPressureRows({ weights: 28, roles: { dense: 27, experts: 0, embedding: 1 }, system, cacheBytes: 4 }))
    .toEqual([{ kind: 'vram', required: 32, available: 24 }]);
});

it('preserves unified CUDA memory without adding a second pressure bar', () => {
  expect(memoryPressureRows({ weights: 83, roles: { dense: 2, experts: 80, embedding: 1 },
    system: { memory_pools: [{ kind: 'uma', capacity_bytes: 128 }] }, available: 96, cacheBytes: 4 }))
    .toEqual([{ kind: 'shared', required: 87, available: 96 }]);
});

it('does not invent a MoE byte split when tensor metadata is unavailable', () => {
  expect(memoryPressureRows({ weights: 83, system, moe: true }))
    .toEqual([{ kind: 'vram', required: null, available: 24 }, { kind: 'ram', required: null, available: 64 }]);
});
