import { expect, it } from 'vitest';
import { hasSeparateVram, residentMemoryCapacity, memoryPoolUsage } from './memoryArchitecture';

it('uses reported topology for CUDA UMA devices and separate VRAM devices', () => {
  expect(hasSeparateVram({ memory_architecture: 'unified', backend: 'cuda' })).toBe(false);
  expect(hasSeparateVram({ memory_architecture: 'discrete', backend: 'cuda' })).toBe(true);
  expect(hasSeparateVram({ memory_architecture: 'unknown', backend: 'cuda' })).toBe(false);
  expect(hasSeparateVram(null)).toBe(false);
  expect(residentMemoryCapacity({ memory_architecture: 'discrete', device_memory_total_bytes: 24, host_memory_total_bytes: 128 })).toBe(24);
  expect(residentMemoryCapacity({ memory_architecture: 'unified', host_memory_total_bytes: 128 })).toBe(128);
  expect(residentMemoryCapacity({ memory_architecture: 'discrete', runtime_memory_effective_budget_bytes: 20, device_memory_total_bytes: 24 })).toBe(20);
  expect(residentMemoryCapacity({ memory_architecture: 'discrete', device_total_bytes: 24 })).toBe(24);
});

it('shares capacity across all weights and KV and respects physically available memory', () => {
  expect(memoryPoolUsage(16, [4, 4], [2, 2], 3)).toEqual({ capacity: 15, used: 12, available: 3, cacheCapacity: 7 });
  expect(memoryPoolUsage(16, [4, 4], [2, 2], 10).available).toBe(4);
  expect(memoryPoolUsage(4, [8], [2])).toEqual({ capacity: 4, used: 10, available: 0, cacheCapacity: 0 });
});

it('excludes other occupancy once across models and retains our allocations when no free memory remains', () => {
  expect(memoryPoolUsage(24, [8, 4], [2, 1], 3)).toEqual({ capacity: 18, used: 15, available: 3, cacheCapacity: 6 });
  expect(memoryPoolUsage(24, [8, 4], [2, 1], 0)).toEqual({ capacity: 15, used: 15, available: 0, cacheCapacity: 3 });
  expect(memoryPoolUsage(14, [8, 4], [2, 1], 3)).toEqual({ capacity: 14, used: 15, available: 0, cacheCapacity: 2 });
  expect(memoryPoolUsage(null, [8], [2], 3)).toEqual({ capacity: 13, used: 10, available: 3, cacheCapacity: 5 });
  expect(memoryPoolUsage(24, [], [], 5)).toEqual({ capacity: 5, used: 0, available: 5, cacheCapacity: 5 });
});

it('keeps unknown telemetry and capacity unknown without drawing fabricated full bars', () => {
  expect(memoryPoolUsage(16, [4, null], [2], 8)).toEqual({ capacity: 16, used: null, available: null, cacheCapacity: null });
  expect(memoryPoolUsage(null, [4], [2])).toEqual({ capacity: null, used: 6, available: null, cacheCapacity: null });
  expect(memoryPoolUsage(16, [4], [null]).available).toBeNull();
});
