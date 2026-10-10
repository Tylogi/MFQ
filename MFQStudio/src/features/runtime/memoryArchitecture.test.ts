import { expect, it } from 'vitest';
import { hasSeparateVram, residentMemoryCapacity } from './memoryArchitecture';

it('uses reported topology for CUDA UMA devices and separate VRAM devices', () => {
  expect(hasSeparateVram({ memory_architecture: 'unified', backend: 'cuda' })).toBe(false);
  expect(hasSeparateVram({ memory_architecture: 'discrete', backend: 'cuda' })).toBe(true);
  expect(hasSeparateVram({ memory_architecture: 'unknown', backend: 'cuda' })).toBe(false);
  expect(hasSeparateVram(null)).toBe(false);
  expect(residentMemoryCapacity({ memory_architecture: 'discrete', device_memory_total_bytes: 24, host_memory_total_bytes: 128 })).toBe(24);
  expect(residentMemoryCapacity({ memory_architecture: 'unified', host_memory_total_bytes: 128 })).toBe(128);
  expect(residentMemoryCapacity({ memory_architecture: 'discrete', runtime_memory_effective_budget_bytes: 20, device_memory_total_bytes: 24 })).toBe(20);
});
