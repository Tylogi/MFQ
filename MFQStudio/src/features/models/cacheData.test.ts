import { expect, it } from 'vitest';
import { analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';
import { cacheBreakdown, cacheUsageBytes, estimateCacheBytes, kvOnlyProfile } from './cacheData';

it('keeps QSA raw, indexer and total amounts consistent at every context', () => {
  const profile = analysisFixture().cache_profile!;
  for (const context of [1, 16, 17, 2048, 2049, 4096, 4097, 8192, 262144]) {
    const groups = cacheBreakdown(profile, context);
    expect(groups.reduce((sum, item) => sum + item.usage, 0)).toBe(cacheUsageBytes(profile, context));
    expect(groups.reduce((sum, item) => sum + item.budget, 0)).toBe(estimateCacheBytes(profile, context));
    const qsa = groups[0];
    expect(qsa.name).toBe('QSA');
    expect(qsa.layers).toBe(3);
    expect(qsa.children.map(item => item.name)).toEqual(['raw_kv', 'indexer']);
    for (const item of [qsa, qsa.children[1]]) {
      expect(item.children.reduce((sum, child) => sum + child.usage, 0)).toBe(item.usage);
      expect(item.children.reduce((sum, child) => sum + child.budget, 0)).toBe(item.budget);
    }
  }
});

it('respects the QSA pooling threshold, complete blocks, reserved rows and maximum capacity', () => {
  const profile = analysisFixture().cache_profile!;
  const pooled = (context: number) => cacheBreakdown(profile, context)[0].children[1].children[1];
  expect(pooled(2048).usage).toBe(0);
  expect(pooled(2048).budget).toBe(16 * 1536);
  expect(pooled(2049).usage).toBe(512 * 1536);
  expect(pooled(4097).usage).toBe(1024 * 1536);
  expect(pooled(4097).budget).toBe(1024 * 1536);
  expect(pooled(4100).budget).toBe(2048 * 1536);
  const limited = { ...profile, max_context: 4101 };
  expect(cacheBreakdown(limited, 9000)[0].children[1].children[1].budget).toBe(1026 * 1536);
});

it('excludes GDN and PLE states from KV without losing them from the full runtime budget', () => {
  const original = analysisFixture().cache_profile!;
  const profile = { ...original, fixed_bytes: 1000, fixed_components: [
    { group: 'GDN', name: 'recurrent_state', bytes: 900 }, { group: 'PLE', name: 'convolution_state', bytes: 100 },
  ] };
  const kv = kvOnlyProfile(profile);
  expect(kv.fixed_bytes).toBe(0);
  expect(kv.fixed_components).toEqual([]);
  expect(cacheBreakdown(kv, 4096).map(item => item.name)).toEqual(['QSA']);
  expect(estimateCacheBytes(profile, 4096) - estimateCacheBytes(kv, 4096)).toBe(1000);
});

it('itemizes fixed cache without double counting and handles older metadata', () => {
  const profile = { ...analysisFixture(true).cache_profile!, fixed_bytes: 1000,
    fixed_components: [{ group: 'GQA', name: 'sliding_window', bytes: 800 }] };
  const groups = cacheBreakdown(profile, 17);
  expect(groups.reduce((sum, item) => sum + item.budget, 0)).toBe(estimateCacheBytes(profile, 17));
  expect(groups.reduce((sum, item) => sum + item.usage, 0)).toBe(cacheUsageBytes(profile, 17));
});
