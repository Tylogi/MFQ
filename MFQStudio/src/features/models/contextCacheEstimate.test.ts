import { expect, it } from 'vitest';
import type { ModelCacheProfile } from '../../shared/api/types';
import { contextCacheEstimate, streamingCacheEstimate, quantizedCacheProfile } from './cacheData';

const profile: ModelCacheProfile = { max_context: 262144, fixed_bytes: 1024 + 29978, fixed_components: [
  { group: 'GDN', name: 'recurrent_state', bytes: 1024 },
  { group: 'QSA', name: 'indexer_tail', bytes: 29978 },
], components: [
  { group: 'QSA', name: 'raw_kv', layers: 13, bytes_per_row: 26624, tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16, max_read_rows_per_token: 2048 },
  { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656, tokens_per_row: 4, allocation: 'power_of_two', minimum_rows: 16,
    row_rounding: 'floor' },
] };

it('uses the exact requested capacity, not the native limit or the next allocation bucket', () => {
  const result = contextCacheEstimate(profile, 678920)!;
  expect(result.raw).toBe(678920 * 26624);
  expect(result.indexer).toBe(29978 + 169730 * 6656);
  expect(result.total).toBe(result.raw + result.indexer);
  expect(result.total).toBeLessThan(2 ** 20 * (26624 + 3328 + 6656 / 4));
});

it('counts packed K/V with FP32 norms and never quantizes Indexer or recurrent state', () => {
  const source = { ...profile, components: profile.components.map(item => item.name === 'raw_kv'
    ? { ...item, head_dimension: 256, kv_heads: 2 } : item) };
  const original = contextCacheEstimate(source, 678920)!;
  for (const bits of [2, 2.5, 3, 3.5, 4, 6, 8] as const) {
    const quantized = quantizedCacheProfile(source, { enabled: true, bits, algorithm: 'turboquant' })!;
    const words = 2 + 8 * Math.floor(bits) + 8 * Math.ceil(bits);
    const result = contextCacheEstimate(quantized, 678920)!;
    expect(result.raw).toBe(678920 * 13 * 2 * words * 4);
    expect(result.indexer).toBe(original.indexer);
    expect(quantized.fixed_components).toEqual(source.fixed_components);
    expect(streamingCacheEstimate(quantized, 678920, 2 ** 31)!.rawReadPerToken).toBe(13 * 2 * words * 4 * 2048);
    expect(streamingCacheEstimate(quantized, 678920, 2 ** 31)!.indexerReadPerToken).toBe(0);
  }
  expect(quantizedCacheProfile(profile, { enabled: true, bits: 4, algorithm: 'turboquant' })).toBeNull();
  expect(quantizedCacheProfile(profile, { enabled: false, bits: 4, algorithm: 'turboquant' })).toBe(profile);
});

it('keeps allocation rounding and pooling rules without counting recurrent state', () => {
  const short = contextCacheEstimate(profile, 2048)!;
  expect(short.indexer).toBe(29978 + 512 * 6656);
  const odd = contextCacheEstimate(profile, 4101)!;
  expect(odd.raw).toBe(4101 * 26624);
  expect(odd.indexer).toBe(29978 + 1026 * 6656);
  const gqa = contextCacheEstimate({ ...profile, fixed_bytes: 0, fixed_components: [], components: [profile.components[0]] }, 4096)!;
  expect(gqa.indexer).toBe(0);
  expect(gqa.total).toBe(gqa.raw);
});

it('does not produce numeric estimates from invalid drafts', () => {
  for (const context of [0, -1, NaN, Infinity, 512.5, Number.MAX_SAFE_INTEGER + 1]) {
    expect(contextCacheEstimate(profile, context)).toBeNull();
  }
});

it('caps the full Indexer requirement by the streaming budget and leaves no raw floor when it cannot fit', () => {
  const full = contextCacheEstimate(profile, 678920)!;
  for (const budget of [128 * 2 ** 10, 2 ** 20, full.indexer]) {
    expect(streamingCacheEstimate(profile, 678920, budget)).toEqual({ indexerLimit: budget,
      rawFloor: 0, requiredIndexer: full.indexer, indexerReadPerToken: full.indexer - budget, rawReadPerToken: 52 * 2 ** 20 });
  }
});

it('subtracts shared I/O buffers and never estimates more raw KV than the context contains', () => {
  const full = contextCacheEstimate(profile, 678920)!;
  expect(streamingCacheEstimate(profile, 678920, 4 * 2 ** 30)).toEqual({ indexerLimit: full.indexer,
    rawFloor: 4 * 2 ** 30 - 64 * 2 ** 20 - full.indexer, requiredIndexer: full.indexer, indexerReadPerToken: 0, rawReadPerToken: 52 * 2 ** 20 });
  expect(streamingCacheEstimate(profile, 678920, 21 * 2 ** 30)!.rawFloor).toBe(full.raw);
  expect(streamingCacheEstimate(profile, 512, 2 ** 31)!.rawFloor).toBe(contextCacheEstimate(profile, 512)!.raw);
});

it('does not turn missing or invalid budgets into zero Indexer estimates', () => {
  for (const budget of [0, -1, NaN, Infinity, .5, Number.MAX_SAFE_INTEGER + 1]) {
    expect(streamingCacheEstimate(profile, 678920, budget)).toBeNull();
  }
  expect(streamingCacheEstimate(profile, 0, 2 ** 31)).toBeNull();
});

it('reports only the Indexer overflow without charging the shared I/O buffer as SSD reads', () => {
  const required = contextCacheEstimate(profile, 678920)!.indexer;
  expect(streamingCacheEstimate(profile, 678920, 2 ** 31)!.indexerReadPerToken).toBe(0);
  expect(streamingCacheEstimate(profile, 678920, required)!.indexerReadPerToken).toBe(0);
  expect(streamingCacheEstimate(profile, 678920, required - 1)!.indexerReadPerToken).toBe(1);
  expect(streamingCacheEstimate(profile, 678920, 2 ** 30)!.indexerReadPerToken).toBe(56011034);
});

it('uses each QSA selection limit and already-summed row bytes, independent of the resident budget', () => {
  for (const context of [512, 2048, 678920]) {
    for (const budget of [128 * 2 ** 10, 2 ** 30, 2 ** 31]) {
      expect(streamingCacheEstimate(profile, context, budget)!.rawReadPerToken).toBe(52 * 2 ** 20);
    }
  }
  const mixed = { ...profile, components: [...profile.components,
    { ...profile.components[0], layers: 2, bytes_per_row: 1024, max_read_rows_per_token: 4096 },
    { ...profile.components[0], group: 'GQA', bytes_per_row: 65536 },
  ] };
  expect(streamingCacheEstimate(mixed, 678920, 2 ** 31)!.rawReadPerToken).toBe(56 * 2 ** 20);
});

it('does not fabricate a raw read maximum when selection metadata is absent or invalid', () => {
  for (const rows of [undefined, null, 0, -1, NaN, .5]) {
    const missing = { ...profile, components: [{ ...profile.components[0], max_read_rows_per_token: rows }, profile.components[1]] };
    expect(streamingCacheEstimate(missing, 678920, 2 ** 31)!.rawReadPerToken).toBeNull();
  }
  expect(streamingCacheEstimate({ ...profile, components: [] }, 678920, 2 ** 31)!.rawReadPerToken).toBeNull();
});
