import type { ModelCacheProfile } from '../../shared/api/types';
import type { KvQuantizationSettings } from '../../shared/api/resources/runtime';

type Component = ModelCacheProfile['components'][number];

export function quantizedCacheProfile(profile: ModelCacheProfile, settings: KvQuantizationSettings): ModelCacheProfile | null {
  if (!settings.enabled) return profile;
  const raw = profile.components.filter(item => item.name === 'raw_kv' && (item.group === 'QSA' || item.group === 'GQA'));
  if (!raw.length || raw.some(item => !item.head_dimension || !item.kv_heads || !item.layers)) return null;
  return { ...profile, components: profile.components.map(item => {
    if (!raw.includes(item)) return item;
    const width = 2 ** Math.ceil(Math.log2(item.head_dimension!));
    const words = Math.ceil(width * Math.floor(settings.bits) / 32) + Math.ceil(width * Math.ceil(settings.bits) / 32) + 2;
    return { ...item, bytes_per_row: item.layers! * item.kv_heads! * words * 4 };
  }) };
}

export interface CacheBreakdown {
  id: string;
  name: string;
  layers?: number | null;
  ratio?: number;
  usage: number;
  budget: number;
  children: CacheBreakdown[];
}

function tokensAt(profile: ModelCacheProfile, context: number): number {
  return Math.max(1, Math.min(profile.max_context, context));
}

function componentBytes(profile: ModelCacheProfile, component: Component, context: number, budget: boolean): number {
  const tokens = tokensAt(profile, budget ? Math.floor(context) : context);
  const limit = Math.ceil(profile.max_context / component.tokens_per_row);
  const active = tokens > (component.active_after ?? 0);
  const logical = active ? tokens / component.tokens_per_row : 0;
  const used = component.row_rounding === 'floor' ? Math.floor(logical) : logical;
  if (!budget) return Math.min(limit, used) * component.bytes_per_row;
  const rows = Math.max(component.minimum_rows, Math.ceil(used));
  const capacity = component.allocation === 'power_of_two' && rows > 0 ? 2 ** Math.ceil(Math.log2(rows)) : rows;
  return Math.min(limit, capacity) * component.bytes_per_row;
}

export function estimateCacheBytes(profile: ModelCacheProfile, context: number): number {
  return profile.fixed_bytes + profile.components.reduce((sum, component) => sum + componentBytes(profile, component, context, true), 0);
}

export function cacheUsageBytes(profile: ModelCacheProfile, context: number): number {
  return profile.fixed_bytes + profile.components.reduce((sum, component) => sum + componentBytes(profile, component, context, false), 0);
}

export function kvOnlyProfile(profile: ModelCacheProfile): ModelCacheProfile {
  const excluded = new Set(['GDN', 'PLE']);
  const fixed = profile.fixed_components ?? [];
  return { ...profile, fixed_bytes: Math.max(0, profile.fixed_bytes - fixed.filter(item => excluded.has(item.group)).reduce((sum, item) => sum + item.bytes, 0)),
    fixed_components: fixed.filter(item => !excluded.has(item.group)), components: profile.components.filter(item => !excluded.has(item.group ?? '')) };
}

export function contextCacheEstimate(source: ModelCacheProfile, context: number): { total: number; raw: number; indexer: number } | null {
  if (!Number.isSafeInteger(context) || context < 1) return null;
  const profile = { ...kvOnlyProfile(source), max_context: context };
  let indexer = (profile.fixed_components ?? []).filter(item => item.name.startsWith('indexer')).reduce((sum, item) => sum + item.bytes, 0);
  let raw = profile.fixed_bytes - indexer;
  for (const component of profile.components) {
    const bytes = componentBytes(profile, component, context, true);
    if (component.subgroup === 'indexer' || component.name?.startsWith('indexer')) indexer += bytes;
    else raw += bytes;
  }
  return { total: raw + indexer, raw, indexer };
}

export function streamingCacheEstimate(profile: ModelCacheProfile, context: number, budget: number): {
  indexerLimit: number; rawFloor: number; requiredIndexer: number; indexerReadPerToken: number; rawReadPerToken: number | null;
} | null {
  if (!Number.isSafeInteger(budget) || budget <= 0) return null;
  const full = contextCacheEstimate(profile, context);
  if (!full) return null;
  const buffer = Math.min(64 * 2 ** 20, Math.floor(budget / 4));
  const raw = profile.components.filter(component => component.group === 'QSA' && component.name === 'raw_kv');
  const rawReadPerToken = raw.length && raw.every(component => Number.isSafeInteger(component.max_read_rows_per_token)
    && component.max_read_rows_per_token! > 0)
    ? raw.reduce((sum, component) => sum + component.bytes_per_row * component.max_read_rows_per_token!, 0) : null;
  return { indexerLimit: Math.min(budget, full.indexer),
    rawFloor: Math.min(full.raw, Math.max(0, budget - buffer - full.indexer)), requiredIndexer: full.indexer,
    indexerReadPerToken: Math.max(0, full.indexer - budget), rawReadPerToken };
}

export function cacheBreakdown(profile: ModelCacheProfile, context: number): CacheBreakdown[] {
  const groups = new Map<string, CacheBreakdown>();
  function group(name: string, layers?: number | null) {
    let item = groups.get(name);
    if (!item) { item = { id: name, name, usage: 0, budget: 0, children: [] }; groups.set(name, item); }
    if (layers != null) item.layers = Math.max(item.layers ?? 0, layers);
    return item;
  }
  function add(parent: CacheBreakdown, item: CacheBreakdown) {
    parent.children.push(item); parent.usage += item.usage; parent.budget += item.budget;
  }
  profile.components.forEach((component, index) => {
    const parent = group(component.group ?? 'KV', component.layers);
    const item: CacheBreakdown = { id: `${parent.id}:${index}`, name: component.name ?? 'raw_kv', layers: component.layers,
      ratio: component.tokens_per_row, usage: componentBytes(profile, component, context, false),
      budget: componentBytes(profile, component, context, true), children: [] };
    if (component.subgroup) {
      let subgroup = parent.children.find(child => child.name === component.subgroup);
      if (!subgroup) { subgroup = { id: `${parent.id}:${component.subgroup}`, name: component.subgroup, usage: 0, budget: 0, children: [] }; parent.children.push(subgroup); }
      add(subgroup, item); parent.usage += item.usage; parent.budget += item.budget;
    } else add(parent, item);
  });
  const fixed = profile.fixed_components ?? [];
  fixed.forEach((component, index) => add(group(component.group, component.layers), { id: `fixed:${index}`, name: component.name,
    layers: component.layers, usage: component.bytes, budget: component.bytes, children: [] }));
  const remainder = Math.max(0, profile.fixed_bytes - fixed.reduce((sum, item) => sum + item.bytes, 0));
  if (remainder) add(group('KV'), { id: 'fixed', name: 'fixed_cache', usage: remainder, budget: remainder, children: [] });
  return [...groups.values()];
}

export function cacheComponentLabel(item: Pick<CacheBreakdown, 'name' | 'ratio'>, tr: (zh: string, en: string) => string): string {
  const labels: Record<string, [string, string]> = {
    raw_kv: ['原始 KV Cache', 'Raw KV cache'], indexer: ['Indexer Cache', 'Indexer cache'],
    indexer_key: ['Indexer Key Cache', 'Indexer key cache'], indexer_pooled: ['Indexer 压缩块 Cache', 'Indexer pooled cache'],
    indexer_tail: ['Indexer 尾部状态', 'Indexer tail state'],
    compressed_kv: ['压缩 KV Cache', 'Compressed KV cache'], sliding_window: ['滑动窗口 KV Cache', 'Sliding-window KV cache'],
    compression_state_4: ['压缩状态（×4）', 'Compression state (×4)'], compression_state_128: ['压缩状态（×128）', 'Compression state (×128)'],
    fixed_cache: ['固定缓存', 'Fixed cache'],
  };
  const pair = labels[item.name];
  const label = pair ? tr(...pair) : item.name;
  return item.name === 'compressed_kv' && item.ratio ? `${label} (×${item.ratio})` : label;
}
