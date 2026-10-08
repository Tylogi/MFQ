import type { ModelCacheProfile } from '../../shared/api/types';

type Component = ModelCacheProfile['components'][number];

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
    compressed_kv: ['压缩 KV Cache', 'Compressed KV cache'], sliding_window: ['滑动窗口 KV Cache', 'Sliding-window KV cache'],
    compression_state_4: ['压缩状态（×4）', 'Compression state (×4)'], compression_state_128: ['压缩状态（×128）', 'Compression state (×128)'],
    fixed_cache: ['固定缓存', 'Fixed cache'],
  };
  const pair = labels[item.name];
  const label = pair ? tr(...pair) : item.name;
  return item.name === 'compressed_kv' && item.ratio ? `${label} (×${item.ratio})` : label;
}
