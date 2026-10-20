import type { AnalysisAdvanced, AnalysisExpert, AnalysisTensor, CheckpointAnalysis } from '../../shared/api/types';

export type Translate = (zh: string, en: string) => string;

export function activationLabel(value?: string | null): string {
  const labels: Record<string, string> = { silu: 'SiLU', sigmoid: 'Sigmoid', softmax: 'Softmax',
    gelu: 'GELU', gelu_pytorch_tanh: 'GELU (tanh)', relu: 'ReLU', sqrtsoftplus: '√Softplus' };
  return value ? labels[value] ?? value : '—';
}

export function routerWeightLabel(info?: AnalysisAdvanced): string {
  const activation = activationLabel(info?.router_activation);
  return info?.router_activation && info.router_activation !== 'softmax' && info.router_normalization === 'l1_topk'
    ? `L1(${activation})` : activation;
}

export function categoryLabel(category: string): string {
  const labels: Record<string, string> = {
    routed_experts: 'Routed experts', shared_expert: 'Shared experts',
    attention: 'Attention', gdn: 'GDN', mhc: 'MHC',
    router: 'Router', ffn: 'Dense FFN', ple: 'PLE tables',
    ple_projection: 'PLE projections', embedding: 'Embedding',
    lm_head: 'LM Head', predictor: 'MTP', vision: 'Vision',
    audio: 'Audio', tts: 'Speech output', other: 'Other',
  };
  return labels[category] ?? category;
}

export function analysisCategoryLabel(analysis: CheckpointAnalysis, category: string, projection?: string): string {
  if (category === 'shared_expert' && projection && /(^|\.)router(\.|$)/.test(projection)) return 'Shared expert router';
  if (category !== 'attention') return categoryLabel(category);
  const ids = new Set(analysis.tensors.filter(item => item.category === category &&
    (!projection || item.projection === projection)).map(item => item.layer));
  const specific = (kind: string) => !['attention', 'unknown', 'full_attention', 'linear_attention', 'gdn'].includes(kind.toLowerCase());
  const kinds = [...new Set(analysis.layers.filter(layer => ids.has(layer.layer)).map(layer => layer.attention_type).filter(specific))];
  if (kinds.length) return kinds.join(' / ');
  const known = Object.keys(analysis.attention_distribution).filter(specific);
  if (known.length) return known.join(' / ');
  const query = analysis.attention_heads, kv = analysis.kv_heads;
  if (query && kv && query % kv === 0) return query === kv ? 'MHA' : 'GQA';
  return 'Unidentified type';
}

export function parameterLabel(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

export const PRECISION_BPW_MIN = 1;
export const PRECISION_BPW_MAX = 8;

export function precisionColor(value: number): string {
  const amount = Math.max(0, Math.min(1, (value - PRECISION_BPW_MIN) / (PRECISION_BPW_MAX - PRECISION_BPW_MIN)));
  const shade = Math.round(245 - 200 * amount);
  return `rgb(${shade},${shade},${shade})`;
}

export function averageBpw(items: Iterable<{ parameters: number; bpw: number | null }>): number | null {
  let parameters = 0, bits = 0;
  for (const item of items) {
    if (item.bpw == null || item.parameters <= 0) continue;
    parameters += item.parameters;
    bits += item.bpw * item.parameters;
  }
  return parameters ? bits / parameters : null;
}

export interface ExpertPoint {
  layer: number;
  expert: number;
  parameters: number;
  storedBytes: number;
  bpw: number;
  parts: AnalysisExpert[];
}

export function expertPoints(experts: AnalysisExpert[], projection?: string): Map<string, ExpertPoint> {
  const points = new Map<string, ExpertPoint>();
  for (const item of experts) {
    if (projection && item.projection !== projection) continue;
    const key = `${item.layer}:${item.expert}`;
    let point = points.get(key);
    if (!point) {
      point = { layer: item.layer, expert: item.expert, parameters: 0, storedBytes: 0, bpw: 0, parts: [] };
      points.set(key, point);
    }
    point.parameters += item.parameters;
    point.storedBytes += item.stored_bytes;
    point.bpw = point.storedBytes * 8 / point.parameters;
    point.parts.push(item);
  }
  return points;
}

export function projectionSeries(tensors: AnalysisTensor[], category: string, projection: string, layers: number) {
  return tensorPrecisionSeries(tensors.filter(item => item.category === category && item.projection === projection), layers);
}

export function isFfnTensor(item: { category: string; projection: string }): boolean {
  return ['ffn', 'routed_experts', 'shared_expert'].includes(item.category) && !/(^|\.)router(\.|$)/.test(item.projection);
}

export function tensorPrecisionSeries(tensors: AnalysisTensor[], layers: number) {
  return Array.from({ length: layers }, (_, layer) => {
    const items = tensors.filter(item => item.layer === layer);
    const parameters = items.reduce((sum, item) => sum + item.parameters, 0);
    const bytes = items.reduce((sum, item) => sum + item.stored_bytes, 0);
    return { layer, bpw: parameters && items.every(item => item.bpw != null) ? bytes * 8 / parameters : null,
      parameters, bytes, formats: [...new Set(items.map(item => item.format))] };
  });
}

export function ffnPrecisionSeries(analysis: CheckpointAnalysis, projection?: string, shared = false) {
  return tensorPrecisionSeries(analysis.tensors.filter(item => isFfnTensor(item) && item.category === (shared ? 'shared_expert' : 'ffn') &&
    (!projection || item.projection === projection)), analysis.layer_count);
}

export function otherProjectionSeries(analysis: CheckpointAnalysis, category?: string, projection?: string) {
  return tensorPrecisionSeries(analysis.tensors.filter(item => !isFfnTensor(item) &&
    (!category || item.category === category) && (!projection || item.projection === projection)), analysis.layer_count);
}

function tensorBudget(items: AnalysisTensor[]) {
  const parameters = items.reduce((sum, item) => sum + item.parameters, 0);
  const storedBytes = items.reduce((sum, item) => sum + item.stored_bytes, 0);
  return { parameters, storedBytes, bpw: parameters && items.every(item => item.bpw != null) ? storedBytes * 8 / parameters : null };
}

export function specialTensorBudgets(analysis: CheckpointAnalysis) {
  return ['ple', 'ple_projection'].map(category => {
    const items = analysis.tensors.filter(item => item.category === category);
    return { category, count: items.length, ...tensorBudget(items) };
  }).filter(item => item.count > 0);
}

export function layerBudgetSeries(analysis: CheckpointAnalysis) {
  const groups = new Map<number, { ffn: AnalysisTensor[]; attention: AnalysisTensor[]; total: AnalysisTensor[] }>();
  for (const item of analysis.tensors) {
    if (item.layer == null || ['ple', 'ple_projection'].includes(item.category)) continue;
    const kind = isFfnTensor(item) ? 'ffn'
      : ['attention', 'gdn'].includes(item.category) ? 'attention' : null;
    let group = groups.get(item.layer);
    if (!group) { group = { ffn: [], attention: [], total: [] }; groups.set(item.layer, group); }
    group.total.push(item);
    if (kind) group[kind].push(item);
  }
  return analysis.layers.map(layer => ({
    ...layer, ffn: tensorBudget(groups.get(layer.layer)?.ffn ?? []), attention: tensorBudget(groups.get(layer.layer)?.attention ?? []),
    total: tensorBudget(groups.get(layer.layer)?.total ?? []),
  }));
}
