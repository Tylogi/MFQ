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

export function categoryLabel(category: string, tr: Translate): string {
  const labels: Record<string, [string, string]> = {
    routed_experts: ['路由专家', 'Routed experts'], shared_expert: ['共享专家', 'Shared experts'],
    attention: ['注意力', 'Attention'], gdn: ['GDN', 'GDN'], mhc: ['MHC', 'MHC'],
    router: ['路由器', 'Router'], ffn: ['稠密 FFN', 'Dense FFN'], ple: ['PLE 表', 'PLE tables'],
    ple_projection: ['PLE 投影', 'PLE projections'], embedding: ['嵌入', 'Embedding'],
    lm_head: ['LM Head', 'LM Head'], predictor: ['MTP', 'MTP'], vision: ['视觉', 'Vision'],
    audio: ['音频', 'Audio'], tts: ['语音输出', 'Speech output'], other: ['其他', 'Other'],
  };
  return labels[category] ? tr(...labels[category]) : category;
}

export function parameterLabel(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

export function precisionColor(value: number, min: number, max: number): string {
  const amount = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const low = [224, 236, 251], high = [27, 75, 144];
  return `rgb(${low.map((channel, index) => Math.round(channel + (high[index] - channel) * amount)).join(',')})`;
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
  return Array.from({ length: layers }, (_, layer) => {
    const items = tensors.filter(item => item.layer === layer && item.category === category && item.projection === projection);
    const parameters = items.reduce((sum, item) => sum + item.parameters, 0);
    const bytes = items.reduce((sum, item) => sum + item.stored_bytes, 0);
    return { layer, bpw: parameters && items.every(item => item.bpw != null) ? bytes * 8 / parameters : null,
      parameters, bytes, formats: [...new Set(items.map(item => item.format))] };
  });
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
  const groups = new Map<number, { moe: AnalysisTensor[]; attention: AnalysisTensor[]; total: AnalysisTensor[] }>();
  for (const item of analysis.tensors) {
    if (item.layer == null || ['ple', 'ple_projection'].includes(item.category)) continue;
    const kind = ['routed_experts', 'shared_expert', 'router'].includes(item.category) ? 'moe'
      : ['attention', 'gdn'].includes(item.category) ? 'attention' : null;
    let group = groups.get(item.layer);
    if (!group) { group = { moe: [], attention: [], total: [] }; groups.set(item.layer, group); }
    group.total.push(item);
    if (kind) group[kind].push(item);
  }
  return analysis.layers.map(layer => ({
    ...layer, moe: tensorBudget(groups.get(layer.layer)?.moe ?? []), attention: tensorBudget(groups.get(layer.layer)?.attention ?? []),
    total: tensorBudget(groups.get(layer.layer)?.total ?? []),
  }));
}

export function precisionRange(analysis: CheckpointAnalysis): [number, number] {
  if (!analysis.experts.length) return [0, 8];
  const minimum = analysis.experts.reduce((value, item) => Math.min(value, item.bpw), Infinity);
  const maximum = analysis.experts.reduce((value, item) => Math.max(value, item.bpw), -Infinity);
  return [Math.floor(minimum * 10) / 10, Math.ceil(maximum * 10) / 10];
}
