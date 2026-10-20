import { expect, it } from 'vitest';
import type { AnalysisExpert, AnalysisTensor } from '../../shared/api/types';
import { analysisCategoryLabel, averageBpw, categoryLabel, expertPoints, ffnPrecisionSeries, layerBudgetSeries, otherProjectionSeries, PRECISION_BPW_MIN, PRECISION_BPW_MAX, precisionColor, projectionSeries, routerWeightLabel, specialTensorBudgets } from './analysisData';
import { analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';

it('labels attention projections by their concrete architecture and their actual layers', () => {
  const dense = analysisFixture(true);
  expect(analysisCategoryLabel(dense, 'attention', 'query')).toBe('GQA');
  const mini = { ...dense, architecture: 'minicpmo', attention_heads: 32, kv_heads: 8 };
  expect(analysisCategoryLabel(mini, 'attention', 'query')).toBe('GQA');
  const sparse = analysisFixture();
  expect(analysisCategoryLabel(sparse, 'attention', 'query')).toBe('QSA');
  sparse.layers[3].attention_type = 'MLA';
  sparse.attention_distribution = { MLA: [3], QSA: [7, 11] };
  expect(analysisCategoryLabel(sparse, 'attention', 'query')).toBe('MLA / QSA');
  expect(analysisCategoryLabel(sparse, 'gdn')).toBe('GDN');
});

it('always uses English technical names in tensor and projection lists', () => {
  const analysis = analysisFixture();
  expect(analysisCategoryLabel(analysis, 'router')).toBe('Router');
  expect(analysisCategoryLabel(analysis, 'shared_expert', 'router')).toBe('Shared expert router');
  expect(categoryLabel('embedding')).toBe('Embedding');
  expect(categoryLabel('ple')).toBe('PLE tables');
  expect(categoryLabel('ffn')).toBe('Dense FFN');
});

it('only adds L1 to non-Softmax router scores and preserves operation order', () => {
  expect(routerWeightLabel({ router_activation: 'softmax', router_normalization: 'l1_topk' })).toBe('Softmax');
  expect(routerWeightLabel({ router_activation: 'sigmoid', router_normalization: 'l1_topk' })).toBe('L1(Sigmoid)');
  expect(routerWeightLabel({ router_activation: 'sqrtsoftplus', router_normalization: 'l1_topk' })).toBe('L1(√Softplus)');
  expect(routerWeightLabel({ router_activation: 'sigmoid', router_normalization: 'none' })).toBe('Sigmoid');
  expect(routerWeightLabel({ router_activation: null, router_normalization: 'l1_topk' })).toBe('—');
});

it('weights Gate/Up/Down precision by parameters and keeps sparse expert IDs', () => {
  const experts: AnalysisExpert[] = [
    { layer: 3, expert: 7, projection: 'gate', parameters: 100, stored_bytes: 25, bpw: 2, format: 'NVQ2', granularity: 'cohort' },
    { layer: 3, expert: 7, projection: 'down', parameters: 200, stored_bytes: 200, bpw: 8, format: 'NINTv2', granularity: 'row' },
  ];
  expect(expertPoints(experts).get('3:7')?.bpw).toBe(6);
  expect(expertPoints(experts, 'gate').get('3:7')?.bpw).toBe(2);
  expect(expertPoints(experts).has('3:0')).toBe(false);
  expect(averageBpw(expertPoints(experts).values())).toBe(6);
  expect(averageBpw(expertPoints(experts, 'gate').values())).toBe(2);
});

it('weights chart averages by parameters and excludes missing precision', () => {
  expect(averageBpw([{ parameters: 100, bpw: 2 }, { parameters: 300, bpw: 6 },
    { parameters: 1000, bpw: null }, { parameters: 0, bpw: 32 }])).toBe(5);
  expect(averageBpw([])).toBeNull();
  expect(averageBpw([{ parameters: 100, bpw: 0 }])).toBe(0);
});

it('distinguishes absent projections from a zero-bit precision', () => {
  const tensors: AnalysisTensor[] = [{ name: 'q', shape: [4, 8], layer: 2, category: 'attention', projection: 'query', parameters: 32, stored_bytes: 16, bpw: 4, format: 'NINTv2' }];
  expect(projectionSeries(tensors, 'attention', 'query', 4).map(item => item.bpw)).toEqual([null, null, 4, null]);
});

it('weights Dense FFN totals by parameters and keeps Shared precision separate', () => {
  const analysis = analysisFixture(true);
  analysis.tensors = [
    { name: 'gate', shape: [10, 10], category: 'ffn', projection: 'gate', layer: 0, parameters: 100, stored_bytes: 25, bpw: 2, format: 'NVQ2' },
    { name: 'up', shape: [10, 20], category: 'ffn', projection: 'up', layer: 0, parameters: 200, stored_bytes: 150, bpw: 6, format: 'NINTv2' },
    { name: 'shared', shape: [10, 10], category: 'shared_expert', projection: 'gate', layer: 0, parameters: 100, stored_bytes: 100, bpw: 8, format: 'NINTv2' },
    { name: 'shared_router', shape: [10, 10], category: 'shared_expert', projection: 'router', layer: 0, parameters: 100, stored_bytes: 200, bpw: 16, format: 'BF16' },
  ];
  expect(ffnPrecisionSeries(analysis)[0]).toEqual({ layer: 0, parameters: 300, bytes: 175, bpw: 175 * 8 / 300, formats: ['NVQ2', 'NINTv2'] });
  expect(ffnPrecisionSeries(analysis, 'gate')[0].bpw).toBe(2);
  expect(ffnPrecisionSeries(analysis, undefined, true)[0].bpw).toBe(8);
  expect(ffnPrecisionSeries(analysis, 'down')[0].bpw).toBeNull();
});

it('All includes routers and every non-FFN layer tensor but excludes dense, routed and shared FFN weights', () => {
  const analysis = analysisFixture();
  const tensor = (category: string, parameters: number, stored_bytes: number): AnalysisTensor => ({
    name: category, category, projection: 'query', layer: 0, shape: [parameters], parameters, stored_bytes,
    bpw: stored_bytes * 8 / parameters, format: 'NINTv2',
  });
  analysis.tensors = [
    ...['ffn', 'routed_experts', 'shared_expert'].map(category => tensor(category, 1000, 1000)),
    tensor('router', 1000, 1000), { ...tensor('shared_expert', 100, 200), projection: 'router' },
    tensor('attention', 100, 25), tensor('gdn', 100, 50), tensor('mhc', 100, 100), tensor('ple', 100, 100),
    { ...tensor('embedding', 1000, 2000), layer: null },
  ];
  expect(otherProjectionSeries(analysis)[0]).toEqual({ layer: 0, parameters: 1500, bytes: 1475, bpw: 1475 * 8 / 1500, formats: ['NINTv2'] });
  expect(otherProjectionSeries(analysis, 'attention', 'query')[0].bpw).toBe(2);
  expect(otherProjectionSeries(analysis, 'ffn')[0].bpw).toBeNull();
  expect(otherProjectionSeries(analysis)[1].bpw).toBeNull();
});

it('focuses absolute contrast on 1–8 bpw and saturates every 8-bit or higher precision', () => {
  expect([PRECISION_BPW_MIN, PRECISION_BPW_MAX]).toEqual([1, 8]);
  expect(precisionColor(32)).toBe('rgb(45,45,45)');
  expect(precisionColor(16)).toBe('rgb(45,45,45)');
  expect(precisionColor(8)).toBe('rgb(45,45,45)');
  expect(precisionColor(8.4)).toBe(precisionColor(8.7));
  const shade = (bpw: number) => Number(precisionColor(bpw).match(/\d+/)![0]);
  expect(shade(4) - shade(4.3)).toBeLessThanOrEqual(9);
});

it('uses a clamped grayscale precision scale that darkens with bpw', () => {
  const colors = [1, 2, 4, 6, 8].map(value => precisionColor(value).match(/\d+/g)!.map(Number));
  expect(colors[0]).toEqual([245, 245, 245]);
  expect(colors.at(-1)).toEqual([45, 45, 45]);
  colors.forEach((color, index) => {
    expect(color[0]).toBe(color[1]);
    expect(color[1]).toBe(color[2]);
    if (index) color.forEach((channel, i) => expect(channel).toBeLessThan(colors[index - 1][i]));
  });
  expect(precisionColor(-1)).toBe(precisionColor(1));
  expect(precisionColor(33)).toBe(precisionColor(8));
});

it('separates FFN and Attention budgets and excludes PLE from every layer total', () => {
  const analysis = analysisFixture();
  const tensor = (category: string, parameters: number, stored_bytes: number, layer: number | null = 0): AnalysisTensor =>
    ({ name: category, shape: [parameters], category, projection: 'weight', layer, parameters, stored_bytes, bpw: stored_bytes * 8 / parameters, format: 'NINTv2' });
  analysis.tensors = [
    tensor('routed_experts', 100, 50), tensor('shared_expert', 50, 50), tensor('router', 10, 20),
    tensor('gdn', 100, 100), tensor('attention', 300, 150), tensor('mhc', 20, 40),
    tensor('ple', 1000, 1000), tensor('ple_projection', 40, 80), tensor('embedding', 20, 40, null),
    { ...tensor('attention', 0, 8, 1), bpw: null },
  ];
  const [first, second] = layerBudgetSeries(analysis);
  expect(first.ffn).toEqual({ parameters: 150, storedBytes: 100, bpw: 100 * 8 / 150 });
  expect(first.attention).toEqual({ parameters: 400, storedBytes: 250, bpw: 5 });
  expect(first.total.parameters).toBe(580);
  expect(first.total.storedBytes).toBe(410);
  expect(first.total.bpw).toBeCloseTo(410 * 8 / 580);
  expect(first.total.bpw).not.toBe(first.ffn.bpw! + first.attention.bpw!);
  expect(second.ffn).toEqual({ parameters: 0, storedBytes: 0, bpw: null });
  expect(second.attention).toEqual({ parameters: 0, storedBytes: 8, bpw: null });
  expect(second.total.bpw).toBeNull();
  expect(specialTensorBudgets(analysis)).toEqual([
    { category: 'ple', count: 1, parameters: 1000, storedBytes: 1000, bpw: 8 },
    { category: 'ple_projection', count: 1, parameters: 40, storedBytes: 80, bpw: 16 },
  ]);
  expect(specialTensorBudgets({ ...analysis, tensors: [] })).toEqual([]);
});

it('counts dense FFN Gate, Up, Down and norm separately from Attention in every layer', () => {
  const analysis = analysisFixture(true);
  analysis.tensors = analysis.layers.flatMap(({ layer }) => [
    ...['gate', 'up', 'down'].map(projection => ({ name: `model.block.${layer}.mlp.${projection}.weight`,
      shape: [4, 8], layer, category: 'ffn', projection, parameters: 32, stored_bytes: 16, bpw: 4, format: 'NINTv2' })),
    { name: `model.block.${layer}.mlp.norm.weight`, shape: [8], layer, category: 'ffn', projection: 'norm',
      parameters: 8, stored_bytes: 16, bpw: 16, format: 'F16' },
    { name: `model.block.${layer}.attention.query.weight`, shape: [4, 8], layer, category: 'attention', projection: 'query',
      parameters: 32, stored_bytes: 32, bpw: 8, format: 'NINTv2' },
  ]);
  for (const layer of layerBudgetSeries(analysis)) {
    expect(layer.ffn).toEqual({ parameters: 104, storedBytes: 64, bpw: 64 * 8 / 104 });
    expect(layer.attention).toEqual({ parameters: 32, storedBytes: 32, bpw: 8 });
    expect(layer.total).toEqual({ parameters: 136, storedBytes: 96, bpw: 96 * 8 / 136 });
  }
});
