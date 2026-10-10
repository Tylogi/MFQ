import { expect, it } from 'vitest';
import type { AnalysisExpert, AnalysisTensor, CheckpointAnalysis } from '../../shared/api/types';
import { averageBpw, expertPoints, layerBudgetSeries, precisionColor, precisionRange, projectionSeries, routerWeightLabel, specialTensorBudgets } from './analysisData';
import { analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';

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

it('keeps one precision range even for large expert banks', () => {
  const analysis = { experts: Array.from({ length: 200_000 }, (_, i) => ({ bpw: i % 2 ? 1.81 : 6.13 })) } as CheckpointAnalysis;
  expect(precisionRange(analysis)).toEqual([1.8, 6.2]);
  expect(precisionColor(1.8, 1.8, 6.2)).not.toBe(precisionColor(6.2, 1.8, 6.2));
});

it('uses a clamped gold-to-copper precision scale that darkens with bpw', () => {
  const colors = [1, 3, 5, 7, 9].map(value => precisionColor(value, 1, 9).match(/\d+/g)!.map(Number));
  expect(colors[0]).toEqual([245, 234, 216]);
  expect(colors.at(-1)).toEqual([126, 70, 29]);
  colors.forEach((color, index) => {
    expect(color[0]).toBeGreaterThan(color[1]);
    expect(color[1]).toBeGreaterThan(color[2]);
    if (index) color.forEach((channel, i) => expect(channel).toBeLessThan(colors[index - 1][i]));
  });
  expect(precisionColor(-1, 1, 9)).toBe(precisionColor(1, 1, 9));
  expect(precisionColor(12, 1, 9)).toBe(precisionColor(9, 1, 9));
});

it('separates MoE and Attention budgets and excludes PLE from every layer total', () => {
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
  expect(first.moe).toEqual({ parameters: 160, storedBytes: 120, bpw: 6 });
  expect(first.attention).toEqual({ parameters: 400, storedBytes: 250, bpw: 5 });
  expect(first.total.parameters).toBe(580);
  expect(first.total.storedBytes).toBe(410);
  expect(first.total.bpw).toBeCloseTo(410 * 8 / 580);
  expect(first.total.bpw).not.toBe(first.moe.bpw! + first.attention.bpw!);
  expect(second.moe).toEqual({ parameters: 0, storedBytes: 0, bpw: null });
  expect(second.attention).toEqual({ parameters: 0, storedBytes: 8, bpw: null });
  expect(second.total.bpw).toBeNull();
  expect(specialTensorBudgets(analysis)).toEqual([
    { category: 'ple', count: 1, parameters: 1000, storedBytes: 1000, bpw: 8 },
    { category: 'ple_projection', count: 1, parameters: 40, storedBytes: 80, bpw: 16 },
  ]);
  expect(specialTensorBudgets({ ...analysis, tensors: [] })).toEqual([]);
});
