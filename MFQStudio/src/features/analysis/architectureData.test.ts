import { expect, it } from 'vitest';
import { analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';
import { architectureReference, architectureSchedule, architectureViews } from './architectureData';
const tr = (_zh: string, en: string) => en;

it('derives the attention cycle and available subgraphs from each checkpoint', () => {
  const analysis = analysisFixture();
  expect(architectureSchedule(analysis)).toBe('3 × [3 × GDN → QSA]');
  const views = architectureViews(analysis, tr);
  expect(views.map(view => view.id)).toEqual(['overview', 'gdn', 'qsa', 'ffn', 'residual', 'ple', 'mtp']);
  expect(views.find(view => view.id === 'residual')?.title).toBe('GR');
  expect(views.find(view => view.id === 'ffn')?.nodes.some(node => node.id === 'shared')).toBe(true);
  for (const component of ['gdn', 'qsa', 'ffn']) expect(views.find(view => view.id === component)?.nodes.find(node => node.id === 'in')?.title).toBe('Hidden state from GR');
  expect(views.find(view => view.id === 'residual')?.nodes.find(node => node.id === 'gate')?.title).toBe('Down → SiLU → Up');
  for (const view of views) {
    const ids = new Set(view.nodes.map(node => node.id));
    expect(ids.size).toBe(view.nodes.length);
    for (const edge of view.edges) { expect(ids.has(edge.from)).toBe(true); expect(ids.has(edge.to)).toBe(true); }
    for (const node of view.nodes) {
      expect(node.y + node.height).toBeLessThanOrEqual(view.height);
      if (node.action) expect(views.some(item => item.id === node.action)).toBe(true);
    }
  }
  expect(architectureReference(analysis)?.href).toBe('https://huggingface.co/Qwen/Qwen3.8-Flash-Next');
});

it('does not add PLE, GR, MTP or shared experts to a dense checkpoint without them', () => {
  const analysis = analysisFixture(true);
  analysis.attention_distribution = { GDN: [0, 1, 2], GQA: [3] };
  analysis.layers = analysis.layers.slice(0, 4).map(layer => ({ ...layer, attention_type: layer.layer === 3 ? 'GQA' : 'GDN' }));
  const views = architectureViews(analysis, tr);
  expect(views.map(view => view.id)).toEqual(['overview', 'gdn', 'attention', 'ffn']);
  expect(views.find(view => view.id === 'ffn')?.title).toBe('FFN');
  expect(views.find(view => view.id === 'attention')?.title).toBe('GQA');
  expect(views.find(view => view.id === 'attention')?.nodes.find(node => node.id === 'attn')?.title).toBe('GQA');
  expect(architectureSchedule(analysis)).toBe('GDN × 3 · GQA × 1');
  expect(views.find(view => view.id === 'gdn')?.nodes.find(node => node.id === 'norm')?.title).toBe('GroupedRMSNorm');
  expect(views.find(view => view.id === 'gdn')?.nodes.find(node => node.id === 'norm')?.detail).toBe('⊙ SiLU(Gate)');
  expect(views.find(view => view.id === 'attention')?.nodes.find(node => node.id === 'cache')?.detail).toBe('GroupedRMSNorm · KV Cache');
  expect(views[0].edges.filter(edge => edge.residual).map(edge => [edge.from, edge.to])).toEqual([
    ['streams', 'write-attn'], ['write-attn', 'write-ffn'],
  ]);
  expect(architectureReference(analysis)).toBeNull();
});

it('keeps non-periodic layer distributions and PLE injection layer IDs', () => {
  const analysis = analysisFixture();
  analysis.layers[0].attention_type = 'QSA';
  expect(architectureSchedule(analysis)).toBe('GDN × 9 · QSA × 3');
  analysis.tensors.push({ name: 'ple', layer: 1, category: 'ple', projection: 'ngram', shape: [2, 8], parameters: 16, stored_bytes: 8, bpw: 4, format: 'NINTv2' });
  const views = architectureViews(analysis, tr);
  expect(views[0].nodes.find(node => node.id === 'ple')?.detail).toBe('L1');
  expect(views.find(view => view.id === 'ple')?.nodes.find(node => node.id === 'sum')?.detail).toBe('L1');
});

it('draws the FFN and attention gate activations reported by the model', () => {
  const analysis = analysisFixture();
  analysis.advanced = { ...analysis.advanced, ffn_activation: 'gelu', gdn_gate_activation: 'silu' };
  const views = architectureViews(analysis, tr);
  expect(views.find(view => view.id === 'ffn')?.nodes.find(node => node.id === 'multiply')?.title).toBe('GELU(Gate) ⊙ Up');
  expect(views.find(view => view.id === 'qsa')?.nodes.find(node => node.id === 'gate')?.title).toBe('⊙ Sigmoid(Gate)');
  expect(views.find(view => view.id === 'gdn')?.nodes.find(node => node.id === 'norm')?.title).toBe('GroupedRMSNorm');
  expect(views.find(view => view.id === 'gdn')?.nodes.find(node => node.id === 'norm')?.detail).toBe('⊙ SiLU(Gate)');
  delete analysis.advanced;
  expect(architectureViews(analysis, tr).find(view => view.id === 'ffn')?.nodes.find(node => node.id === 'gate')?.detail).toBe('Activation');
});

it('shows normalization after non-Softmax router activations', () => {
  const analysis = analysisFixture();
  analysis.advanced = { ...analysis.advanced, router_activation: 'sqrtsoftplus', router_normalization: 'l1_topk' };
  const router = architectureViews(analysis, tr).find(view => view.id === 'ffn')?.nodes.find(node => node.id === 'router');
  expect(router?.detail).toBe('L1(√Softplus) · 4 / 32');
});
