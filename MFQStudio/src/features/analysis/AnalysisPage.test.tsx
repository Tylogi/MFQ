import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { AnalysisPage } from './AnalysisPage';
import { modelsApi } from '../../shared/api/resources/models';
import { analysisArtifacts, analysisFixture } from '../../../tests/fixtures/checkpointAnalysis';
import { averageBpw, expertPoints } from './analysisData';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ selectedModel: analysisArtifacts[0].name, connectionRevision: 0 }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { modelArtifacts: vi.fn(), checkpointAnalysis: vi.fn() } }));

const fillRect = vi.fn();

beforeEach(() => {
  fillRect.mockClear();
  vi.mocked(modelsApi.modelArtifacts).mockResolvedValue(analysisArtifacts);
  vi.mocked(modelsApi.checkpointAnalysis).mockImplementation(async id => analysisFixture(id === analysisArtifacts[1].id));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ({ fillRect }) as unknown as CanvasRenderingContext2D);
});

it('keeps the loading state concise', () => {
  vi.mocked(modelsApi.modelArtifacts).mockReturnValue(new Promise(() => {}));
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  expect(screen.getByRole('status')).toHaveTextContent(/^Reading model metadata$/);
});

it('inspects an unloaded checkpoint with four expert heatmaps and projection details', async () => {
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  expect(await screen.findByText('Model structure')).toBeInTheDocument();
  expect(screen.getAllByRole('img').filter(item => item.tagName === 'CANVAS')).toHaveLength(4);
  const experts = analysisFixture().experts;
  for (const projection of [undefined, 'gate', 'up', 'down']) {
    const title = projection ? projection[0].toUpperCase() + projection.slice(1) : 'FFN precision';
    const section = screen.getByRole('heading', { name: title }).closest('section')!;
    expect(section.querySelector('.precision-chart-average')).toHaveTextContent(`Average ${averageBpw(expertPoints(experts, projection).values())!.toFixed(3)} bpw`);
  }
  expect(fillRect.mock.calls.filter(call => call[2] === 2).every(call => call[3] === 7)).toBe(true);
  fireEvent.focus(screen.getAllByRole('img').find(item => item.tagName === 'CANVAS')!);
  expect(screen.getByText('L0 · E0')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Attention & other projections' }));
  expect(screen.getByRole('combobox', { name: 'Select projection' })).toBeInTheDocument();
  expect(screen.getByRole('img', { name: 'All per-layer bpw bar heatmap' })).toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: 'Select projection' })).toHaveValue('all');
  expect(screen.getByRole('heading', { name: 'Selected projection precision' })).toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox', { name: 'Select projection' }), { target: { value: 'gdn:qkv' } });
  expect(screen.getByText('Average 5.200 bpw')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Tensor details'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Search tensors' }), { target: { value: 'block.11.' } });
  expect(screen.getByText('model.block.11.attention.query.weight')).toBeInTheDocument();
  expect(screen.queryByText('model.block.0.linear_attention.qkv.weight')).not.toBeInTheDocument();
});

it('shows Dense FFN total and Gate/Up/Down bars with the absolute grayscale scale', async () => {
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysisFixture(true));
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  expect(within(screen.getByRole('tablist', { name: 'Quantization charts' })).getByRole('tab', { name: 'FFN' })).toHaveAttribute('aria-selected', 'true');
  expect(container.querySelectorAll('.expert-heatmap canvas')).toHaveLength(0);
  const charts = container.querySelectorAll('.dense-ffn-bars .projection-bars');
  expect(charts).toHaveLength(4);
  for (const chart of charts) {
    expect(chart.querySelector('.precision-chart-average')).toHaveTextContent('Average 4.000 bpw');
    expect(chart.querySelector('.precision-legend')).toHaveTextContent('1.08+ bpw');
    expect(chart.querySelectorAll('rect[fill^="rgb"]')).toHaveLength(12);
  }
  expect(screen.getByRole('img', { name: 'FFN per-layer bpw bar heatmap' })).toBeInTheDocument();
  for (const projection of ['gate', 'up', 'down']) {
    expect(screen.getByRole('img', { name: `${projection} per-layer bpw bar heatmap` })).toBeInTheDocument();
  }
  fireEvent.click(screen.getByRole('tab', { name: 'Attention & other projections' }));
  const option = within(screen.getByRole('combobox', { name: 'Select projection' })).getByRole('option', { name: 'GQA · query' });
  expect(option).toHaveValue('attention:query');
  expect(container.querySelector('.analysis-projection-table')).toHaveTextContent('GQA');
});

it('keeps Shared charts collapsed, excludes FFN from All and places both routers in other projections', async () => {
  const analysis = analysisFixture();
  for (const category of ['ffn', 'shared_expert', 'routed_experts']) {
    for (const projection of ['gate', 'up', 'down']) {
      analysis.tensors.push({ name: `model.block.0.${category}.${projection}.weight`, shape: [4, 8],
        category, projection, layer: 0, parameters: 32, stored_bytes: 8, bpw: 2, format: 'NINTv2' });
      analysis.projections.push({ name: projection, category, parameters: 32, stored_bytes: 8,
        average_bpw: 2, tensor_count: 1, formats: ['NINTv2'] });
    }
  }
  for (const [category, bpw] of [['router', 2], ['shared_expert', 16]] as const) {
    analysis.tensors.push({ name: `model.block.0.${category}.router.weight`, shape: [4, 8], category, projection: 'router',
      layer: 0, parameters: 32, stored_bytes: bpw * 4, bpw, format: 'NINTv2' });
    analysis.projections.push({ name: 'router', category, parameters: 32, stored_bytes: bpw * 4, average_bpw: bpw,
      tensor_count: 1, formats: ['NINTv2'] });
  }
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  const shared = container.querySelector('.analysis-shared-ffn') as HTMLDetailsElement;
  expect(shared.open).toBe(false);
  expect(shared.querySelectorAll('.projection-bars')).toHaveLength(4);
  fireEvent.click(within(shared).getByText('Shared experts'));
  expect(shared.open).toBe(true);
  expect(within(shared).getByRole('img', { name: 'Shared FFN per-layer bpw bar heatmap' })).toBeInTheDocument();
  expect(within(shared).getAllByText('Average 2.000 bpw')).toHaveLength(4);
  fireEvent.click(screen.getByRole('tab', { name: 'Attention & other projections' }));
  const choice = screen.getByRole('combobox', { name: 'Select projection' });
  expect(choice).toHaveValue('all');
  for (const option of within(choice).getAllByRole('option')) {
    if (option.getAttribute('value') !== 'shared_expert:router') expect(option.getAttribute('value')).not.toMatch(/^(ffn|shared_expert|routed_experts):/);
  }
  expect(within(choice).getByRole('option', { name: 'Router · router' })).toBeInTheDocument();
  expect(within(choice).getByRole('option', { name: 'Shared expert router · router' })).toBeInTheDocument();
  expect(container.querySelectorAll('.other-projection-bars .precision-legend')).toHaveLength(1);
  fireEvent.focus(container.querySelector('.other-projection-bars rect[role="button"]')!);
  expect(container.querySelector('.other-projection-bars .chart-readout')).toHaveTextContent(`${((32768 * 5.2 + 32 * 18) / (32768 + 64)).toFixed(3)} bpw`);
  fireEvent.change(choice, { target: { value: 'router:router' } });
  fireEvent.click(container.querySelector('.other-projection-bars rect[role="button"]')!);
  expect(container.querySelector('.other-projection-bars .chart-readout')).toHaveTextContent('2.000 bpw');
  fireEvent.change(choice, { target: { value: 'shared_expert:router' } });
  fireEvent.click(container.querySelector('.other-projection-bars rect[role="button"]')!);
  expect(container.querySelector('.other-projection-bars .chart-readout')).toHaveTextContent('16.000 bpw');
  fireEvent.change(choice, { target: { value: 'attention:query' } });
  expect(screen.getByRole('img', { name: 'query per-layer bpw bar heatmap' })).toBeInTheDocument();
  fireEvent.change(choice, { target: { value: 'all' } });
  expect(screen.getByRole('img', { name: 'All per-layer bpw bar heatmap' })).toBeInTheDocument();
});

it('marks incomplete shards without requesting invented precision', async () => {
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  fireEvent.change(screen.getByRole('combobox', { name: 'Select model' }), { target: { value: analysisArtifacts[2].id } });
  expect(await screen.findByRole('alert')).toHaveTextContent('Model shards are incomplete');
  expect(modelsApi.checkpointAnalysis).toHaveBeenCalledTimes(1);
});

it('adds a collapsed KV calculator with projection breakdown and resets it per model', async () => {
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  expect((container.querySelector('.kv-cache-planner') as HTMLDetailsElement).open).toBe(false);
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  expect(screen.getByRole('row', { name: /QSA total cache/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'ctx 8192' }));
  expect(screen.getByRole('spinbutton', { name: 'Expected ctx' })).toHaveValue(8192);
  fireEvent.change(screen.getByRole('combobox', { name: 'Select model' }), { target: { value: analysisArtifacts[1].id } });
  await screen.findByText('qwen3_5');
  expect((container.querySelector('.kv-cache-planner') as HTMLDetailsElement).open).toBe(false);
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  expect(screen.getByRole('spinbutton', { name: 'Expected ctx' })).toHaveValue(4096);
  expect(screen.getByRole('row', { name: /GQA total cache/ })).toBeInTheDocument();
  expect(within(container.querySelector('.kv-cache-planner')!).queryByRole('row', { name: /QSA|GDN|PLE|Indexer/ })).not.toBeInTheDocument();
});

it('does not show a KV calculator for an architecture with only GDN layers', async () => {
  const analysis = analysisFixture(true);
  analysis.attention_distribution = { GDN: Array.from({ length: 12 }, (_, index) => index) };
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  expect(screen.queryByText('KV Cache curve & calculator')).not.toBeInTheDocument();
});

it('expands model configuration details and resets them when switching models', async () => {
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  const details = container.querySelector('.analysis-advanced') as HTMLDetailsElement;
  expect(details.open).toBe(false);
  fireEvent.click(screen.getByText('Advanced'));
  expect(details.open).toBe(true);
  const value = (label: string) => screen.getByText(label).nextElementSibling;
  expect(value('Expert hidden size')).toHaveTextContent('640');
  expect(value('Shared expert count')).toHaveTextContent('1');
  expect(value('GR bottleneck dimension')).toHaveTextContent('320');
  expect(value('FFN activation')).toHaveTextContent('SiLU');
  expect(value('Router activation')).toHaveTextContent('Softmax');
  expect(screen.queryByText('Router weight normalization')).not.toBeInTheDocument();
  expect(value('Router activation')?.closest('section')).toHaveTextContent('FFN & experts');
  expect(screen.queryByText('Attention gate activation')).not.toBeInTheDocument();
  expect(value('GR gate activation')?.closest('section')).toHaveTextContent('Residual streams & gates');
  expect(value('GR gate activation')).toHaveTextContent('Sigmoid');
  expect(value('QSA output gate activation')?.closest('section')).toHaveTextContent('Sparse attention & indexer');
  expect(value('QSA output gate activation')).toHaveTextContent('Sigmoid');
  expect(value('GDN output gate activation')?.closest('section')).toHaveTextContent('GDN');
  expect(value('GDN output gate activation')).toHaveTextContent('Sigmoid');
  const gdn = screen.getByRole('heading', { name: /^GDN$/ }).closest('section')!;
  const ple = screen.getByRole('heading', { name: /^PLE$/ }).closest('section')!;
  const detail = (section: Element, label: string) => [...section.querySelectorAll('dt')].find(node => node.textContent === label)?.nextElementSibling;
  expect(detail(gdn, 'Convolution activation')).toHaveTextContent('SiLU');
  expect(detail(gdn, 'Convolution stride')).toHaveTextContent('1');
  expect(detail(gdn, 'Convolution dilation')).toHaveTextContent('1');
  expect(detail(ple, 'Convolution activation')).toHaveTextContent('SiLU');
  expect(detail(ple, 'Convolution stride')).toHaveTextContent('1');
  expect(detail(ple, 'Convolution dilation')).toHaveTextContent('3');
  expect(gdn.querySelector('.analysis-normalizations')).toHaveTextContent('L2Norm');
  expect(gdn.querySelector('.analysis-normalizations')).toHaveTextContent('d = 128 · ε = 1e-6');
  expect(ple.querySelector('.analysis-normalizations')).toHaveTextContent('GroupedRMSNorm');
  expect(ple.querySelector('.analysis-normalizations')).toHaveTextContent('d = 2,560 · ε = 1e-6 · 4 groups');
  const sparse = screen.getByRole('heading', { name: 'Sparse attention & indexer' }).closest('section')!;
  expect(sparse.querySelector('.analysis-normalizations')).toHaveTextContent('Indexer · Q');
  expect(sparse.querySelector('.analysis-normalizations')).toHaveTextContent('d = 256');
  expect(screen.getByText('L0 · Indexer · Q')).toBeInTheDocument();
  expect(screen.queryByText('block.0.attention.indexer.query_norm')).not.toBeInTheDocument();
  expect(value('Active block limit')).toHaveTextContent('512');
  expect(value('Active token limit')).toHaveTextContent('2,048');
  expect(value('Indexer Q heads')).toHaveTextContent('4');
  fireEvent.change(screen.getByRole('combobox', { name: 'Select model' }), { target: { value: analysisArtifacts[1].id } });
  await screen.findByText('qwen3_5');
  expect((container.querySelector('.analysis-advanced') as HTMLDetailsElement).open).toBe(false);
  expect(value('Dense FFN hidden size')).toHaveTextContent('17,408');
  expect(value('GDN output gate activation')).toHaveTextContent('SiLU');
  expect(screen.getByRole('heading', { name: /^GQA$/ })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /^GQA$/ }).closest('section')?.querySelector('.analysis-normalizations')).toHaveTextContent('GroupedRMSNorm');
  expect(screen.getByRole('heading', { name: /^GQA$/ }).closest('section')?.querySelector('.analysis-normalizations')).toHaveTextContent('d = 256 · ε = 1e-6 · 24 groups');
  expect(screen.getByRole('heading', { name: /^GQA$/ }).closest('section')?.querySelector('.analysis-normalizations')).toHaveTextContent('d = 256 · ε = 1e-6 · 2 groups');
  expect(screen.getByRole('tab', { name: /^GQA$/ })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: /^Attention$/ })).not.toBeInTheDocument();
  expect(screen.queryByText('Expert hidden size')).not.toBeInTheDocument();
  expect(screen.queryByText('GR bottleneck dimension')).not.toBeInTheDocument();
});

it('shows DeepSeek router activation and normalization independently', async () => {
  const analysis = analysisFixture();
  analysis.architecture = 'deepseek_v4';
  analysis.advanced = { ...analysis.advanced, router_activation: 'sqrtsoftplus', router_normalization: 'l1_topk' };
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  expect(screen.getByText('Router activation').nextElementSibling).toHaveTextContent('√Softplus');
  expect(screen.getByText('Router weight normalization').nextElementSibling).toHaveTextContent('L1(√Softplus)');
});

it('shows L1 after a Sigmoid router without applying it to Softmax', async () => {
  const analysis = analysisFixture();
  analysis.advanced = { ...analysis.advanced, router_activation: 'sigmoid', router_normalization: 'l1_topk' };
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  expect(screen.getByText('Router activation').nextElementSibling).toHaveTextContent('Sigmoid');
  expect(screen.getByText('Router weight normalization').nextElementSibling).toHaveTextContent('L1(Sigmoid)');
});

it('shows three budget lines, switches units and reports PLE separately', async () => {
  const analysis = analysisFixture();
  analysis.tensors.push(
    { name: 'gate_up', shape: [100], category: 'routed_experts', projection: 'gate_up', layer: 0, parameters: 100, stored_bytes: 50, bpw: 4, format: 'MFE' },
    { name: 'ple', shape: [2 ** 30], category: 'ple', projection: 'ngram', layer: 0, parameters: 2 ** 30, stored_bytes: 2 ** 31, bpw: 16, format: 'NINTv2' },
  );
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  const chart = container.querySelector('.layer-budget')!;
  expect(chart.querySelectorAll('.budget-line')).toHaveLength(3);
  expect(chart.querySelector('.budget-legend')).toHaveTextContent('FFNAttentionTotal');
  expect(chart.querySelector('.analysis-special-weights')).toHaveTextContent('PLE tables1.07B2.000 GiB16.000');
  fireEvent.focus(chart.querySelector('rect[role="button"]')!);
  const readout = chart.querySelector('.chart-readout')!;
  expect(readout).toHaveTextContent('FFN0.000 GiB · 4.000 bpw');
  expect(readout).toHaveTextContent('Attention0.000 GiB · 5.200 bpw');
  expect(readout).toHaveTextContent('Total0.000 GiB · 5.196 bpw');
  fireEvent.click(screen.getByRole('button', { name: 'GiB' }));
  expect(screen.getByRole('button', { name: 'GiB' })).toHaveAttribute('aria-pressed', 'true');
  expect(chart.querySelector('rect[role="button"]')).toHaveAttribute('aria-label', 'L0: FFN 0.000, Attention 0.000, Total 0.000 GiB');
});

it('shows dense FFN budgets without putting their weights into Attention', async () => {
  const analysis = analysisFixture(true);
  vi.mocked(modelsApi.checkpointAnalysis).mockResolvedValue(analysis);
  const { container } = render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await screen.findByText('Model structure');
  const chart = container.querySelector('.layer-budget')!;
  expect(chart.querySelector('.budget-legend')).toHaveTextContent('FFNAttentionTotal');
  expect(chart).not.toHaveTextContent('MoE');
  fireEvent.focus(chart.querySelector('rect[role="button"]')!);
  expect(chart.querySelector('.budget-series-ffn.budget-readout-series')).toHaveTextContent('4.000 bpw');
  expect(chart.querySelector('.budget-series-attention.budget-readout-series')).toHaveTextContent('5.200 bpw');
});

it('ignores a late response from the previously selected checkpoint', async () => {
  let complete!: (result: ReturnType<typeof analysisFixture>) => void;
  vi.mocked(modelsApi.checkpointAnalysis).mockImplementation(id => id === analysisArtifacts[0].id
    ? new Promise(resolve => { complete = resolve; }) : Promise.resolve(analysisFixture(true)));
  render(<MemoryRouter><AnalysisPage /></MemoryRouter>);
  await waitFor(() => expect(modelsApi.checkpointAnalysis).toHaveBeenCalledTimes(1));
  const signal = vi.mocked(modelsApi.checkpointAnalysis).mock.calls[0][1];
  fireEvent.change(screen.getByRole('combobox', { name: 'Select model' }), { target: { value: analysisArtifacts[1].id } });
  await screen.findByText('Model structure');
  expect(signal?.aborted).toBe(true);
  complete(analysisFixture());
  await waitFor(() => expect(screen.getByText('qwen3_5')).toBeInTheDocument());
  expect(screen.queryByText('qwen4_exp')).not.toBeInTheDocument();
});
