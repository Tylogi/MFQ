import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { EvaluationsPage } from './EvaluationsPage';
import { evaluationsApi } from '../../shared/api/resources/evaluations';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { toast } from '../../stores/toastStore';
import type { DatasetResource, EvaluationResult, JobResource, ModelArtifact, RuntimeInstance, OfficialDataset } from '../../shared/api/types';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_: string, en: string) => en }) }));
const state = vi.hoisted(() => ({ instances: [] as RuntimeInstance[], addJob: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../../shared/api/resources/evaluations', () => ({ evaluationsApi: { datasets: vi.fn(), evaluations: vi.fn(), tools: vi.fn(), catalog: vi.fn(), createDataset: vi.fn(), deleteDataset: vi.fn(), compareEvaluations: vi.fn() } }));
vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { modelArtifacts: vi.fn() } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { jobs: vi.fn(), createJob: vi.fn(), cancelJob: vi.fn() } }));

const job = { id: 'job', kind: 'benchmark.inference', status: 'queued', progress: 0, payload: {}, cancel_requested: false } as JobResource;
beforeEach(() => {
  state.instances = [{ id: 'instance', model: 'Qwen model', state: 'ready', mtp_available: true, context_size: 32768 }] as RuntimeInstance[];
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'wt2', kind: 'wikitext2', name: 'WT2', sha256: 'a'.repeat(64) }] as DatasetResource[]);
  vi.mocked(evaluationsApi.evaluations).mockResolvedValue([]);
  vi.mocked(evaluationsApi.catalog).mockResolvedValue([{ id: 'wt2-raw-test', name: 'WikiText-2 raw · test', kind: 'wikitext2', sha256: 'a'.repeat(64), byte_size: 732610, rows: 4358, repository: 'Salesforce/wikitext', revision: 'b'.repeat(40), filename: 'test.parquet', license: 'CC BY-SA 3.0 / GFDL' }] as OfficialDataset[]);
  vi.mocked(evaluationsApi.tools).mockResolvedValue({ workspace_root: '/workspace', api_base: 'http://127.0.0.1:8090/v1', quality_available: true, benchmark_available: true, accuracy_available: true, task_benchmarks: {
    'mmlu-pro-test': { available: true, repository: 'TIGER-AI-Lab/MMLU-Pro', revision: 'a'.repeat(40), protocol: 'official-api-cot', scoring_files: {}, defaults: { available: true, source: 'evaluate_from_api.py', parameters: { protocol: 'mcq', sample_count: 0, max_tokens: 4000, seed: null, temperature: 0, top_p: 1, top_k: 0, num_generations: 1, enable_thinking: false, enable_mtp: false } } },
  } });
  vi.mocked(modelsApi.modelArtifacts).mockResolvedValue([{ id: 'model', name: 'Qwen model', format: 'mfq', complete: true }] as ModelArtifact[]);
  vi.mocked(jobsApi.jobs).mockResolvedValue([]);
  vi.mocked(jobsApi.createJob).mockResolvedValue(job);
});
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); vi.clearAllMocks(); });

it('ignores a submitted job that finishes after leaving the evaluation page', async () => {
  let finish!: (value: JobResource) => void;
  vi.mocked(jobsApi.createJob).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
  const view = render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'Inference benchmark' }));
  fireEvent.click(screen.getByRole('button', { name: 'Start benchmark' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalled());
  const notify = vi.spyOn(toast, 'success');
  view.unmount();
  await act(async () => finish(job));
  expect(state.addJob).not.toHaveBeenCalled();
  expect(notify).not.toHaveBeenCalled();
});

it('does not send late cancellation errors into another page', async () => {
  vi.mocked(jobsApi.jobs).mockResolvedValue([{ ...job, status: 'running', progress: .5 }]);
  let fail!: (error: Error) => void;
  vi.mocked(jobsApi.cancelJob).mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }));
  const view = render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('button', { name: 'Jobs & results' }));
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await waitFor(() => expect(jobsApi.cancelJob).toHaveBeenCalledWith('job'));
  const notify = vi.spyOn(toast, 'error');
  view.unmount();
  await act(async () => fail(new Error('old service error')));
  expect(notify).not.toHaveBeenCalled();
});

it('submits the WT2 dataset and reference contract, not an opaque tool form', async () => {
  state.instances = [];
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.change(screen.getByLabelText('Reference logits file'), { target: { value: 'datasets/ref.logits' } });
  fireEvent.change(screen.getByLabelText('Reference manifest'), { target: { value: 'datasets/ref.json' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run WT2 KLD / Top1' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('evaluate.wikitext2', {
    model: 'Qwen model', dataset_id: 'wt2', reference_logits: 'datasets/ref.logits', reference_manifest: 'datasets/ref.json', context_size: 512, chunks: 8, parallel: 1,
  }));
});

it('preflights duplicate resident weights before allowing quality evaluation', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  expect(await screen.findByText(/Unload it first to avoid loading its weights twice/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Reference logits file'), { target: { value: 'datasets/ref.logits' } });
  fireEvent.change(screen.getByLabelText('Reference manifest'), { target: { value: 'datasets/ref.json' } });
  expect(screen.getByRole('button', { name: 'Run WT2 KLD / Top1' })).toBeDisabled();
});

it('submits exact token lengths, both modes and explicit sampling conditions', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'Inference benchmark' }));
  expect(screen.getByRole('group', { name: 'Candidate model' })).toContainElement(screen.getByLabelText('Loaded model'));
  expect(screen.getByRole('group', { name: 'Test conditions' })).toContainElement(screen.getByLabelText('Test mode'));
  expect(screen.getByRole('group', { name: 'Input sizes' })).toContainElement(screen.getByLabelText('Input token lengths'));
  fireEvent.change(screen.getByLabelText('Input token lengths'), { target: { value: '1024, 4096, 8192' } });
  fireEvent.change(screen.getByLabelText('Output token limit'), { target: { value: '256' } });
  fireEvent.click(screen.getByRole('button', { name: 'Start benchmark' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('benchmark.inference', expect.objectContaining({
    instance_id: 'instance', prompt_tokens: 1024, prompt_token_lengths: [1024, 4096, 8192], output_tokens: 256, mode: 'both', warmup_runs: 1, repetitions: 3, seed: 42, temperature: 0,
  })));
  expect(state.addJob).toHaveBeenCalledWith(job);
});

it('blocks invalid ctx and unavailable MTP rather than silently falling back', async () => {
  state.instances[0].mtp_available = false;
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'Inference benchmark' }));
  expect(screen.getByRole('button', { name: 'Start benchmark' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Test mode'), { target: { value: 'decode' } });
  expect(screen.getByRole('button', { name: 'Start benchmark' })).toBeEnabled();
  fireEvent.change(screen.getByLabelText('Input token lengths'), { target: { value: '1024, 32768' } });
  expect(screen.getByRole('button', { name: 'Start benchmark' })).toBeDisabled();
  expect(jobsApi.createJob).not.toHaveBeenCalled();
});

it('continues polling a running evaluation discovered after returning to the page', async () => {
  vi.useFakeTimers();
  vi.mocked(jobsApi.jobs).mockResolvedValueOnce([{ ...job, status: 'running', progress: .5 }]).mockResolvedValue([{ ...job, status: 'succeeded', progress: 1 }]);
  render(<EvaluationsPage />);
  await act(async () => {});
  fireEvent.click(screen.getByRole('button', { name: 'Jobs & results' }));
  expect(screen.getByText('Running · 50%')).toBeInTheDocument();
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(screen.getByText('Succeeded · 100%')).toBeInTheDocument();
});

it('integrates answer evaluation in the native catalog without an external service', async () => {
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'questions', kind: 'custom', name: 'MMLU-Pro test', sha256: 'b'.repeat(64), metadata: { official_id: 'mmlu-pro-test' }, artifact_uri: 'workspace://test.parquet', byte_size: 4144185, created_at: '2026-10-06', updated_at: '2026-10-06' }]);
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  expect(screen.queryByRole('tab', { name: 'Benchi' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'MMLU-Pro' }));
  fireEvent.click(screen.getByLabelText('Official defaults'));
  fireEvent.change(screen.getByLabelText('Sample count (0 = all)'), { target: { value: '20' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run answer evaluation' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('evaluate.accuracy', expect.objectContaining({ instance_id: 'instance', dataset_id: 'questions', protocol: 'mcq', sample_count: 20, enable_thinking: false, enable_mtp: false })));
});

it('blocks malformed or repeated input lengths', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'Inference benchmark' }));
  for (const input of ['', '1024, 1024', '1024, abc', '0', '1.5']) {
    fireEvent.change(screen.getByLabelText('Input token lengths'), { target: { value: input } });
    expect(screen.getByRole('button', { name: 'Start benchmark' })).toBeDisabled();
  }
});

it('shows per-length timing and question failure details instead of a blended score', async () => {
  const base = { job_id: 'job', parameters: {}, dataset_manifest: {}, hardware_identity: {}, runtime_identity: {}, comparison_key: 'a'.repeat(64) };
  vi.mocked(evaluationsApi.evaluations).mockResolvedValue([
    { ...base, id: 'speed', kind: 'inference_benchmark', model_id: 'Qwen model', created_at: '2026-10-06', metrics: { series: [{ prompt_tokens: 1024, decode_prefill_tps: 2800, decode_decode_tps: 60, decode_ttft_ms: 400, decode_completion_tokens: 128 }, { prompt_tokens: 8192, decode_prefill_tps: 3000, decode_decode_tps: 59, decode_ttft_ms: 2700, decode_completion_tokens: 127 }] } },
    { ...base, id: 'accuracy', kind: 'accuracy_benchmark', model_id: 'Qwen model', created_at: '2026-10-06', metrics: { accuracy: .5, correct: 1, sample_count: 2, questions: [{ id: 'q1', category: 'math', question: 'A question', status: 'parse_error', response: 'I cannot tell', expected: 'B', predicted: null }] } },
  ] as EvaluationResult[]);
  render(<EvaluationsPage />);
  fireEvent.click(screen.getByRole('button', { name: 'Jobs & results' }));
  expect(await screen.findByText('2800.0')).toBeInTheDocument();
  expect(screen.getByText('3000.0')).toBeInTheDocument();
  expect(screen.getByText('50.00%')).toBeInTheDocument();
  const summary = screen.getByText('50.00%').closest('dl');
  expect(summary).toHaveClass('evaluation-metrics');
  expect(summary?.querySelectorAll('dt')).toHaveLength(3);
  expect(summary).toContainElement(screen.getByText('Correct answers'));
  expect(summary).toContainElement(screen.getByText('Questions tested'));
  fireEvent.click(screen.getByText(/Category scores & question results/));
  expect(screen.getByText('Parse error', { selector: 'b' })).toBeInTheDocument();
});

it('offers official downloads without arbitrary dataset registration', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('button', { name: 'Official collections' }));
  expect(screen.queryByRole('button', { name: 'Register test dataset' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Download collection' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('dataset.download', { dataset: 'wt2-raw-test' }));
  expect(evaluationsApi.createDataset).not.toHaveBeenCalled();
});

it('groups the base tests and six task benchmarks by discipline', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  expect(screen.getByRole('heading', { name: 'Basic' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Task benchmarks' })).toBeInTheDocument();
  const groups = { Mathematics: ['AIME2025'], Reasoning: ['MMLU-Pro', 'GPQA-Diamond'], Code: ['LiveCodeBench'], Facts: ['MMLU', 'TruthfulQA'] };
  for (const [group, tasks] of Object.entries(groups)) {
    const section = screen.getByRole('region', { name: group });
    expect(within(section).getAllByRole('tab')).toHaveLength(tasks.length);
    for (const name of tasks) expect(within(section).getByRole('tab', { name })).toBeInTheDocument();
  }
});

it('uses native probabilities without generation settings for MMLU', async () => {
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'mmlu', kind: 'custom', name: 'MMLU test', metadata: { official_id: 'mmlu-test' }, artifact_uri: 'workspace://mmlu.parquet', sha256: 'a'.repeat(64), byte_size: 1, created_at: '2026-10-06T00:00:00Z', updated_at: '2026-10-06T00:00:00Z' }]);
  vi.mocked(evaluationsApi.tools).mockResolvedValue({ api_base: 'http://test/v1', workspace_root: '/workspace', quality_available: true, benchmark_available: true, accuracy_available: true, task_benchmarks: { 'mmlu-test': { available: true, repository: 'hendrycks/test', revision: 'a'.repeat(40), protocol: 'official-5shot-likelihood', scoring_files: {}, defaults: { available: true, source: 'evaluate_flan.py', parameters: { protocol: 'likelihood', sample_count: 0, max_tokens: 1, seed: null, enable_thinking: false, enable_mtp: false } } } } });
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: /^MMLU$/ }));
  await screen.findByRole('option', { name: 'MMLU test' });
  expect(screen.queryByLabelText('Output token limit')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Enable thinking')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('MTP')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Run probability evaluation' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('evaluate.accuracy', expect.objectContaining({ instance_id: 'instance', dataset_id: 'mmlu', protocol: 'likelihood', max_tokens: 1, enable_mtp: false, enable_thinking: false })));
});

it('locks official parameters by default and preserves manual edits when toggled', async () => {
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'questions', kind: 'custom', name: 'MMLU-Pro test', metadata: { official_id: 'mmlu-pro-test' }, artifact_uri: 'workspace://test.parquet', sha256: 'a'.repeat(64), byte_size: 1, created_at: '2026-10-06', updated_at: '2026-10-06' }]);
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'MMLU-Pro' }));
  const toggle = screen.getByLabelText('Official defaults');
  const samples = screen.getByLabelText('Sample count (0 = all)');
  const tokens = screen.getByLabelText('Output token limit');
  expect(toggle).toBeChecked();
  expect(samples).toBeDisabled();
  expect(samples).toHaveValue(0);
  expect(tokens).toBeDisabled();
  expect(tokens).toHaveValue(4000);
  expect(screen.getByLabelText('Enable thinking')).toBeDisabled();
  expect(screen.getByLabelText('Loaded model')).toBeEnabled();
  expect(screen.getByLabelText('Official test dataset')).toBeEnabled();
  fireEvent.click(toggle);
  expect(tokens).toBeEnabled();
  fireEvent.change(samples, { target: { value: '20' } });
  fireEvent.change(tokens, { target: { value: '600' } });
  fireEvent.click(toggle);
  expect(samples).toHaveValue(0);
  expect(tokens).toHaveValue(4000);
  fireEvent.click(screen.getByRole('button', { name: 'Run answer evaluation' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('evaluate.accuracy', expect.objectContaining({ official_defaults: true, sample_count: 0, max_tokens: 4000, seed: null })));
  fireEvent.click(screen.getByRole('button', { name: 'Test setup' }));
  fireEvent.click(toggle);
  expect(samples).toHaveValue(20);
  expect(tokens).toHaveValue(600);
});

it('submits unspecified AIME parameters for model-default resolution without inventing official values', async () => {
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'aime', kind: 'custom', name: 'AIME2025', metadata: { official_id: 'aime-2025' }, artifact_uri: 'workspace://test.parquet', sha256: 'a'.repeat(64), byte_size: 1, created_at: '2026-10-06', updated_at: '2026-10-06' }]);
  vi.mocked(evaluationsApi.tools).mockResolvedValue({ api_base: 'http://test/v1', workspace_root: '/workspace', quality_available: true, benchmark_available: true, accuracy_available: true, task_benchmarks: { 'aime-2025': { available: true, repository: 'eth-sri/matharena', revision: 'a'.repeat(40), protocol: 'official-matharena-aime2025', scoring_files: {}, defaults: { available: true, source: 'configs/competitions/aime/aime_2025.yaml', parameters: { sample_count: 0, max_tokens: null, temperature: null, top_p: null, top_k: null, seed: null, enable_thinking: null, enable_mtp: false, num_generations: 1 } } } } });
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'AIME2025' }));
  expect(screen.getByLabelText('Official defaults')).toBeChecked();
  expect(screen.getByLabelText('Output token limit')).toHaveValue(null);
  expect(screen.getByRole('button', { name: 'Run answer evaluation' })).toBeEnabled();
  expect(screen.getByText(/Empty means no official default/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Run answer evaluation' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('evaluate.accuracy', expect.objectContaining({ official_defaults: true, max_tokens: null, temperature: null, top_p: null, top_k: null })));
  fireEvent.click(screen.getByRole('button', { name: 'Test setup' }));
  fireEvent.click(screen.getByLabelText('Official defaults'));
  expect(screen.getByRole('button', { name: 'Run answer evaluation' })).toBeEnabled();
  expect(screen.getByLabelText('Output token limit')).toHaveValue(4096);
  for (const label of ['Output token limit', 'Temperature', 'Top P', 'Top K']) fireEvent.change(screen.getByLabelText(label), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: 'Run answer evaluation' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenLastCalledWith('evaluate.accuracy', expect.objectContaining({ official_defaults: false, max_tokens: null, temperature: null, top_p: null, top_k: null })));
});

it('renders distinct benchmark monograms and outline category symbols', async () => {
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  for (const [name, letter, suffix] of [['AIME2025', 'A', '2025'], ['LiveCodeBench', 'L', '6'], ['MMLU-Pro', 'M', 'P'], ['GPQA-Diamond', 'G', 'D'], ['MMLU', 'M', ''], ['TruthfulQA', 'T', '']]) {
    const mark = screen.getByRole('tab', { name }).querySelector('.evaluation-benchmark-mark');
    expect(mark).toHaveAttribute('aria-hidden', 'true');
    expect(mark?.querySelector('b')).toHaveTextContent(letter);
    expect(mark?.querySelector('sup')?.textContent || '').toBe(suffix);
  }
  for (const [name, group] of [['Mathematics', 'math'], ['Reasoning', 'reasoning'], ['Code', 'code'], ['Facts', 'facts']]) {
    expect(screen.getByRole('heading', { name }).querySelector('svg')).toHaveAttribute('data-group', group);
  }
});

it('does not reuse MMLU-Pro data or enable an unintegrated official runner', async () => {
  vi.mocked(evaluationsApi.datasets).mockResolvedValue([{ id: 'questions', kind: 'custom', name: 'MMLU-Pro test', metadata: { official_id: 'mmlu-pro-test' }, artifact_uri: 'workspace://questions.parquet', sha256: 'a'.repeat(64), byte_size: 1, created_at: '2026-10-06T00:00:00Z', updated_at: '2026-10-06T00:00:00Z' }]);
  render(<EvaluationsPage />);
  await screen.findByRole('option', { name: 'Qwen model' });
  fireEvent.click(screen.getByRole('tab', { name: 'AIME2025' }));
  expect(screen.queryByRole('option', { name: 'MMLU-Pro test' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Run answer evaluation' })).toBeDisabled();
  expect(screen.getByText(/official runner is not fully integrated/)).toBeInTheDocument();
  expect(jobsApi.createJob).not.toHaveBeenCalled();
});
