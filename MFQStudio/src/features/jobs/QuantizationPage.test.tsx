import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { QuantizationPage } from './QuantizationPage';
import { quantizationApi } from '../../shared/api/resources/quantization';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useJobStore } from '../../stores/jobStore';
import type { JobResource } from '../../shared/api/types';
import { ApiError } from '../../shared/api/client';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../shared/api/resources/quantization', () => ({ quantizationApi: { workspace: vi.fn(), configure: vi.fn(), source: vi.fn(), files: vi.fn(), exportRecipe: vi.fn(), loadingPlan: vi.fn() } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { jobs: vi.fn(), createJob: vi.fn(), cancelJob: vi.fn(), jobEvents: vi.fn() } }));

const settings = { import_directory: '/models', export_directory: '/outputs', candidates: ['NVQ3J-L', 'NINT4', 'NINT8'], official_imatrix_url: null };
beforeEach(() => {
  vi.clearAllMocks(); useJobStore.getState().setJobs([]);
  vi.mocked(quantizationApi.workspace).mockResolvedValue(settings);
  vi.mocked(jobsApi.jobs).mockResolvedValue([]);
  vi.mocked(quantizationApi.source).mockResolvedValue({ path: '/models/full', architecture: 'qwen4_exp', tensors: 12, parameters: 1000, format: 'hf', full_precision: true });
});

it('uses one workspace with all candidates selected and deliberate placeholders', async () => {
  const { container } = render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  expect(container.querySelectorAll('.quantization-workbench')).toHaveLength(1);
  expect(screen.getAllByRole('checkbox', { hidden: true }).every((item) => (item as HTMLInputElement).checked)).toBe(true);
  expect(screen.getByRole('button', { name: /Official imatrix/ })).toBeDisabled();
  expect(screen.getByRole('radio', { name: 'Preserve service' })).toBeChecked();
  fireEvent.click(screen.getByRole('button', { name: 'Calibrated' }));
  expect(screen.getByText(/Calibrated allocation is not selected/)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Generate recipe' })).not.toBeInTheDocument();
});

it('saves default directories on the server and browses from the default import directory', async () => {
  vi.mocked(quantizationApi.configure).mockResolvedValue({ ...settings, import_directory: '/inputs', export_directory: '/exports' });
  vi.mocked(quantizationApi.files).mockResolvedValue({ path: '/inputs', parent: '/', data: [] });
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.click(screen.getByText('Default directories'));
  fireEvent.change(screen.getByLabelText('Default import directory'), { target: { value: '/inputs' } });
  fireEvent.change(screen.getByLabelText('Default export directory'), { target: { value: '/exports' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save default directories' }));
  await waitFor(() => expect(quantizationApi.configure).toHaveBeenCalledWith('/inputs', '/exports'));
  await screen.findByText('Default directories saved');
  fireEvent.click(screen.getByRole('button', { name: 'Browse' }));
  await waitFor(() => expect(quantizationApi.files).toHaveBeenCalledWith('/inputs', expect.any(AbortSignal)));
});

it('registers an AlphaQ recipe task with selected candidates and shared service policy', async () => {
  vi.mocked(jobsApi.createJob).mockResolvedValue({ id: 'job1', kind: 'quantization.recipe', status: 'queued', progress: 0, payload: {}, cancel_requested: false, created_at: '', updated_at: '' });
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/full' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('qwen4_exp');
  expect(screen.getByLabelText('Complete model output path')).toHaveValue('/outputs/models/full-quantized.mfq');
  fireEvent.click(screen.getByRole('button', { name: 'Generate recipe' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('quantization.recipe', expect.objectContaining({ input: '/models/full', target_bpw: 4, candidates: settings.candidates, service_policy: 'preserve', output: expect.stringMatching(/^\/outputs\/recipe-/) })));
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0');
});

it('explains automatic chat templates, defaults on, and submits the user opt-out', async () => {
  vi.mocked(quantizationApi.source).mockResolvedValue({ path: '/models/full', architecture: 'qwen3_5', tensors: 12, parameters: 1000, format: 'hf', full_precision: true, imatrix_supported: true });
  vi.mocked(jobsApi.createJob).mockResolvedValue({ id: 'matrix1', kind: 'quantization.imatrix', status: 'queued', progress: 0, payload: {}, cancel_requested: false, created_at: '', updated_at: '' });
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/full' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('qwen3_5');
  fireEvent.click(screen.getByText('Generate from original model'));
  expect(screen.getByRole('checkbox', { name: 'Layerwise' })).toBeChecked();
  const checkbox = screen.getByRole('checkbox', { name: 'Automatically apply model chat template' });
  expect(checkbox).toBeChecked();
  expect(screen.getByText(/already formatted records are not wrapped again/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Calibration corpus'), { target: { value: '/inputs/data.jsonl' } });
  fireEvent.click(screen.getByRole('button', { name: 'Generate imatrix' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenLastCalledWith('quantization.imatrix', expect.objectContaining({ apply_chat_template: true })));
  fireEvent.click(checkbox);
  expect(screen.getByText(/Automatic chat templates are disabled/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Generate imatrix' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenLastCalledWith('quantization.imatrix', expect.objectContaining({ apply_chat_template: false })));
});

it('reuses a completed imported recipe and fits to an overridden output path', async () => {
  const job: JobResource = { id: 'recipe1', kind: 'quantization.import', status: 'succeeded', progress: 1, cancel_requested: false, created_at: '', updated_at: '', payload: {}, result: { artifact_kind: 'recipe', output: '/external/legal.json', method: 'DF-V1-AlphaQ' } };
  vi.mocked(jobsApi.jobs).mockResolvedValue([job]);
  vi.mocked(jobsApi.createJob).mockResolvedValue({ ...job, id: 'fit1', kind: 'quantization.fit', status: 'queued', progress: 0 });
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/full' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('qwen4_exp');
  fireEvent.click(screen.getByRole('button', { name: 'Use this artifact' }));
  fireEvent.change(screen.getByLabelText('Complete model output path'), { target: { value: '/custom/output.mfq' } });
  fireEvent.click(screen.getByRole('radio', { name: 'Allow service impact' }));
  fireEvent.click(screen.getByRole('button', { name: 'Register quantization task' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('quantization.fit', expect.objectContaining({ recipe: '/external/legal.json', output: '/custom/output.mfq', service_policy: 'allow', imatrix: null })));
});

it('shows busy-service progress from the global job store after returning to the page', async () => {
  vi.mocked(jobsApi.jobs).mockResolvedValue([{ id: 'fit1', kind: 'quantization.fit', status: 'running', progress: .4, progress_data: { phase: 'waiting' }, payload: {}, cancel_requested: false, created_at: '', updated_at: '' }]);
  render(<QuantizationPage />);
  await screen.findByText('Waiting for quiet service or available resources');
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.4');
});

it.each(['FP8-SQ', 'MXFP8-SQ'] as const)('groups %s candidates and only submits source-compatible formats', async (family) => {
  const fp8 = Array.from({ length: 8 }, (_, index) => `FP8-SQ-${index + 1}`);
  const mxfp8 = Array.from({ length: 8 }, (_, index) => `MXFP8-SQ-${index + 1}`);
  const native = family === 'FP8-SQ' ? fp8 : mxfp8;
  const incompatible = family === 'FP8-SQ' ? mxfp8 : fp8;
  const catalog = [...settings.candidates, 'MXFP4-SQ-1', 'MXFP4-SQ-F', ...mxfp8, ...fp8];
  vi.mocked(quantizationApi.workspace).mockResolvedValue({ ...settings, candidates: catalog, candidate_groups: {
    NVQ: ['NVQ3J-L'], NINT: ['NINT4', 'NINT8'], 'MXFP4-SQ': ['MXFP4-SQ-1', 'MXFP4-SQ-F'], 'MXFP8-SQ': mxfp8, 'FP8-SQ': fp8,
  } });
  vi.mocked(quantizationApi.source).mockResolvedValue({ path: '/models/fp8', architecture: 'qwen4_exp', tensors: 12, parameters: 1000,
    format: 'hf', full_precision: true, source_precisions: ['BF16', family === 'FP8-SQ' ? 'FP8' : 'MXFP8'], eligible_candidates: [...settings.candidates, ...native] });
  vi.mocked(jobsApi.createJob).mockResolvedValue({ id: 'job1', kind: 'quantization.recipe', status: 'queued', progress: 0, payload: {}, cancel_requested: false, created_at: '', updated_at: '' });
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.click(screen.getByText('Candidates'));
  for (const name of native) expect(screen.getByRole('checkbox', { name })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/fp8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('11 / 11');
  for (const name of native) {
    expect(screen.getByRole('checkbox', { name })).toBeEnabled();
    expect(screen.getByRole('checkbox', { name })).toBeChecked();
  }
  expect(screen.getByRole('checkbox', { name: 'MXFP4-SQ-F' })).toBeDisabled();
  for (const name of incompatible) {
    expect(screen.getByRole('checkbox', { name })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name })).not.toBeChecked();
  }
  fireEvent.click(screen.getByRole('button', { name: 'Generate recipe' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('quantization.recipe', expect.objectContaining({ candidates: [...settings.candidates, ...native] })));
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/bf16' } });
  for (const name of native) expect(screen.getByRole('checkbox', { name })).toBeDisabled();
});

it('explicitly reports empty candidates and invalid targets without submitting a job', async () => {
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/full' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('qwen4_exp');
  fireEvent.click(screen.getByText('Candidates'));
  fireEvent.click(screen.getByRole('button', { name: /^Clear$/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Generate recipe' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Select at least one source-compatible candidate.');
  fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
  fireEvent.change(screen.getByLabelText('Target bpw'), { target: { value: '0' } });
  fireEvent.click(screen.getByRole('button', { name: 'Generate recipe' }));
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Target bpw must be finite and in (0, 32].'));
  expect(jobsApi.createJob).not.toHaveBeenCalled();
});

it.each([
  ['quantization_target_above_candidates', 'highest-precision candidate plan 8.5000 bpw'],
  ['quantization_target_below_candidates', 'minimum budget 4.5000 bpw'],
  ['quantization_target_above_source', 'source model average 16.0000 bpw'],
])('shows a numerical, actionable error for %s', async (code, text) => {
  vi.mocked(jobsApi.createJob).mockRejectedValue(new ApiError(422, { error: { code, message: 'validation failed', retryable: false,
    details: { target_bpw: 12, minimum_bpw: 4.5, maximum_bpw: 8.5, source_bpw: 16 } } }));
  render(<QuantizationPage />);
  await screen.findByText('3 / 3');
  fireEvent.change(screen.getByLabelText('Source model directory or MFQ file'), { target: { value: '/models/full' } });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect model' }));
  await screen.findByText('qwen4_exp');
  fireEvent.change(screen.getByLabelText('Target bpw'), { target: { value: '12' } });
  fireEvent.click(screen.getByRole('button', { name: 'Generate recipe' }));
  expect(await screen.findByRole('alert')).toHaveTextContent(text);
  expect(useJobStore.getState().jobs).toHaveLength(0);
});
