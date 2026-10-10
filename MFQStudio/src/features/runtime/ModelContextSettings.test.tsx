import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelContextSettings } from './ModelContextSettings';
import { useJobStore } from '../../stores/jobStore';
import type { ModelCacheProfile, RuntimeInstance } from '../../shared/api/types';

const mocks = vi.hoisted(() => ({ policy: vi.fn(), info: vi.fn(), reload: vi.fn(), warning: vi.fn(), artifacts: vi.fn(), profile: vi.fn() }));
const state = vi.hoisted(() => ({ language: 'en', ready: true, connectionRevision: 0, reloadingInstances: {},
  instances: [{ id: 'a', model: 'Model A', state: 'ready', context_size: 262144, context_capacity: 262144 }] as RuntimeInstance[] }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ...state, reloadModelContext: mocks.reload }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string, en: string) => state.language === 'en' ? en : zh }) }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { contextPolicy: mocks.policy, yarnContextInfo: mocks.info } }));
vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { modelArtifacts: mocks.artifacts, modelCacheProfile: mocks.profile } }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: vi.fn(), error: vi.fn(), warning: mocks.warning } }));

beforeEach(() => {
  vi.clearAllMocks();
  state.language = 'en';
  state.instances = [{ id: 'a', model: 'Model A', state: 'ready', context_size: 262144, context_capacity: 262144 } as RuntimeInstance];
  useJobStore.getState().setJobs([]);
  mocks.policy.mockResolvedValue({ max_context_size: null, model_overrides: {}, model_yarn_enabled: {} });
  mocks.info.mockResolvedValue({ supported: true, native_context: 262144, maximum_context: 1048576,
    maximum_factor: 4, enabled: false, effective_factor: 1 });
  mocks.artifacts.mockResolvedValue([{ id: 'artifact-a', name: 'Model A' }]);
  mocks.profile.mockResolvedValue({ max_context: 262144, fixed_bytes: 29978, fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
    { group: 'QSA', name: 'raw_kv', bytes_per_row: 26624, tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16, max_read_rows_per_token: 2048 },
    { name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656, tokens_per_row: 4, allocation: 'power_of_two', minimum_rows: 16,
      row_rounding: 'floor', active_after: 0 },
  ] } satisfies ModelCacheProfile);
  mocks.reload.mockImplementation(async (id, size, enabled, storage) => {
    const job = { id: 'ctx', kind: 'runtime.context.configure', status: 'running', progress: 0,
      payload: { instance_id: id, context_size: size, yarn_enabled: enabled, qsa_kv_offload: storage }, cancel_requested: false,
      created_at: '2026-10-09T00:00:00Z', updated_at: '2026-10-09T00:00:00Z' } as const;
    useJobStore.getState().addJob(job);
    return job;
  });
});

it('defaults KV quantization to collapsed and off with TurboQuant 4 bit, then saves the whole state once', async () => {
  state.instances[0].kv_quantization_supported = true;
  mocks.profile.mockResolvedValue({ max_context: 262144, fixed_bytes: 29978,
    fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
      { group: 'QSA', name: 'raw_kv', layers: 13, head_dimension: 256, kv_heads: 2, bytes_per_row: 26624,
        tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16 },
      { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656, tokens_per_row: 4,
        allocation: 'power_of_two', minimum_rows: 16, row_rounding: 'floor' },
    ] } satisfies ModelCacheProfile);
  render(<ModelContextSettings />);
  const caret = await screen.findByRole('button', { name: 'KV quantization' });
  const toggle = screen.getByRole('checkbox', { name: 'Enable KV quantization' });
  await waitFor(() => expect(toggle).toBeEnabled());
  expect(caret).toHaveAttribute('aria-expanded', 'false');
  expect(toggle).not.toBeChecked();
  fireEvent.click(caret);
  const level = screen.getByRole('combobox', { name: 'KV quantization level' });
  expect(level).toHaveValue('4');
  expect(level.querySelectorAll('option')).toHaveLength(7);
  fireEvent.click(toggle);
  fireEvent.change(level, { target: { value: '2.5' } });
  await waitFor(() => expect(screen.getByLabelText('Model A KV estimate')).toHaveTextContent('Raw KV 1.07 GiB · Indexer 0.41 GiB'));
  expect(mocks.reload).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Save to this model' }));
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledExactlyOnceWith('a', null, false, undefined,
    { enabled: true, bits: 2.5, algorithm: 'turboquant' }));
});

it('does not enable KV quantization on an unsupported backend', async () => {
  render(<ModelContextSettings />);
  expect(await screen.findByRole('checkbox', { name: 'Enable KV quantization' })).toBeDisabled();
});

it('keeps independent KV quantization drafts while switching models', async () => {
  state.instances[0].kv_quantization_supported = true;
  state.instances.push({ ...state.instances[0], id: 'b', model: 'Model B' });
  render(<ModelContextSettings />);
  const toggle = await screen.findByRole('checkbox', { name: 'Enable KV quantization' });
  await waitFor(() => expect(toggle).toBeEnabled());
  fireEvent.click(toggle);
  fireEvent.change(screen.getByRole('combobox', { name: 'KV quantization level' }), { target: { value: '3.5' } });
  const selector = screen.getByRole('combobox', { name: 'Per-model context model' });
  fireEvent.change(selector, { target: { value: 'Model B' } });
  expect(screen.getByRole('checkbox', { name: 'Enable KV quantization' })).not.toBeChecked();
  fireEvent.change(selector, { target: { value: 'Model A' } });
  expect(screen.getByRole('checkbox', { name: 'Enable KV quantization' })).toBeChecked();
  fireEvent.click(screen.getByRole('button', { name: 'KV quantization' }));
  expect(screen.getByRole('combobox', { name: 'KV quantization level' })).toHaveValue('3.5');
});

async function controls() {
  render(<ModelContextSettings />);
  const save = screen.getByRole('button', { name: 'Save to this model' });
  await waitFor(() => expect(save).toBeEnabled());
  return { save, input: screen.getByRole('spinbutton', { name: 'Model A maximum context' }),
    toggle: screen.getByRole('checkbox', { name: 'Model A YaRN context limit extension' }) };
}

it('clamps native overflow on save instead of disabling the save button', async () => {
  const { save, input, toggle } = await controls();
  expect(toggle).not.toBeChecked();
  fireEvent.change(input, { target: { value: '524288' } });
  expect(save).toBeEnabled();
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledWith('a', 262144, false, undefined, undefined));
  expect(input).toHaveValue(262144);
  expect(screen.getByRole('alert')).toHaveTextContent('Cannot set a value above the native context limit');
});

it('saves extended values with YaRN and displays the multiplier and cap', async () => {
  const { save, input, toggle } = await controls();
  expect(screen.getByText(/Up to 4× · Extended limit 1,048,576/)).toBeInTheDocument();
  fireEvent.click(toggle);
  fireEvent.change(input, { target: { value: '524288' } });
  expect(screen.getByLabelText('Model A final state')).toHaveTextContent('2.00×');
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledWith('a', 524288, true, undefined, undefined));
  expect(toggle).toBeChecked();
});

it('clamps overflow to the YaRN maximum and exposes the specified error', async () => {
  const { save, input, toggle } = await controls();
  fireEvent.click(toggle);
  fireEvent.change(input, { target: { value: '4294967296' } });
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledWith('a', 1048576, true, undefined, undefined));
  expect(input).toHaveValue(1048576);
  expect(mocks.warning).toHaveBeenCalledWith('Exceeds the YaRN extended context limit');
});

it('restores saved YaRN settings and disables unsupported models', async () => {
  mocks.policy.mockResolvedValue({ max_context_size: null, model_overrides: { 'Model A': 524288 }, model_yarn_enabled: { 'Model A': true } });
  const { input, toggle } = await controls();
  expect(toggle).toBeChecked();
  expect(input).toHaveValue(524288);
});

it('disables YaRN when the model or backend does not support it', async () => {
  mocks.info.mockResolvedValue({ supported: false, native_context: 262144, maximum_context: 262144, enabled: false });
  const { toggle } = await controls();
  expect(toggle).toBeDisabled();
  expect(screen.getByText(/YaRN extension is not currently supported/)).toBeInTheDocument();
});

it('keeps automatic context and rejects invalid integers', async () => {
  const { input, save } = await controls();
  fireEvent.change(input, { target: { value: '0' } });
  expect(save).toBeDisabled();
  fireEvent.change(input, { target: { value: '512.5' } });
  expect(save).toBeDisabled();
  fireEvent.change(input, { target: { value: '' } });
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledWith('a', null, false, undefined, undefined));
});

it('uses the requested Chinese wording for both boundary warnings', async () => {
  state.language = 'zh-CN';
  render(<ModelContextSettings />);
  const save = screen.getByRole('button', { name: '保存到该模型' });
  await waitFor(() => expect(save).toBeEnabled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Model A 最大上下文' }), { target: { value: '524288' } });
  fireEvent.click(save);
  await waitFor(() => expect(mocks.warning).toHaveBeenCalledWith('不能设置超出最大值的值'));
});

it('updates the full KV estimate from unsaved input, including beyond native capacity', async () => {
  const { input } = await controls();
  const output = screen.getByLabelText('Model A KV estimate');
  await waitFor(() => expect(output).toHaveTextContent('KV 6.91 GiB'));
  expect(output).toHaveTextContent('(Raw KV 6.50 GiB · Indexer 0.41 GiB)');
  fireEvent.change(input, { target: { value: '524288' } });
  expect(output).toHaveTextContent('KV 13.81 GiB');
  expect(output).toHaveTextContent('(Raw KV 13.00 GiB · Indexer 0.81 GiB)');
  expect(mocks.reload).not.toHaveBeenCalled();
  expect(mocks.profile).toHaveBeenCalledTimes(1);
  fireEvent.change(input, { target: { value: '678920' } });
  expect(output).toHaveTextContent('KV 17.89 GiB');
  fireEvent.change(input, { target: { value: '0' } });
  expect(output).toHaveTextContent('Enter a valid context');
});

it('uses the saved global cap for automatic context and displays Chinese breakdown labels', async () => {
  state.language = 'zh-CN';
  mocks.policy.mockResolvedValue({ max_context_size: 131072, model_overrides: {} });
  render(<ModelContextSettings />);
  const output = screen.getByLabelText('Model A KV 估算');
  await waitFor(() => expect(output).toHaveTextContent('KV 3.45 GiB'));
  expect(output).toHaveTextContent('（原始KV 3.25 GiB · Indexer 0.20 GiB）');
  fireEvent.change(screen.getByRole('spinbutton', { name: '全局上下文上限' }), { target: { value: '65536' } });
  expect(output).toHaveTextContent('KV 3.45 GiB');
});

it('reports missing cache metadata without blocking context settings', async () => {
  mocks.profile.mockRejectedValue(new Error('metadata read failed'));
  const { save } = await controls();
  const output = screen.getByLabelText('Model A KV estimate');
  await waitFor(() => expect(output).toHaveTextContent('Unavailable'));
  expect(output).toHaveAttribute('title', 'metadata read failed');
  expect(save).toBeEnabled();
});

it('keeps estimates and unsaved context drafts independent for multiple models', async () => {
  state.instances.push({ id: 'b', model: 'Model B', state: 'ready', context_size: 262144, context_capacity: 262144 } as RuntimeInstance);
  mocks.artifacts.mockResolvedValue([{ id: 'artifact-a', name: 'Model A' }, { id: 'artifact-b', name: 'Model B' }]);
  const { input } = await controls();
  const first = screen.getByLabelText('Model A KV estimate');
  await waitFor(() => expect(first).toHaveTextContent('KV 6.91 GiB'));
  expect(screen.queryByLabelText('Model B KV estimate')).not.toBeInTheDocument();
  fireEvent.change(input, { target: { value: '524288' } });
  expect(first).toHaveTextContent('KV 13.81 GiB');
  fireEvent.change(screen.getByRole('combobox', { name: 'Per-model context model' }), { target: { value: 'Model B' } });
  const second = screen.getByLabelText('Model B KV estimate');
  expect(second).toHaveTextContent('KV 6.91 GiB');
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Model B maximum context' }), { target: { value: '131072' } });
  expect(second).toHaveTextContent('KV 3.45 GiB');
  fireEvent.change(screen.getByRole('combobox', { name: 'Per-model context model' }), { target: { value: 'Model A' } });
  expect(screen.getByRole('spinbutton', { name: 'Model A maximum context' })).toHaveValue(524288);
  expect(screen.getByLabelText('Model A KV estimate')).toHaveTextContent('KV 13.81 GiB');
  expect(mocks.reload).not.toHaveBeenCalled();
});

it('saves context, YaRN and streaming in one request and shows SSD estimates only when enabled', async () => {
  state.instances[0].qsa_kv_offload_supported = true;
  const { input, toggle, save } = await controls();
  const summary = screen.getByLabelText('Model A final state');
  expect(summary).not.toHaveTextContent('SSD');
  expect(summary).not.toHaveTextContent('Maximum reads/token');
  fireEvent.click(toggle);
  fireEvent.change(input, { target: { value: '524288' } });
  fireEvent.click(screen.getByRole('checkbox', { name: 'Enable streaming sparse attention' }));
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Resident KV budget' }), { target: { value: '0.5' } });
  expect(summary).toHaveTextContent('524,288 tokens');
  expect(summary).toHaveTextContent('2.00×');
  expect(summary).toHaveTextContent('Resident 512 MiB · SSD 320.03 MiB');
  expect(summary).toHaveTextContent('Maximum reads/token');
  expect(summary).toHaveTextContent('52 MiB');
  expect(mocks.reload).not.toHaveBeenCalled();
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledOnce());
  expect(mocks.reload).toHaveBeenCalledWith('a', 524288, true, { enabled: true, budget_bytes: 2 ** 29 }, undefined);
  expect(screen.queryByRole('button', { name: 'Apply settings' })).not.toBeInTheDocument();
});

it('retains model settings and progress while its instance is being replaced', async () => {
  const view = render(<ModelContextSettings />);
  const save = screen.getByRole('button', { name: 'Save to this model' });
  await waitFor(() => expect(save).toBeEnabled());
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Model A maximum context' }), { target: { value: '131072' } });
  fireEvent.click(save);
  await waitFor(() => expect(mocks.reload).toHaveBeenCalledOnce());
  state.instances = [];
  view.rerender(<ModelContextSettings />);
  expect(screen.getByRole('spinbutton', { name: 'Model A maximum context' })).toHaveValue(131072);
  expect(screen.getByRole('progressbar', { name: 'Model A context update progress' })).toBeInTheDocument();
  expect(save).toBeDisabled();
});

it('keeps drafts and permits retry when submitting fails, without reporting saved state', async () => {
  mocks.reload.mockRejectedValueOnce(new Error('runtime busy'));
  const { input, save } = await controls();
  fireEvent.change(input, { target: { value: '131072' } });
  fireEvent.click(save);
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('runtime busy'));
  expect(input).toHaveValue(131072);
  expect(save).toBeEnabled();
  expect(screen.queryByText('Saved to this model')).not.toBeInTheDocument();
});
