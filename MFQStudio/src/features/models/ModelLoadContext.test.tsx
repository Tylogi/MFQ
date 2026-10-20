import { useState } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelLoadContext } from './ModelLoadContext';
import { DEFAULT_LOAD_CONTEXT, readLoadContext, saveLoadContext } from './loadContextDefaults';
import { setApiBaseUrl } from '../../shared/api/client';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact, ModelCacheProfile, RuntimeStatus } from '../../shared/api/types';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string) => zh }) }));

beforeEach(() => { vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl(''); });
function Form({ models = [], runtime }: { models?: ModelArtifact[]; runtime?: RuntimeStatus } = {}) {
  const [value, setValue] = useState(readLoadContext);
  return <ModelLoadContext models={models} runtime={runtime} value={value} onChange={next => { setValue(next); saveLoadContext(next); }} />;
}

it('defaults to 32768 with collapsed automatic settings and remembers explicit context', () => {
  const view = render(<Form />);
  const context = screen.getByRole('spinbutton', { name: '默认上下文' });
  expect(context).toHaveAttribute('placeholder', '32768');
  expect(context).toHaveValue(null);
  expect(screen.getByRole('button', { name: '高级' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('checkbox', { name: '自动' })).toBeChecked();
  fireEvent.change(context, { target: { value: '65536' } });
  view.unmount(); render(<Form />);
  expect(screen.getByRole('spinbutton')).toHaveValue(65536);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
  expect(readLoadContext().context_size).toBeNull();
});

it('enforces the sparse RAM/SSD dependency and keeps non-sparse KV in VRAM', () => {
  render(<Form />);
  fireEvent.click(screen.getByRole('button', { name: '高级' }));
  expect(screen.getByRole('checkbox', { name: 'SSD KV' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: '默认 KV 量化档位' })).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  fireEvent.click(screen.getByRole('checkbox', { name: 'RAM 预分配' }));
  expect(screen.getByRole('checkbox', { name: 'SSD KV' })).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox', { name: 'SSD KV' }));
  expect(screen.getByRole('checkbox', { name: 'RAM 预分配' })).toBeChecked();
  fireEvent.click(screen.getByRole('button', { name: '非稀疏注意力模型' }));
  expect(screen.queryByRole('checkbox', { name: 'SSD KV' })).not.toBeInTheDocument();
  expect(screen.queryByText(/SSD KV/)).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox', { name: 'RAM 预分配' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox', { name: 'VRAM 预分配' }));
  expect(readLoadContext().vram_preallocated).toBe(false);
  fireEvent.change(screen.getByRole('combobox', { name: '默认 KV 量化档位' }), { target: { value: '2.5' } });
  expect(readLoadContext().kv_quantization).toEqual({ enabled: true, bits: 2.5, algorithm: 'turboquant' });
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  expect(screen.getByRole('combobox', { name: '默认 KV 量化档位' })).toHaveValue('off');
  expect(screen.getByRole('checkbox', { name: 'VRAM 预分配' })).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  expect(screen.getByRole('combobox', { name: '默认 KV 量化档位' })).toHaveValue('2.5');
});

const GiB = 2 ** 30;
const model: ModelArtifact = { id: 'small', name: 'Qwen-S2-L', architecture: 'qwen4_exp', format: 'mfq',
  shard_count: 5, total_bytes: 40 * GiB, estimated_resident_weight_bytes: 8 * GiB, ssd_ple_bytes: 32 * GiB,
  context_capacity: 131072, tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true,
  modified_at: '2026-10-15T00:00:00Z' };
const profile: ModelCacheProfile = { max_context: 131072, fixed_bytes: GiB,
  fixed_components: [{ group: 'GDN', name: 'recurrent_state', bytes: GiB }],
  components: [{ group: 'QSA', name: 'raw_kv', layers: 2, kv_heads: 16, head_dimension: 256,
    bytes_per_row: 32768, tokens_per_row: 1, allocation: 'exact', minimum_rows: 16 }] };

it('shows zero backend headroom even when the OS still reports three GiB available', () => {
  render(<Form runtime={{ memory_architecture: 'unified', host_memory_available_bytes: 3 * GiB,
    runtime_memory_headroom_bytes: 0 }} />);
  expect(screen.getByText(/当前剩余/)).toHaveTextContent('0 GiB');
});

it('defaults to no model and updates total with context and KV precision without loading a model', async () => {
  const cache = vi.spyOn(modelsApi, 'modelCacheProfile').mockResolvedValue(profile);
  const load = vi.spyOn(modelsApi, 'loadModel');
  render(<Form models={[model]} runtime={{ memory_architecture: 'discrete', host_memory_available_bytes: 12 * GiB,
    gpu_devices: [{ id: '0', name: 'GPU', memory_available_bytes: 3 * GiB }] }} />);
  const selector = screen.getByRole('combobox', { name: '估算模型' });
  expect(selector).toHaveValue('');
  expect(screen.getByLabelText('预计总占用')).toHaveTextContent('—');
  expect(screen.getByText(/当前剩余/)).toHaveTextContent('15 GiB');
  expect(cache).not.toHaveBeenCalled();
  fireEvent.change(selector, { target: { value: model.id } });
  await waitFor(() => expect(screen.getByLabelText('预计总占用')).toHaveTextContent('10 GiB'));
  expect(cache).toHaveBeenCalledWith(model.id, expect.any(AbortSignal), true);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '65536' } });
  expect(screen.getByLabelText('预计总占用')).toHaveTextContent('11 GiB');
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  fireEvent.change(screen.getByRole('combobox', { name: '默认 KV 量化档位' }), { target: { value: '4' } });
  expect(screen.getByLabelText('预计总占用')).toHaveTextContent('9.5 GiB');
  expect(cache).toHaveBeenCalledTimes(1);
  expect(load).not.toHaveBeenCalled();
  fireEvent.change(selector, { target: { value: '' } });
  expect(screen.getByLabelText('预计总占用')).toHaveTextContent('—');
});

it('marks insufficient space and clears a stale estimate when the model disappears', async () => {
  vi.spyOn(modelsApi, 'modelCacheProfile').mockResolvedValue(profile);
  const runtime: RuntimeStatus = { memory_architecture: 'unified', host_memory_available_bytes: 9 * GiB };
  const view = render(<Form models={[model]} runtime={runtime} />);
  fireEvent.change(screen.getByRole('combobox', { name: '估算模型' }), { target: { value: model.id } });
  await waitFor(() => expect(screen.getByLabelText('预计总占用')).toHaveClass('model-load-memory-exceeded'));
  view.rerender(<Form models={[]} runtime={runtime} />);
  expect(screen.getByRole('combobox', { name: '估算模型' })).toHaveValue('');
  expect(screen.getByLabelText('预计总占用')).toHaveTextContent('—');
});

it('does not report a partial total when cache metadata is unavailable', async () => {
  vi.spyOn(modelsApi, 'modelCacheProfile').mockResolvedValue(null);
  render(<Form models={[model]} />);
  fireEvent.change(screen.getByRole('combobox', { name: '估算模型' }), { target: { value: model.id } });
  await waitFor(() => expect(screen.getByLabelText('预计总占用')).toHaveTextContent('暂无法估算'));
});

it('isolates stored defaults by server and tolerates blocked storage', () => {
  setApiBaseUrl('http://127.0.0.1:8090');
  saveLoadContext({ ...DEFAULT_LOAD_CONTEXT, context_size: 8192 });
  setApiBaseUrl('http://127.0.0.1:8092');
  expect(readLoadContext()).toEqual(DEFAULT_LOAD_CONTEXT);
  const write = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  expect(() => saveLoadContext(DEFAULT_LOAD_CONTEXT)).not.toThrow();
  write.mockRestore();
  setApiBaseUrl('http://127.0.0.1:8090');
  expect(readLoadContext().context_size).toBe(8192);
});
