import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import { useState } from 'react';
import { SettingsProvider } from '../settings/SettingsProvider';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from '../settings/configuration';
import { SettingsRoute } from '../settings/SettingsRoute';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { modelsApi } from '../../shared/api/resources/models';
import { setApiBaseUrl } from '../../shared/api/client';
import { ConcurrencyFields } from './ConcurrencyFields';
import { DEFAULT_CONCURRENCY } from './concurrencySettings';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { ModelLoadContext } from './ModelLoadContext';
import { DEFAULT_LOAD_CONTEXT } from './loadContextDefaults';
import { useModelCatalog } from './useModelCatalog';
import type { ModelArtifact, ModelCacheProfile } from '../../shared/api/types';

const state = vi.hoisted(() => ({ connectionRevision: 0, refreshRuntime: vi.fn(), setSelectedModel: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ...state, ready: false, runtime: null,
  models: [], instances: [], studio: null }) }));

const policy = { defaults: { max_requests: 16, max_queued_requests: 32, queue_timeout_seconds: 0 },
  limits: { max_requests: 16, max_queued_requests: 32, queue_timeout_seconds: null } };

beforeEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
  setApiBaseUrl('');
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en' }));
  vi.spyOn(runtimeApi, 'concurrencyPolicy').mockResolvedValue(policy);
  state.refreshRuntime.mockResolvedValue(true);
});

function Form() {
  const [value, setValue] = useState(DEFAULT_CONCURRENCY);
  return <ConcurrencyFields value={value} onChange={setValue} policy={policy} />;
}

it('shows server ranges and validates integers, queue zero and unlimited timeout', () => {
  render(<SettingsProvider><Form /></SettingsProvider>);
  const rows = screen.getByRole('spinbutton', { name: 'Concurrent requests' });
  expect(rows).toHaveAttribute('max', '16');
  expect(rows).toHaveAttribute('placeholder', '16');
  fireEvent.change(rows, { target: { value: '17' } });
  expect(rows).toHaveAttribute('aria-invalid', 'true');
  expect(screen.getByRole('alert')).toHaveTextContent('server limits');
  fireEvent.change(rows, { target: { value: '1.5' } });
  expect(rows).toHaveAttribute('aria-invalid', 'true');
  fireEvent.change(rows, { target: { value: '16' } });
  fireEvent.change(screen.getByLabelText('Queue limit'), { target: { value: '0' } });
  fireEvent.change(screen.getByLabelText('Queue timeout (seconds)'), { target: { value: '0' } });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

function LoadForm() {
  const catalog = useModelCatalog();
  return <><ModelLoadPolicy catalog={catalog} />
    <button disabled={!catalog.concurrencyValid} onClick={() => void catalog.loadArtifact('model')}>
      Load model</button></>;
}

it('inherits saved defaults, blocks excess values and sends the chosen load request', async () => {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en',
    concurrencyDefaults: { max_requests: 8, max_queued_requests: 2, queue_timeout_seconds: 1.25 } }));
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ operation_id: 'load', status: 'accepted' }),
    { status: 202, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetchMock);
  render(<MemoryRouter><SettingsProvider><LoadForm /></SettingsProvider></MemoryRouter>);
  const rows = screen.getByLabelText('Concurrent requests');
  await waitFor(() => expect(rows).toBeEnabled());
  expect(rows).toHaveValue(8);
  fireEvent.change(rows, { target: { value: '17' } });
  expect(screen.getByText('Load model')).toBeDisabled();
  fireEvent.change(rows, { target: { value: '16' } });
  fireEvent.click(screen.getByText('Load model'));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  const [url, options] = fetchMock.mock.calls[0];
  expect(url).toContain('/api/v1/models/load');
  expect(JSON.parse(options.body)).toMatchObject({ model: 'model', max_requests: 16,
    max_queued_requests: 2, queue_timeout_seconds: 1.25, context_size: 32768 });
});

it('persists settings defaults and disables Apply for an invalid server range', async () => {
  render(<SettingsProvider><SettingsRoute /></SettingsProvider>);
  const rows = screen.getByLabelText('Concurrent requests');
  await waitFor(() => expect(rows).toBeEnabled());
  fireEvent.change(rows, { target: { value: '17' } });
  for (const button of screen.getAllByRole('button', { name: 'Apply settings' })) expect(button).toBeDisabled();
  fireEvent.change(rows, { target: { value: '12' } });
  fireEvent.change(screen.getByLabelText('Queue limit'), { target: { value: '4' } });
  fireEvent.click(screen.getAllByRole('button', { name: 'Apply settings' })[0]);
  expect(JSON.parse(localStorage.getItem(SETTINGS_KEY)!).concurrencyDefaults).toEqual({
    max_requests: 12, max_queued_requests: 4, queue_timeout_seconds: null,
  });
  fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
  expect(rows).not.toBeInTheDocument();
  expect(screen.getByLabelText('Concurrent requests')).toHaveValue(null);
});

it('updates KV and recurrent memory estimates when request rows change, keeping weights fixed', async () => {
  const GiB = 2 ** 30;
  const model = { id: 'model', name: 'model', complete: true, loadable: true, total_bytes: GiB,
    estimated_resident_weight_bytes: 8 * GiB, context_capacity: 32768 } as ModelArtifact;
  const profile: ModelCacheProfile = { max_context: 32768, fixed_bytes: GiB,
    components: [{ bytes_per_row: 32768, tokens_per_row: 1, allocation: 'exact', minimum_rows: 0 }] };
  vi.spyOn(modelsApi, 'modelCacheProfile').mockResolvedValue(profile);
  function Estimate() {
    const [value, setValue] = useState(DEFAULT_LOAD_CONTEXT);
    const [concurrency, setConcurrency] = useState(1);
    return <><button onClick={() => setConcurrency(4)}>Four rows</button>
      <ModelLoadContext concurrency={concurrency} value={value} onChange={setValue} models={[model]} /></>;
  }
  render(<SettingsProvider><Estimate /></SettingsProvider>);
  fireEvent.change(screen.getByLabelText('Model to estimate'), { target: { value: model.id } });
  await waitFor(() => expect(screen.getByLabelText('Estimated total memory')).toHaveTextContent('10 GiB'));
  fireEvent.click(screen.getByText('Four rows'));
  expect(screen.getByLabelText('Estimated total memory')).toHaveTextContent('16 GiB');
});
