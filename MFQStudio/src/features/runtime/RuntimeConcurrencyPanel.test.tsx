import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SettingsProvider } from '../settings/SettingsProvider';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from '../settings/configuration';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { JobResource, RuntimeInstance } from '../../shared/api/types';
import { RuntimeConcurrencyPanel } from './RuntimeConcurrencyPanel';

const state = vi.hoisted(() => ({ instances: [] as RuntimeInstance[], connectionRevision: 0,
  refreshRuntime: vi.fn(), addJob: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));

const concurrency = { requested: { max_requests: 16, max_queued_requests: 32, queue_timeout_seconds: 0 },
  effective: { max_requests: 8, max_queued_requests: 32, queue_timeout_seconds: 0 },
  limits: { max_requests: 32, max_queued_requests: 32, queue_timeout_seconds: null } };

beforeEach(() => {
  vi.restoreAllMocks();
  state.refreshRuntime.mockReset().mockResolvedValue(true);
  state.addJob.mockReset();
  state.instances = [{ id: 'instance', model: 'test-model', state: 'ready', devices: ['cuda'],
    active_sessions: 0, queued_requests: 2, request_capacity: 8, concurrency,
    queue_wait_ms_avg: 12, queue_wait_ms_max: 30, rejected_requests: 3 }];
  localStorage.clear();
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en' }));
});

it('shows negotiated capacity, queue telemetry and applies queue edits without reloading', async () => {
  const update = vi.spyOn(runtimeApi, 'updateConcurrency').mockResolvedValue({ concurrency, reload_required: false });
  const reload = vi.spyOn(runtimeApi, 'reloadConcurrency');
  render(<SettingsProvider><RuntimeConcurrencyPanel /></SettingsProvider>);
  expect(screen.getByText('Requested 16, model effective 8')).toBeInTheDocument();
  expect(screen.getByText(/Queued 2/)).toHaveTextContent('Average wait 12 ms · Max wait 30 ms · Rejected 3');
  fireEvent.change(screen.getByLabelText('Queue limit'), { target: { value: '10' } });
  fireEvent.change(screen.getByLabelText('Queue timeout (seconds)'), { target: { value: '2.5' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save queue settings' }));
  await waitFor(() => expect(update).toHaveBeenCalledWith('instance', {
    max_requests: 16, max_queued_requests: 10, queue_timeout_seconds: 2.5,
  }));
  expect(reload).not.toHaveBeenCalled();
});

it('makes row changes explicit and submits a reload only on the reload button', async () => {
  state.instances[0].queued_requests = 0;
  const update = vi.spyOn(runtimeApi, 'updateConcurrency').mockResolvedValue({ concurrency, reload_required: true });
  const job = { id: 'reload', kind: 'runtime.concurrency.configure', status: 'queued' } as JobResource;
  const reload = vi.spyOn(runtimeApi, 'reloadConcurrency').mockResolvedValue(job);
  render(<SettingsProvider><RuntimeConcurrencyPanel /></SettingsProvider>);
  fireEvent.change(screen.getByLabelText('Concurrent requests'), { target: { value: '24' } });
  expect(screen.getByRole('status')).toHaveTextContent('require a reload');
  expect(reload).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Save queue settings' }));
  await waitFor(() => expect(update).toHaveBeenCalled());
  await waitFor(() => expect(screen.getByRole('button', { name: 'Reload with new settings' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Reload with new settings' }));
  await waitFor(() => expect(reload).toHaveBeenCalledWith('instance', {
    max_requests: 24, max_queued_requests: 32, queue_timeout_seconds: 0,
  }));
  expect(state.addJob).toHaveBeenCalledWith(job);
});

it('rejects invalid edits and shows update errors', async () => {
  vi.spyOn(runtimeApi, 'updateConcurrency').mockRejectedValue(new Error('update failed'));
  render(<SettingsProvider><RuntimeConcurrencyPanel /></SettingsProvider>);
  fireEvent.change(screen.getByLabelText('Queue limit'), { target: { value: '33' } });
  expect(screen.getByRole('button', { name: 'Save queue settings' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Queue limit'), { target: { value: '3' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save queue settings' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('update failed');
});
