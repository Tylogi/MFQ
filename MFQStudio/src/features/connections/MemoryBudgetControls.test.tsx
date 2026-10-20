import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryBudgetControls } from './MemoryBudgetControls';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { ApiError } from '../../shared/api/client';
import { useJobStore } from '../../stores/jobStore';

const api = vi.hoisted(() => ({ policy: vi.fn(), configure: vi.fn(), getJob: vi.fn(), addJob: vi.fn() }));
const state = vi.hoisted(() => ({ runtime: null as { runtime_memory_effective_budget_bytes: number } | null }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { memoryPolicy: api.policy, configureMemoryPolicy: api.configure } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { getJob: api.getJob } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: api.addJob, runtime: state.runtime, instances: [] }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

beforeEach(() => {
  vi.clearAllMocks();
  useJobStore.getState().setJobs([]);
  api.addJob.mockImplementation(job => useJobStore.getState().addJob(job));
  state.runtime = null;
  api.policy.mockResolvedValue({ total_limit_bytes: null, effective_total_limit_bytes: 107.52 * 2 ** 30 });
  api.configure.mockResolvedValue({ operation_id: 'job' });
  api.getJob.mockResolvedValue({ id: 'job' });
});

it('updates the automatic amount from live runtime telemetry without resetting manual edits', async () => {
  state.runtime = { runtime_memory_effective_budget_bytes: 90 * 2 ** 30 };
  const view = render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByText('90.0 GiB');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Apply budgets' })).not.toBeDisabled());
  expect(screen.queryByText('107.5 GiB')).not.toBeInTheDocument();
  state.runtime = { runtime_memory_effective_budget_bytes: 84 * 2 ** 30 };
  view.rerender(<MemoryBudgetControls residency="80 GiB" />);
  expect(screen.getByText('84.0 GiB')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Total resident memory budget mode'), { target: { value: 'manual' } });
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveDisplayValue('84.0');
  fireEvent.change(screen.getByLabelText('Total resident memory budget budget limit'), { target: { value: '96' } });
  state.runtime = { runtime_memory_effective_budget_bytes: 82 * 2 ** 30 };
  view.rerender(<MemoryBudgetControls residency="80 GiB" />);
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveValue(96);
  expect(screen.queryByText('82.0 GiB')).not.toBeInTheDocument();
});

it.each(['-1', '', '1e300'])('explains and restores an invalid budget %s on blur without submitting it', async invalid => {
  api.policy.mockResolvedValue({ total_limit_bytes: 96 * 2 ** 30 });
  render(<MemoryBudgetControls residency="80 GiB" />);
  const input = await screen.findByLabelText('Total resident memory budget budget limit');
  fireEvent.change(input, { target: { value: invalid } });
  fireEvent.blur(input);
  expect(input).toHaveValue(96);
  expect(screen.getByRole('alert')).toHaveTextContent('previous value was restored');
  expect(api.configure).not.toHaveBeenCalled();
});

it('restores the effective automatic policy when the server rejects a budget', async () => {
  api.configure.mockRejectedValue(new ApiError(413, { error: { code: 'resident_budget_exhausted', message: 'native failure', retryable: false, details: {} } }));
  render(<MemoryBudgetControls residency="80 GiB" />);
  const mode = screen.getByLabelText('Total resident memory budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  fireEvent.change(mode, { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('Total resident memory budget budget limit'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Increase it or unload another model');
  expect(mode).toHaveValue('automatic');
  expect(screen.getByText('107.5 GiB')).toBeInTheDocument();
});

it('restores the policy after an accepted asynchronous budget task fails', async () => {
  api.getJob.mockResolvedValue({ id: 'job', kind: 'runtime.memory.configure', status: 'queued', progress: 0, payload: {}, cancel_requested: false, created_at: '', updated_at: '' });
  render(<MemoryBudgetControls residency="80 GiB" />);
  const mode = screen.getByLabelText('Total resident memory budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  fireEvent.change(mode, { target: { value: 'manual' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  await waitFor(() => expect(api.addJob).toHaveBeenCalled());
  useJobStore.getState().updateJob('job', { status: 'failed', error: { code: 'resident_budget_exhausted', message: 'native failure', retryable: false, details: {} } });
  expect(await screen.findByRole('alert')).toHaveTextContent('Increase it or unload another model');
  expect(mode).toHaveValue('automatic');
});

it('shows detected automatic capacity and submits one shared total ceiling', async () => {
  render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByText('107.5 GiB');
  const totalMode = screen.getByLabelText('Total resident memory budget mode');
  expect(screen.getByText('107.5 GiB').compareDocumentPosition(totalMode) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  fireEvent.change(screen.getByLabelText('Total resident memory budget mode'), { target: { value: 'manual' } });
  expect(screen.getByLabelText('Total resident memory budget budget limit').closest('.memory-budget-input')?.nextElementSibling).toBe(totalMode);
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveDisplayValue('107.5');
  fireEvent.change(screen.getByLabelText('Total resident memory budget budget limit'), { target: { value: '96' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ total_limit_bytes: 96 * 2 ** 30, model_limit_bytes: null, prefix_limit_bytes: null }));
});

it('initializes manual mode from the currently displayed one-decimal automatic amount', async () => {
  state.runtime = { runtime_memory_effective_budget_bytes: 84.66 * 2 ** 30 };
  const view = render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByText('84.7 GiB');
  const mode = screen.getByLabelText('Total resident memory budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  fireEvent.change(mode, { target: { value: 'manual' } });
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveDisplayValue('84.7');
  state.runtime = { runtime_memory_effective_budget_bytes: 83.25 * 2 ** 30 };
  view.rerender(<MemoryBudgetControls residency="80 GiB" />);
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveDisplayValue('84.7');
  fireEvent.change(mode, { target: { value: 'automatic' } });
  expect(screen.getByText('83.3 GiB')).toBeInTheDocument();
  fireEvent.change(mode, { target: { value: 'manual' } });
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveDisplayValue('83.3');
});

it('clears the manual total without changing the weight and prefix sublimits', async () => {
  api.policy.mockResolvedValue({ total_limit_bytes: 96 * 2 ** 30, model_limit_bytes: 80 * 2 ** 30, prefix_limit_bytes: 4 * 2 ** 30 });
  render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByLabelText('Total resident memory budget budget limit');
  fireEvent.change(screen.getByLabelText('Total resident memory budget mode'), { target: { value: 'automatic' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ total_limit_bytes: null, model_limit_bytes: 80 * 2 ** 30, prefix_limit_bytes: 4 * 2 ** 30 }));
});

it('places current residency and the shared apply action below expert-cache warmup', async () => {
  render(<MemorySettingsPanel />);
  const apply = screen.getByRole('button', { name: 'Apply budgets' });
  await waitFor(() => expect(apply).toBeEnabled());
  const warmup = screen.getByText('Warm expert cache on launch').closest('.setting-row')!;
  const residency = screen.getByText('Current weight residency: --');
  expect(warmup.compareDocumentPosition(residency) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(residency.parentElement).toContainElement(apply);
  expect(screen.getAllByRole('button', { name: 'Apply budgets' })).toHaveLength(1);
  expect(api.configure).not.toHaveBeenCalled();
});
