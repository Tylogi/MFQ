import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryBudgetControls } from './MemoryBudgetControls';

const api = vi.hoisted(() => ({ policy: vi.fn(), configure: vi.fn(), getJob: vi.fn(), addJob: vi.fn() }));
const state = vi.hoisted(() => ({ runtime: null as { runtime_memory_effective_budget_bytes: number } | null }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { memoryPolicy: api.policy, configureMemoryPolicy: api.configure } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { getJob: api.getJob } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: api.addJob, runtime: state.runtime }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../stores/jobStore', () => ({ useJobStore: (selector: (state: { jobs: [] }) => unknown) => selector({ jobs: [] }) }));

beforeEach(() => {
  vi.clearAllMocks();
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
  fireEvent.change(screen.getByLabelText('Total resident memory budget budget limit'), { target: { value: '96' } });
  state.runtime = { runtime_memory_effective_budget_bytes: 82 * 2 ** 30 };
  view.rerender(<MemoryBudgetControls residency="80 GiB" />);
  expect(screen.getByLabelText('Total resident memory budget budget limit')).toHaveValue(96);
  expect(screen.queryByText('82.0 GiB')).not.toBeInTheDocument();
});

it('shows detected automatic capacity and submits one shared total ceiling', async () => {
  render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByText('107.5 GiB');
  fireEvent.change(screen.getByLabelText('Total resident memory budget mode'), { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('Total resident memory budget budget limit'), { target: { value: '96' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ total_limit_bytes: 96 * 2 ** 30, model_limit_bytes: null, prefix_limit_bytes: null }));
});

it('clears the manual total without changing the weight and prefix sublimits', async () => {
  api.policy.mockResolvedValue({ total_limit_bytes: 96 * 2 ** 30, model_limit_bytes: 80 * 2 ** 30, prefix_limit_bytes: 4 * 2 ** 30 });
  render(<MemoryBudgetControls residency="80 GiB" />);
  await screen.findByLabelText('Total resident memory budget budget limit');
  fireEvent.change(screen.getByLabelText('Total resident memory budget mode'), { target: { value: 'automatic' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply budgets' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ total_limit_bytes: null, model_limit_bytes: 80 * 2 ** 30, prefix_limit_bytes: 4 * 2 ** 30 }));
});
