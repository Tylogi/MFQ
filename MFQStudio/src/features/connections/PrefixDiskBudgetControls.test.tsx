import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { PrefixDiskBudgetControls } from './PrefixDiskBudgetControls';

const api = vi.hoisted(() => ({ policy: vi.fn(), configure: vi.fn(), getJob: vi.fn(), addJob: vi.fn() }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { memoryPolicy: api.policy, configureMemoryPolicy: api.configure } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { getJob: api.getJob } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: api.addJob }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../stores/jobStore', () => ({ useJobStore: (selector: (state: { jobs: [] }) => unknown) => selector({ jobs: [] }) }));

beforeEach(() => {
  vi.clearAllMocks();
  api.policy.mockResolvedValue({ prefix_disk_limit_bytes: null });
  api.configure.mockResolvedValue({ operation_id: 'job' });
  api.getJob.mockResolvedValue({ id: 'job' });
});

it('submits an SSD-only limit, including zero, without changing RAM or weights', async () => {
  render(<PrefixDiskBudgetControls budget={100 * 2 ** 30} />);
  const mode = screen.getByLabelText('SSD budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  fireEvent.change(mode, { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('SSD budget limit'), { target: { value: '0' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ prefix_disk_limit_bytes: 0 }));
  await waitFor(() => expect(api.addJob).toHaveBeenCalledWith({ id: 'job' }));
});

it('automatic mode clears only the SSD override', async () => {
  api.policy.mockResolvedValue({ prefix_disk_limit_bytes: 10 * 2 ** 30 });
  render(<PrefixDiskBudgetControls budget={10 * 2 ** 30} />);
  await screen.findByLabelText('SSD budget limit');
  fireEvent.change(screen.getByLabelText('SSD budget mode'), { target: { value: 'automatic' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ prefix_disk_limit_bytes: null }));
});
