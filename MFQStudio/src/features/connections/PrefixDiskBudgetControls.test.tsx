import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { PrefixCacheControls } from './PrefixCacheControls';

const api = vi.hoisted(() => ({ policy: vi.fn(), configure: vi.fn(), getJob: vi.fn(), addJob: vi.fn(), discrete: false }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { memoryPolicy: api.policy, configureMemoryPolicy: api.configure } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { getJob: api.getJob } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: api.addJob,
  runtime: { memory_architecture: api.discrete ? 'discrete' : 'unified' } }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../stores/jobStore', () => ({ useJobStore: (selector: (state: { jobs: [] }) => unknown) => selector({ jobs: [] }) }));

beforeEach(() => {
  vi.clearAllMocks();
  api.discrete = false;
  api.policy.mockResolvedValue({ prefix_disk_limit_bytes: null });
  api.configure.mockResolvedValue({ operation_id: 'job' });
  api.getJob.mockResolvedValue({ id: 'job' });
});

it('sets RAM independently and disables SSD through its switch', async () => {
  api.discrete = true;
  render(<PrefixCacheControls budget={100 * 2 ** 30} />);
  const mode = screen.getByLabelText('RAM prefix cache budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  fireEvent.change(mode, { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('RAM prefix cache limit'), { target: { value: '3' } });
  fireEvent.click(screen.getByLabelText('Enable SSD tier'));
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ prefix_ram_limit_bytes: 3 * 2 ** 30,
    prefix_disk_limit_bytes: 0 }));
});

it('submits an SSD-only limit, including zero, without changing RAM or weights', async () => {
  render(<PrefixCacheControls budget={100 * 2 ** 30} />);
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
  render(<PrefixCacheControls budget={10 * 2 ** 30} />);
  await screen.findByLabelText('SSD budget limit');
  fireEvent.change(screen.getByLabelText('SSD budget mode'), { target: { value: 'automatic' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledWith({ prefix_disk_limit_bytes: null }));
});

it('restores the saved SSD allowance after an invalid value or failed save', async () => {
  api.policy.mockResolvedValue({ prefix_disk_limit_bytes: 10 * 2 ** 30 });
  api.configure.mockRejectedValue(new Error('save failed'));
  render(<PrefixCacheControls budget={10 * 2 ** 30} />);
  const input = await screen.findByLabelText('SSD budget limit');
  fireEvent.change(input, { target: { value: '-1' } });
  fireEvent.blur(input);
  expect(input).toHaveValue(10);
  expect(screen.getByRole('alert')).toHaveTextContent('previous value was restored');
  fireEvent.change(input, { target: { value: '20' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('save failed');
  expect(input).toHaveValue(10);
});

it('uses one shared apply button and submits both changed fields once', async () => {
  render(<PrefixCacheControls budget={100 * 2 ** 30} />);
  const mode = screen.getByLabelText('SSD budget mode');
  await waitFor(() => expect(mode).toBeEnabled());
  expect(screen.getAllByRole('button', { name: 'Apply' })).toHaveLength(1);
  fireEvent.change(screen.getByLabelText('Prefix cache directory mode'), { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('Prefix cache directory'), { target: { value: '/cache/prefix' } });
  fireEvent.change(mode, { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('SSD budget limit'), { target: { value: '64' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(api.configure).toHaveBeenCalledExactlyOnceWith({ prefix_directory: '/cache/prefix', prefix_disk_limit_bytes: 64 * 2 ** 30 }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled());
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(api.configure).toHaveBeenCalledOnce();
});

it('does not submit unchanged automatic modes or numerically identical manual quotas', async () => {
  const view = render(<PrefixCacheControls budget={64 * 2 ** 30} />);
  await waitFor(() => expect(screen.getByLabelText('SSD budget mode')).toBeEnabled());
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('SSD budget mode'), { target: { value: 'manual' } });
  fireEvent.change(screen.getByLabelText('SSD budget mode'), { target: { value: 'automatic' } });
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  view.unmount();
  api.policy.mockResolvedValue({ prefix_disk_limit_bytes: 64 * 2 ** 30 });
  render(<PrefixCacheControls budget={64 * 2 ** 30} />);
  const input = await screen.findByLabelText('SSD budget limit');
  fireEvent.change(input, { target: { value: '64.0' } });
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(api.configure).not.toHaveBeenCalled();
  expect(input.closest('.server-row-actions')!.firstElementChild).toContainElement(input);
});
