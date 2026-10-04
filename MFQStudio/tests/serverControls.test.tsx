import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelAliasMapping } from '../src/features/connections/ModelAliasMapping';
import { PrefixCacheDirectory } from '../src/features/connections/PrefixCacheDirectory';
import { MemoryBudgetControls } from '../src/features/connections/MemoryBudgetControls';
import { RepositoryFiles } from '../src/features/models/RepositoryFiles';
import { runtimeApi } from '../src/shared/api/resources/runtime';
import { jobsApi } from '../src/shared/api/resources/jobs';
import { useJobStore } from '../src/stores/jobStore';

vi.mock('../src/features/settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: vi.fn() }) }));
vi.mock('../src/shared/api/resources/runtime', () => ({ runtimeApi: {
  modelAliases: vi.fn(), configureModelAliases: vi.fn(), memoryPolicy: vi.fn(), configureMemoryPolicy: vi.fn(),
} }));
vi.mock('../src/shared/api/resources/jobs', () => ({ jobsApi: { getJob: vi.fn() } }));

beforeEach(() => {
  vi.clearAllMocks();
  useJobStore.getState().setJobs([]);
  vi.mocked(runtimeApi.memoryPolicy).mockResolvedValue({ model_limit_bytes: null, prefix_limit_bytes: null,
    prefix_directory: null, actual_prefix_directory: '/data/mfq/prefix-cache' });
  vi.mocked(runtimeApi.configureMemoryPolicy).mockResolvedValue({ operation_id: 'plan' });
  vi.mocked(jobsApi.getJob).mockResolvedValue({ id: 'plan' } as Awaited<ReturnType<typeof jobsApi.getJob>>);
});

it('edits aliases without changing names and displays the selected alias', async () => {
  vi.mocked(runtimeApi.modelAliases).mockResolvedValue({ aliases: { first: 'fast' } });
  vi.mocked(runtimeApi.configureModelAliases).mockResolvedValue({ aliases: { first: 'my-model' } });
  render(<ModelAliasMapping models={['first', 'second']} selectedModel="first" />);
  await screen.findByText('fast');
  fireEvent.click(screen.getByRole('button', { name: 'Alias mapping' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'first alias' }), { target: { value: 'my-model' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  await screen.findByText('my-model');
  expect(runtimeApi.configureModelAliases).toHaveBeenCalledExactlyOnceWith({ first: 'my-model' });
});

it('always displays the actual managed directory and allows a manual location without resetting budgets', async () => {
  render(<PrefixCacheDirectory />);
  await screen.findByText('/data/mfq/prefix-cache');
  fireEvent.change(screen.getByRole('combobox', { name: 'Prefix cache directory mode' }), { target: { value: 'manual' } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Prefix cache directory' }), { target: { value: '/cache/prefix' } });
  expect(screen.getByText('/data/mfq/prefix-cache')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(runtimeApi.configureMemoryPolicy).toHaveBeenCalledExactlyOnceWith({ prefix_directory: '/cache/prefix' }));
});

it('defaults to automatic budgets and submits explicit aggregate limits', async () => {
  render(<MemoryBudgetControls residency="96 GiB" />);
  const model = screen.getByRole('combobox', { name: 'Total model residency mode' });
  await waitFor(() => expect(model).toBeEnabled());
  expect(model).toHaveValue('automatic');
  expect(screen.getByRole('combobox', { name: 'Prefix RAM allowance mode' })).toHaveValue('automatic');
  fireEvent.change(model, { target: { value: 'manual' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply and replan' }));
  await waitFor(() => expect(runtimeApi.configureMemoryPolicy).toHaveBeenCalledExactlyOnceWith({ model_limit_bytes: 64 * 2 ** 30, prefix_limit_bytes: null }));
});

it('downloads a shard folder as one request and allows multiple file selections', () => {
  const files = Array.from({ length: 6 }, (_, index) => ({ name: `S4-L/model-${index + 1}.mfq`, byte_size: 1024 }));
  const onDownload = vi.fn();
  render(<RepositoryFiles files={files} disabled={false} onDownload={onDownload} tr={(_zh, en) => en} />);
  fireEvent.click(screen.getByText(/Files and folders/));
  fireEvent.click(screen.getByRole('button', { name: /S4-L/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Download folder' }));
  expect(onDownload.mock.calls[0][0]).toEqual(files);
  fireEvent.click(screen.getByRole('checkbox', { name: /model-1/ }));
  fireEvent.click(screen.getByRole('checkbox', { name: /model-3/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Download 2 selected files' }));
  expect(onDownload.mock.calls[1][0]).toEqual([files[0], files[2]]);
});
