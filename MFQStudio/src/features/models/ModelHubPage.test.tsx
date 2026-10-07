import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import type { JobResource, OfficialModelList } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useJobStore } from '../../stores/jobStore';
import { ModelHubPage } from './ModelHubPage';

vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { officialHubModels: vi.fn() } }));
vi.mock('../../shared/api/resources/jobs', () => ({ jobsApi: { jobKinds: vi.fn(), createJob: vi.fn(), cancelJob: vi.fn(), retryJob: vi.fn(), deleteJob: vi.fn() } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob: (job: JobResource) => useJobStore.getState().addJob(job) }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

const configuration = { status: 'unknown' as const, recommendation: 'unknown' as const, reasons: [] };
const source = { provider: 'modelscope' as const, repo_id: 'example/model', url: 'https://modelscope.cn/models/example/model', available: true };
const catalog: OfficialModelList = {
  system: { platform: 'macOS', machine: 'arm64', backend: 'metal' },
  data: [{ id: 'model', name: 'Test model', family: 'Test', architecture: 'test', description: '', description_zh: '',
    modalities: ['text'], capabilities: [], precision_options: [], supports_ssd_streaming: false,
    sources: [source], selected_source: source, revision: 'master', downloads: 0, likes: 0,
    configuration, variants: [{ id: 'S4', label: 'S4', format: 'mfq', files: ['S4.mfq'], byte_size: 10, configuration }],
  }],
};
function download(id = 'download-1'): JobResource {
  return { id, kind: 'download.modelscope', status: 'queued', progress: 0, cancel_requested: false,
    payload: { repo_id: source.repo_id, destination: 'models/model/S4' }, created_at: '2026-01-01', updated_at: '2026-01-01' };
}
beforeEach(() => {
  useJobStore.getState().setJobs([]);
  vi.mocked(modelsApi.officialHubModels).mockResolvedValue(catalog);
  vi.mocked(jobsApi.jobKinds).mockResolvedValue([{ kind: 'download.modelscope', payload_schema: {} }]);
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => { vi.clearAllMocks(); useJobStore.getState().setJobs([]); });

it('keeps a submitted download on this page, animates into the circle and opens the third tab', async () => {
  vi.mocked(jobsApi.createJob).mockResolvedValue(download());
  const { container } = render(<ModelHubPage />);
  expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual(['Official', 'Community', 'Download queue']);
  const button = await screen.findByRole('button', { name: 'Download' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  await waitFor(() => expect(container.querySelector('.download-flight')).toBeInTheDocument());
  expect(screen.getByRole('tab', { name: 'Official' })).toHaveAttribute('aria-selected', 'true');
  expect(container.querySelector('.download-circle-count')).toHaveTextContent('1');
  fireEvent.click(screen.getByRole('button', { name: 'Download queue' }));
  expect(screen.getByRole('tab', { name: 'Download queue' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByText(source.repo_id)).toBeInTheDocument();
  expect(screen.getByText('Queued')).toBeInTheDocument();
});

it('filters model-load and quantization jobs, retains downloads across remount and updates live progress', async () => {
  useJobStore.getState().setJobs([download(), { ...download('load'), kind: 'model.load', payload: { repo_id: 'not-a-download' } }]);
  const first = render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  expect(screen.queryByText('not-a-download')).not.toBeInTheDocument();
  act(() => useJobStore.getState().updateJob('download-1', { status: 'running', progress: 0.25 }));
  expect(screen.getByText('25%')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'Download progress' })).toHaveAttribute('value', '0.25');
  first.unmount();
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('button', { name: 'Download queue' }));
  expect(screen.getByText('25%')).toBeInTheDocument();
});

it('cancels and retries only the chosen download through the existing jobs API', async () => {
  useJobStore.getState().setJobs([download()]);
  vi.mocked(jobsApi.cancelJob).mockResolvedValue({ ...download(), status: 'cancelled' });
  vi.mocked(jobsApi.retryJob).mockResolvedValue(download('retry'));
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await screen.findByText('Cancelled');
  expect(jobsApi.cancelJob).toHaveBeenCalledWith('download-1');
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(useJobStore.getState().jobs).toHaveLength(2));
  expect(jobsApi.retryJob).toHaveBeenCalledWith('download-1');
  expect(screen.getByRole('tab', { name: 'Download queue' })).toHaveAttribute('aria-selected', 'true');
});

it('shows transfer speed and deletes a completed record without touching files', async () => {
  useJobStore.getState().setJobs([{ ...download(), status: 'running', progress_data: { downloaded_bytes: 1024, bytes_per_second: 2048, files_completed: 1 } }]);
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  expect(screen.getByText(/2 KiB\/s/)).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Delete download record' })).not.toBeInTheDocument();
  act(() => useJobStore.getState().updateJob('download-1', { status: 'succeeded' }));
  vi.mocked(jobsApi.deleteJob).mockResolvedValue(undefined);
  fireEvent.click(screen.getByRole('button', { name: 'Delete download record' }));
  await waitFor(() => expect(useJobStore.getState().jobs).toHaveLength(0));
  expect(jobsApi.deleteJob).toHaveBeenCalledExactlyOnceWith('download-1');
});

it('does not put an old download into the active connection after leaving its page', async () => {
  let finish!: (job: JobResource) => void;
  vi.mocked(jobsApi.createJob).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const view = render(<ModelHubPage />);
  const button = await screen.findByRole('button', { name: 'Download' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  view.unmount();
  await act(async () => { finish(download()); });
  expect(useJobStore.getState().jobs).toEqual([]);
});
