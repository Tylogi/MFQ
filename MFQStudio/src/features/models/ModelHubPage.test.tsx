import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import type { JobResource, OfficialModelList } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { ApiError } from '../../shared/api/client';
import { useJobStore } from '../../stores/jobStore';
import { ModelHubPage } from './ModelHubPage';
import { DownloadQueue } from './DownloadQueue';

vi.mock('../../shared/api/resources/models', () => ({ modelsApi: { officialHubModels: vi.fn(), downloadDirectory: vi.fn(), modelDirectories: vi.fn(), configureDownloadDirectory: vi.fn(), configureDownloadReclaim: vi.fn(), probeDownloadSources: vi.fn().mockResolvedValue({ data: [] }) } }));
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
  vi.mocked(modelsApi.downloadDirectory).mockResolvedValue({ directory: null, actual_directory: '/workspace/models', default_directory: '/workspace/models', disk_total_bytes: 2048 * 2 ** 30, disk_free_bytes: 200 * 2 ** 30 });
  vi.mocked(jobsApi.jobKinds).mockResolvedValue([{ kind: 'download.modelscope', payload_schema: {} }]);
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); useJobStore.getState().setJobs([]); });

it.each(['Q3/model-EW-Q3-00001-of-00043.mfq', 'Q3/model-EW-Q3*.mfq'])(
  'keeps the selected precision visible when a shared Engram file appears first: %s', (primary) => {
    const job = download();
    job.payload.include = ['Engram/model-EW-Engram-MXFP8-00002-of-00043.mfq', primary];
    const { container } = render(<DownloadQueue jobs={[job]} onJobCreated={vi.fn()} />);
    expect(container.querySelector('.download-item-heading strong')).toHaveTextContent(primary);
  });

it('keeps a submitted download on this page, animates into the circle and opens the third tab', async () => {
  vi.mocked(jobsApi.createJob).mockResolvedValue(download());
  const { container } = render(<ModelHubPage />);
  const header = container.querySelector('.model-browser-header')!;
  expect(header.children[0]).toBe(screen.getByRole('heading', { name: 'Model browser' }));
  expect(header.children[1]).toHaveClass('download-concurrency');
  expect(header.children[2]).toHaveClass('model-browser-description');
  expect(header.children[2].querySelector('.download-directory-control')).toBeInTheDocument();
  expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual(['Official', 'Community', 'Download queue']);
  const button = await screen.findByRole('button', { name: 'Download' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  await waitFor(() => expect(container.querySelector('.download-flight')).toBeInTheDocument());
  expect(jobsApi.createJob).toHaveBeenCalledWith('download.modelscope', expect.objectContaining({ download_directory: '/workspace/models' }));
  expect(screen.getByRole('tab', { name: 'Official' })).toHaveAttribute('aria-selected', 'true');
  expect(container.querySelector('.download-circle-count')).toHaveTextContent('1');
  fireEvent.click(screen.getByRole('button', { name: 'Download queue' }));
  expect(screen.getByRole('tab', { name: 'Download queue' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByText(source.repo_id)).toBeInTheDocument();
  expect(container.querySelector('.download-item-heading strong')).toHaveTextContent('S4');
  expect(screen.getByText('Queued')).toBeInTheDocument();
});

it('reclaims prefix space only after the setting is saved and restores it on failure', async () => {
  vi.mocked(modelsApi.configureDownloadReclaim).mockResolvedValue({ directory: null, actual_directory: '/workspace/models', default_directory: '/workspace/models', disk_total_bytes: 2048 * 2 ** 30, disk_free_bytes: 200 * 2 ** 30, disk_available_bytes: 260 * 2 ** 30, evict_prefix_cache: true });
  render(<ModelHubPage />);
  const checkbox = await screen.findByRole('checkbox', { name: 'Reclaim SSD prefix budget' });
  await waitFor(() => expect(checkbox).toBeEnabled());
  expect(checkbox).not.toBeChecked();
  fireEvent.click(checkbox);
  await waitFor(() => expect(checkbox).toBeChecked());
  expect(modelsApi.configureDownloadReclaim).toHaveBeenCalledWith(true);
  expect(screen.getByText('260.0 / 2048.0 GiB')).toBeInTheDocument();
  vi.mocked(modelsApi.configureDownloadReclaim).mockRejectedValue(new Error('save failed'));
  fireEvent.click(checkbox);
  await waitFor(() => expect(checkbox).toBeEnabled());
  expect(checkbox).toBeChecked();
});

it('distinguishes two precisions of one repository by their file paths and lists all shards', () => {
  const first = 'S2-L/model-00001-of-00002.mfq';
  const second = 'S2-L/model-00002-of-00002.mfq';
  const other = 'V4-XS/model-00001-of-00002.mfq';
  useJobStore.getState().setJobs([
    { ...download('s2'), payload: { ...download().payload, include: [first, second, '*.json', 'tokenizer*'] } },
    { ...download('v4'), payload: { ...download().payload, include: [other, 'V4-XS/model-00002-of-00002.mfq', '*.json'] } },
  ]);
  const { container } = render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  expect(Array.from(container.querySelectorAll('.download-item-heading strong'), (title) => title.textContent)).toEqual([first, other]);
  const shards = container.querySelector('[data-job-id="s2"] .download-item-files')!;
  expect(shards.querySelector('summary')).toHaveTextContent('View 2 files');
  expect(Array.from(shards.querySelectorAll('li'), (file) => file.textContent)).toEqual([first, second]);
  expect(screen.queryByText('*.json')).not.toBeInTheDocument();
});

it('displays and changes the real server download folder before submitting another download', async () => {
  vi.mocked(modelsApi.modelDirectories).mockResolvedValue({ current_id: 'chosen-folder', current_path: '/chosen/downloads', current_name: 'downloads', parent_id: null, data: [], files: [], model_file_count: 0 });
  vi.mocked(modelsApi.configureDownloadDirectory).mockResolvedValue({ directory: '/chosen/downloads', actual_directory: '/chosen/downloads', default_directory: '/workspace/models', disk_total_bytes: 512 * 2 ** 30, disk_free_bytes: 100 * 2 ** 30 });
  vi.mocked(jobsApi.createJob).mockResolvedValue(download());
  render(<ModelHubPage />);
  await screen.findByText('/workspace/models');
  fireEvent.click(screen.getByRole('button', { name: 'Change' }));
  expect(await screen.findByRole('dialog', { name: 'Choose download folder' })).toBeInTheDocument();
  const use = await screen.findByRole('button', { name: 'Use this folder' });
  await waitFor(() => expect(use).toBeEnabled());
  fireEvent.click(use);
  await screen.findByText('/chosen/downloads');
  expect(screen.getByText('100.0 / 512.0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'SSD free space: 19.5%' })).toBeInTheDocument();
  expect(modelsApi.configureDownloadDirectory).toHaveBeenCalledWith('chosen-folder');
  fireEvent.click(screen.getByRole('button', { name: 'Download' }));
  await waitFor(() => expect(jobsApi.createJob).toHaveBeenCalledWith('download.modelscope', expect.objectContaining({ download_directory: '/chosen/downloads' })));
});

it('shows remaining download disk capacity and refreshes it while mounted', async () => {
  vi.useFakeTimers();
  const first = { directory: null, actual_directory: '/workspace/models', default_directory: '/workspace/models', disk_total_bytes: 2048 * 2 ** 30, disk_free_bytes: 200 * 2 ** 30 };
  vi.mocked(modelsApi.downloadDirectory).mockResolvedValueOnce(first).mockResolvedValue({ ...first, disk_free_bytes: 150 * 2 ** 30 });
  let view: ReturnType<typeof render>;
  await act(async () => { view = render(<ModelHubPage />); });
  expect(screen.getByText('200.0 / 2048.0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'SSD free space: 9.8%' })).toHaveAttribute('aria-valuenow', '9.765625');
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(screen.getByText('150.0 / 2048.0 GiB')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'SSD free space: 7.3%' })).toBeInTheDocument();
  view!.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(modelsApi.downloadDirectory).toHaveBeenCalledTimes(2);
});

it('keeps the checked prefix setting and folder through a transient status failure', async () => {
  vi.useFakeTimers();
  vi.mocked(modelsApi.downloadDirectory).mockResolvedValueOnce({ directory: null,
    actual_directory: '/workspace/models', default_directory: '/workspace/models', evict_prefix_cache: true })
    .mockRejectedValueOnce(new Error('temporarily offline'));
  let view: ReturnType<typeof render>;
  await act(async () => { view = render(<ModelHubPage />); });
  const checkbox = screen.getByRole('checkbox', { name: 'Reclaim SSD prefix budget' });
  expect(checkbox).toBeChecked();
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(checkbox).toBeChecked();
  expect(checkbox).toBeEnabled();
  expect(screen.getByText('/workspace/models')).toBeInTheDocument();
  view!.unmount();
});

it('keeps download controls usable when disk capacity is not reported', async () => {
  vi.mocked(modelsApi.downloadDirectory).mockResolvedValue({ directory: null, actual_directory: '/workspace/models', default_directory: '/workspace/models' });
  render(<ModelHubPage />);
  await screen.findByText('/workspace/models');
  expect(screen.getByRole('progressbar', { name: 'SSD free space: —' })).not.toHaveAttribute('aria-valuenow');
  expect(screen.getByRole('button', { name: 'Change' })).toBeEnabled();
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

it('explains a missing folder and restores the last valid location without changing the download folder', async () => {
  vi.mocked(modelsApi.modelDirectories).mockResolvedValueOnce({ current_id: 'current', current_path: '/workspace/models', current_name: 'models', parent_id: null, data: [], files: [], model_file_count: 0 });
  vi.mocked(modelsApi.modelDirectories).mockRejectedValueOnce(new ApiError(404, { error: { code: 'model_directory_not_found', message: 'model directory not found: /missing', retryable: false, details: {} } }));
  render(<ModelHubPage />);
  await screen.findByText('/workspace/models');
  fireEvent.click(screen.getByRole('button', { name: 'Change' }));
  const location = await screen.findByLabelText('Current directory');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Go' })).toBeEnabled());
  fireEvent.change(location, { target: { value: '/missing' } });
  fireEvent.click(screen.getByRole('button', { name: 'Go' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('This folder could not be found');
  expect(location).toHaveValue('/workspace/models');
  expect(modelsApi.configureDownloadDirectory).not.toHaveBeenCalled();
});

it('cancels and retries only the chosen download through the existing jobs API', async () => {
  useJobStore.getState().setJobs([download()]);
  vi.mocked(jobsApi.cancelJob).mockResolvedValue({ ...download(), status: 'cancelled' });
  vi.mocked(jobsApi.retryJob).mockResolvedValue(download());
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  await screen.findByText('Cancelled');
  expect(jobsApi.cancelJob).toHaveBeenCalledWith('download-1');
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  await waitFor(() => expect(screen.getByText('Queued')).toBeInTheDocument());
  expect(useJobStore.getState().jobs).toHaveLength(1);
  expect(document.querySelector('.download-flight')).not.toBeInTheDocument();
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

it('updates download ETA from aggregate remaining bytes and hides it after completion', async () => {
  useJobStore.getState().setJobs([{ ...download(), status: 'running',
    progress_data: { downloaded_bytes: 1024, total_bytes: 8192, bytes_per_second: 1024 } }]);
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  expect(screen.getByLabelText('Estimated time remaining')).toHaveTextContent('7s remaining');
  act(() => useJobStore.getState().updateJob('download-1', {
    progress_data: { downloaded_bytes: 4096, total_bytes: 8192, bytes_per_second: 2048 },
  }));
  expect(screen.getByLabelText('Estimated time remaining')).toHaveTextContent('2s remaining');
  act(() => useJobStore.getState().updateJob('download-1', {
    progress_data: { downloaded_bytes: 8192, total_bytes: 8192, bytes_per_second: 2048 },
  }));
  expect(screen.getByLabelText('Estimated time remaining')).toHaveTextContent('Finishing download');
  act(() => useJobStore.getState().updateJob('download-1', { status: 'succeeded' }));
  expect(screen.queryByLabelText('Estimated time remaining')).not.toBeInTheDocument();
  expect(jobsApi.cancelJob).not.toHaveBeenCalled();
  expect(jobsApi.retryJob).not.toHaveBeenCalled();
});

it('waits for download size and transfer speed instead of showing an invalid ETA', () => {
  useJobStore.getState().setJobs([{ ...download(), status: 'running' }]);
  render(<ModelHubPage />);
  fireEvent.click(screen.getByRole('tab', { name: 'Download queue' }));
  expect(screen.getByLabelText('Estimated time remaining')).toHaveTextContent('Estimating time remaining…');
  act(() => useJobStore.getState().updateJob('download-1', {
    progress_data: { downloaded_bytes: 1024, total_bytes: 8192, bytes_per_second: 0 },
  }));
  expect(screen.getByLabelText('Estimated time remaining')).toHaveTextContent('Estimating time remaining…');
  act(() => useJobStore.getState().updateJob('download-1', { status: 'queued' }));
  expect(screen.queryByLabelText('Estimated time remaining')).not.toBeInTheDocument();
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
