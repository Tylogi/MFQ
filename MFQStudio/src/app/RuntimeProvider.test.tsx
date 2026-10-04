/** Verify application-level failure routing, initial connection retry, and background job-stream recovery. */
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { runtimeApi } from '../shared/api/resources/runtime';
import { jobsApi } from '../shared/api/resources/jobs';
import type { JobEventResource, JobResource, RuntimeStatus } from '../shared/api/types';
import { studioStatus } from '../studio';
import { useJobStore } from '../stores/jobStore';
import { RuntimeProvider, useRuntime } from './RuntimeProvider';

vi.mock('../studio', () => ({
  studioStatus: vi.fn(),
  studioCredential: vi.fn(),
  startLocalStudio: vi.fn(),
}));
vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ setContextSize: vi.fn() }),
}));
/** Verify with an interactive state snapshot that Provider failures do not overwrite one another. */
function RuntimeFixture() {
  const runtime = useRuntime();
  return (
    <>
      <span data-testid="ready">{String(runtime.ready)}</span>
      <span data-testid="connection-error">{runtime.connectionError}</span>
      <span data-testid="refresh-error">{runtime.refreshError}</span>
      <span data-testid="stream-error">{Object.values(runtime.jobStreamErrors).join('; ')}</span>
      <button onClick={() => void runtime.reloadService()} type="button">Reconnect</button>
      <button onClick={() => void runtime.refreshRuntime()} type="button">Refresh</button>
      <button onClick={runtime.retryJobStreams} type="button">Retry stream</button>
      <span data-testid="reloading">{JSON.stringify(runtime.reloadingInstances)}</span>
      <button onClick={() => void runtime.reloadModelContext('instance-a', 8192)} type="button">Reload model</button>
    </>
  );
}

const activeJob = {
  id: 'job-1',
  status: 'running',
  kind: 'model.load',
  payload: {},
} as JobResource;

describe('describes RuntimeProvider test behavior 1', () => {
  beforeEach(() => {
    useJobStore.getState().clearJobStreams();
    useJobStore.getState().setJobs([]);
    vi.restoreAllMocks();
    vi.mocked(studioStatus).mockResolvedValue(null);
    vi.spyOn(runtimeApi, 'runtimeInstances').mockResolvedValue([]);
    vi.spyOn(jobsApi, 'jobs').mockResolvedValue([]);
    vi.spyOn(runtimeApi, 'runtimeStatus').mockResolvedValue({ runtime_state: 'idle' } as RuntimeStatus);
    vi.spyOn(runtimeApi, 'voiceOutputComponent').mockResolvedValue({} as Awaited<ReturnType<typeof runtimeApi.voiceOutputComponent>>);
  });

  it('verifies RuntimeProvider test behavior 2', async () => {
    vi.mocked(runtimeApi.runtimeInstances).mockRejectedValueOnce(new Error('server offline'));
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('refresh-error').textContent).toBe('server offline'));
    expect(screen.getByTestId('ready').textContent).toBe('false');

    fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'));
    expect(screen.getByTestId('refresh-error').textContent).toBe('');
  });

  it('verifies RuntimeProvider test behavior 3', async () => {
    let finish!: (status: RuntimeStatus) => void;
    const reload = vi.spyOn(runtimeApi, 'reloadRuntime').mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    fireEvent.click(screen.getByRole('button', { name: 'Reload model' }));
    expect(reload).toHaveBeenCalledExactlyOnceWith(8192, 'instance-a');
    expect(screen.getByTestId('reloading')).toHaveTextContent('"instance-a":8192');
    await act(async () => { finish({ max_context: 8192 }); });
    await waitFor(() => expect(screen.getByTestId('reloading')).toHaveTextContent('{}'));
  });

  it('verifies RuntimeProvider test behavior 4', async () => {
    vi.mocked(jobsApi.jobs).mockResolvedValue([activeJob]);
    let rejectStream: ((cause: Error) => void) | undefined;
    let resumedEvent: ((event: JobEventResource) => void) | undefined;
    vi.spyOn(jobsApi, 'streamJobEvents')
      .mockImplementationOnce(() => new Promise<void>((_resolve, reject) => { rejectStream = reject; }))
      .mockImplementationOnce((_id, onEvent) => {
        resumedEvent = onEvent;
        return new Promise<void>(() => {});
      });
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(rejectStream).toBeDefined());
    act(() => rejectStream!(new Error('stream offline')));
    await waitFor(() => expect(screen.getByTestId('stream-error').textContent).toBe('stream offline'));

    vi.mocked(runtimeApi.runtimeInstances).mockRejectedValueOnce(new Error('refresh offline'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await waitFor(() => expect(screen.getByTestId('refresh-error').textContent).toBe('refresh offline'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await waitFor(() => expect(screen.getByTestId('refresh-error').textContent).toBe(''));
    expect(screen.getByTestId('stream-error').textContent).toBe('stream offline');

    fireEvent.click(screen.getByRole('button', { name: 'Retry stream' }));
    await waitFor(() => expect(resumedEvent).toBeDefined());
    expect(screen.getByTestId('stream-error').textContent).toBe('stream offline');
    act(() => resumedEvent!({
      job_id: 'job-1',
      sequence: 1,
      type: 'progress',
      level: 'info',
      data: {},
      progress: 50,
      created_at: '2026-09-23T10:10:00Z',
    }));
    await waitFor(() => expect(screen.getByTestId('stream-error').textContent).toBe(''));
  });

  it.each(['model.load', 'download.modelscope'])('verifies RuntimeProvider test behavior 5', async (kind) => {
    const job = { ...activeJob, kind, progress: 0.1, updated_at: '2026-10-03T00:00:00Z' } as JobResource;
    vi.mocked(jobsApi.jobs).mockResolvedValue([job]);
    const stream = vi.spyOn(jobsApi, 'streamJobEvents').mockImplementation(() => new Promise<void>(() => {}));
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    await waitFor(() => expect(stream).toHaveBeenCalledTimes(1));
    vi.mocked(jobsApi.jobs).mockResolvedValue([{ ...job, progress: 0.7, updated_at: '2026-10-03T00:00:05Z' }]);
    fireEvent(window, new Event('focus'));
    await waitFor(() => expect(useJobStore.getState().jobs[0].progress).toBe(0.7));
    await waitFor(() => expect(stream).toHaveBeenCalledTimes(2));
    const calls = stream.mock.calls;
    expect(calls[0][2].aborted).toBe(true);
    expect(calls[1][2].aborted).toBe(false);
  });
});
