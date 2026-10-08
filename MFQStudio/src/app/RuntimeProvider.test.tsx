/** 验证应用级故障分流、初次连接重试和后台任务流恢复。 */
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { runtimeApi } from '../shared/api/resources/runtime';
import { jobsApi } from '../shared/api/resources/jobs';
import type { JobEventResource, JobResource, RuntimeStatus } from '../shared/api/types';
import { studioStatus } from '../studio';
import { useJobStore } from '../stores/jobStore';
import { RuntimeProvider, useRuntime } from './RuntimeProvider';

let capturedRuntime: ReturnType<typeof useRuntime>;

vi.mock('../studio', () => ({
  studioStatus: vi.fn(),
  studioCredential: vi.fn(),
  startLocalStudio: vi.fn(),
}));
vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ setContextSize: vi.fn() }),
}));

/** 以可操作的状态快照验证 Provider 的各类故障互不覆盖。 */
function RuntimeFixture() {
  const runtime = useRuntime();
  capturedRuntime = runtime;
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

describe('RuntimeProvider 故障状态', () => {
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

  it('首次刷新失败保持未就绪，重连成功后清除错误', async () => {
    vi.mocked(runtimeApi.runtimeInstances).mockRejectedValueOnce(new Error('server offline'));
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('refresh-error').textContent).toBe('server offline'));
    expect(screen.getByTestId('ready').textContent).toBe('false');

    fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'));
    expect(screen.getByTestId('refresh-error').textContent).toBe('');
  });

  it('点击立即请求 ctx 重载并持有状态直到服务完成', async () => {
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

  it('does not import late jobs from the previous connection', async () => {
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    const oldAddJob = capturedRuntime.addJob;
    fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    act(() => oldAddJob({ ...activeJob, status: 'succeeded' }));
    expect(useJobStore.getState().jobs).toEqual([]);
    act(() => capturedRuntime.addJob({ ...activeJob, id: 'current-job', status: 'succeeded' }));
    expect(useJobStore.getState().jobs.map((job) => job.id)).toEqual(['current-job']);
  });

  it('does not restore deleted history from an older background poll after a fresh runtime refresh', async () => {
    vi.useFakeTimers();
    const finished = { ...activeJob, status: 'succeeded' } as JobResource;
    vi.mocked(jobsApi.jobs).mockResolvedValue([finished]);
    const view = render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    try {
      await act(async () => {});
      expect(capturedRuntime.ready).toBe(true);
      let resolve!: (jobs: JobResource[]) => void;
      vi.mocked(jobsApi.jobs).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
      await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
      expect(resolve).toBeDefined();
      vi.mocked(jobsApi.jobs).mockResolvedValue([]);
      await act(async () => { await capturedRuntime.refreshRuntime(); });
      expect(useJobStore.getState().jobs).toEqual([]);
      await act(async () => { resolve([finished]); });
      expect(useJobStore.getState().jobs).toEqual([]);
    } finally { view.unmount(); vi.useRealTimers(); }
  });

  it.each(['runtime', 'background'])('does not drop a newly accepted job when an older %s snapshot arrives', async (source) => {
    vi.useFakeTimers();
    const view = render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    try {
      await act(async () => {});
      expect(capturedRuntime.ready).toBe(true);
      let resolve!: (jobs: JobResource[]) => void;
      vi.mocked(jobsApi.jobs).mockReturnValueOnce(new Promise((done) => { resolve = done; }));
      let pending: Promise<boolean> | undefined;
      await act(async () => {
        if (source === 'runtime') pending = capturedRuntime.refreshRuntime();
        else await vi.advanceTimersByTimeAsync(1000);
      });
      expect(resolve).toBeDefined();
      const accepted = { ...activeJob, id: 'newly-accepted', status: 'succeeded' } as JobResource;
      act(() => capturedRuntime.addJob(accepted));
      vi.mocked(jobsApi.jobs).mockResolvedValue([accepted]);
      await act(async () => { resolve([]); await pending; });
      expect(useJobStore.getState().jobs.map((job) => job.id)).toEqual(['newly-accepted']);
    } finally { view.unmount(); vi.useRealTimers(); }
  });

  it('clears old ctx reload state without refreshing the new server when the old request completes', async () => {
    let finish!: (status: RuntimeStatus) => void;
    vi.spyOn(runtimeApi, 'reloadRuntime').mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    render(<RuntimeProvider><RuntimeFixture /></RuntimeProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    fireEvent.click(screen.getByRole('button', { name: 'Reload model' }));
    expect(screen.getByTestId('reloading')).toHaveTextContent('instance-a');
    fireEvent.click(screen.getByRole('button', { name: 'Reconnect' }));
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('reloading')).toHaveTextContent('{}');
    const calls = vi.mocked(runtimeApi.runtimeInstances).mock.calls.length;
    await act(async () => { finish({ max_context: 8192 }); });
    expect(runtimeApi.runtimeInstances).toHaveBeenCalledTimes(calls);
  });

  it('刷新恢复不会清除任务流故障，任务收到事件后才恢复', async () => {
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

  it.each(['model.load', 'download.modelscope'])('返回窗口后重新同步 %s 的进度并恢复断开的事件流', async (kind) => {
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
