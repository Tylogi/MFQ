/** 验证应用级故障分流、初次连接重试和后台任务流恢复。 */
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

/** 以可操作的状态快照验证 Provider 的各类故障互不覆盖。 */
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
});
