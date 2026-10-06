/** Verify on-demand catalog loading, policy propagation, and recovery from registration failures. */
import { act, renderHook, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobResource, ModelArtifact } from '../../shared/api/types';
import { ToastContainer } from '../../shared/ui/Toast';
import { useJobStore } from '../../stores/jobStore';
import { useToastStore } from '../../stores/toastStore';
import { useModelCatalog } from './useModelCatalog';

const state = vi.hoisted(() => ({
  ready: false,
  refreshRuntime: vi.fn(),
  setSelectedModel: vi.fn(),
}));
vi.mock('../../app/RuntimeProvider', () => ({
  useRuntime: () => ({
    ...state,
    runtime: null,
    models: [],
    instances: [],
    jobs: [],
    studio: null,
    selectedModel: '',
  }),
}));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en, contextSize: 8192 }),
}));
vi.mock('../../studio', () => ({ isStudio: () => false, selectLocalModelDirectory: vi.fn() }));

const artifact: ModelArtifact = {
  id: 'model-id',
  name: 'Local Model',
  architecture: 'test',
  format: 'mfq',
  shard_count: 1,
  total_bytes: 128,
  tensor_count: 1,
  record_count: 1,
  dtypes: [],
  complete: true,
  loadable: true,
  modified_at: '2026-09-23T00:00:00Z',
};

/** Provide real router context rather than replacing page navigation with an unconditional-success stub. */
function Wrapper({ children }: { children: ReactNode }) {
  return (
    <MemoryRouter>
      <ToastContainer />
      {children}
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  useJobStore.getState().setJobs([]);
  useToastStore.getState().clearToasts();
  state.ready = false;
  state.refreshRuntime.mockReset().mockResolvedValue(undefined);
  state.setSelectedModel.mockReset();
  vi.spyOn(modelsApi, 'modelArtifacts').mockResolvedValue([artifact]);
});

describe('useModelCatalog', () => {
  it('waits for the platform before fetching catalog and filters by model name', async () => {
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    expect(modelsApi.modelArtifacts).not.toHaveBeenCalled();
    state.ready = true;
    rerender();
    await waitFor(() => expect(result.current.artifacts).toHaveLength(1));
    act(() => result.current.setModelFilter('missing'));
    expect(result.current.filteredArtifacts).toHaveLength(0);
    act(() => result.current.setModelFilter(' LOCAL '));
    expect(result.current.filteredArtifacts).toEqual([artifact]);
  });

  it('passes the page load policy and shared context size before selecting the loaded model', async () => {
    state.ready = true;
    const load = vi
      .spyOn(modelsApi, 'loadModel')
      .mockResolvedValue({ operation_id: 'load-1', status: 'accepted' });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await waitFor(() => expect(result.current.artifacts).toHaveLength(1));
    act(() => {
      result.current.setLoadPinned(true);
      result.current.setLoadIdleTtl(300);
    });
    await act(async () => result.current.loadArtifact(artifact.name));
    expect(load).toHaveBeenCalledWith('Local Model', 8192, 2048, {
      pin: true,
      idle_ttl_seconds: 300,
    });
    expect(state.refreshRuntime).toHaveBeenCalledWith(false);
    expect(state.setSelectedModel).toHaveBeenCalledWith('Local Model');
    expect(result.current.busy).toBe(false);
  });

  it('keeps the server directory dialog open after failed registration', async () => {
    state.ready = true;
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({
      current_id: 'folder-1',
      current_name: 'models',
      current_path: '/models',
      parent_id: null,
      model_file_count: 1,
      data: [],
    });
    vi.spyOn(modelsApi, 'registerModelDirectory').mockRejectedValue(new Error('registration failed'));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    expect(result.current.modelBrowserOpen).toBe(true);
    await act(async () => result.current.registerCurrentModelDirectory());
    expect(screen.getByRole('alert')).toHaveTextContent('registration failed');
    expect(result.current).not.toHaveProperty('error');
    expect(result.current.modelBrowserOpen).toBe(true);
    expect(result.current.busy).toBe(false);
    expect(state.setSelectedModel).not.toHaveBeenCalled();
  });

  it('verifies useModelCatalog test behavior 1', () => {
    const failedJob: JobResource = {
      id: 'old-load',
      kind: 'model.load',
      status: 'failed',
      progress: 0,
      cancel_requested: false,
      payload: { model: 'Local Model' },
      error: { code: 'LOAD_FAILED', message: 'load failed', retryable: false, details: {} },
      created_at: '2026-09-23T10:00:00Z',
      updated_at: '2026-09-23T10:00:01Z',
    };

    useJobStore.getState().setJobs([failedJob]);
    const { unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    unmount();

    useJobStore.getState().setJobs([]);
    renderHook(useModelCatalog, { wrapper: Wrapper });
    act(() => useJobStore.getState().setJobs([failedJob]));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('verifies useModelCatalog test behavior 2', () => {
    const runningJob: JobResource = {
      id: 'new-load',
      kind: 'model.load',
      status: 'running',
      progress: 0,
      cancel_requested: false,
      payload: { model: 'Local Model' },
      created_at: '2026-09-23T10:00:00Z',
      updated_at: '2026-09-23T10:00:01Z',
    };
    const { unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });

    act(() => useJobStore.getState().setJobs([runningJob]));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    act(() => useJobStore.getState().updateJob(runningJob.id, {
      status: 'failed',
      error: { code: 'LOAD_FAILED', message: 'load failed', retryable: false, details: {} },
    }));
    expect(screen.getByRole('alert')).toHaveTextContent('LOAD_FAILED: load failed');

    act(() => useJobStore.getState().setJobs([{ ...useJobStore.getState().jobs[0] }]));
    expect(screen.getAllByRole('alert')).toHaveLength(1);

    unmount();
    useToastStore.getState().clearToasts();
    renderHook(useModelCatalog, { wrapper: Wrapper });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});


/** Create unload jobs that may finish before the initial status refresh. */
function unloadJob(status: JobResource['status'] = 'running'): JobResource {
  return {
    id: 'unload-1', kind: 'model.unload', status, payload: { instance_id: 'instance-1' },
    progress: 0, cancel_requested: false,
    created_at: '2026-10-06T00:00:00Z', updated_at: '2026-10-06T00:00:01Z',
  };
}

it('shows pending unload immediately, prevents double clicks, and waits for the terminal job', async () => {
  let accept!: (value: { operation_id: string; status: 'accepted' }) => void;
  const unload = vi.spyOn(modelsApi, 'unloadModel').mockImplementation(() => new Promise((resolve) => { accept = resolve; }));
  vi.spyOn(jobsApi, 'getJob').mockResolvedValue(unloadJob());
  const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
  let submitting!: Promise<void>;
  act(() => {
    submitting = result.current.unloadInstance('instance-1');
    void result.current.unloadInstance('instance-1');
  });
  expect(unload).toHaveBeenCalledTimes(1);
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(true);
  await act(async () => {
    accept({ operation_id: 'unload-1', status: 'accepted' });
    await submitting;
  });
  expect(result.current.busy).toBe(false);
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(true);
  await act(async () => result.current.unloadInstance('instance-1'));
  expect(unload).toHaveBeenCalledTimes(1);
  act(() => useJobStore.getState().updateJob('unload-1', { status: 'succeeded' }));
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(false);
  expect(screen.getByRole('status')).toHaveTextContent('Model unloaded');
});

it('reports a fast unload failure even if no active job state was observed', async () => {
  vi.spyOn(modelsApi, 'unloadModel').mockResolvedValue({ operation_id: 'unload-1', status: 'accepted' });
  vi.spyOn(jobsApi, 'getJob').mockResolvedValue({ ...unloadJob('failed'),
    error: { code: 'runtime_busy', message: 'Runtime is processing a request', retryable: true, details: {} },
  });
  const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
  await act(async () => result.current.unloadInstance('instance-1'));
  expect(screen.getByRole('alert')).toHaveTextContent('runtime_busy: Runtime is processing a request');
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(false);
});

it('fetches the failure reason when an SSE state event has no error details', async () => {
  vi.spyOn(modelsApi, 'unloadModel').mockResolvedValue({ operation_id: 'unload-1', status: 'accepted' });
  vi.spyOn(jobsApi, 'getJob').mockResolvedValueOnce(unloadJob()).mockResolvedValueOnce({
    ...unloadJob('failed'), error: { code: 'stop_failed', message: 'Could not stop process', retryable: true, details: {} },
  });
  const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
  await act(async () => result.current.unloadInstance('instance-1'));
  await act(async () => useJobStore.getState().updateJob('unload-1', { status: 'failed' }));
  expect(screen.getByRole('alert')).toHaveTextContent('stop_failed: Could not stop process');
  act(() => useJobStore.getState().setJobs([...useJobStore.getState().jobs]));
  expect(screen.getAllByRole('alert')).toHaveLength(1);
});

it('restores the unload action when submission fails', async () => {
  vi.spyOn(modelsApi, 'unloadModel').mockRejectedValue(new Error('connection lost'));
  const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
  await act(async () => result.current.unloadInstance('instance-1'));
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(false);
  expect(result.current.busy).toBe(false);
  expect(screen.getByRole('alert')).toHaveTextContent('connection lost');
});

it('recovers accepted unload tracking when the first job read fails', async () => {
  vi.spyOn(modelsApi, 'unloadModel').mockResolvedValue({ operation_id: 'unload-1', status: 'accepted' });
  vi.spyOn(jobsApi, 'getJob').mockRejectedValue(new Error('temporary read failure'));
  const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
  await act(async () => result.current.unloadInstance('instance-1'));
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(true);
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  act(() => useJobStore.getState().setJobs([unloadJob('succeeded')]));
  expect(result.current.unloadingInstanceIds.has('instance-1')).toBe(false);
});
