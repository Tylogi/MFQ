/** 验证模型目录的按需加载、策略传递及注册失败恢复。 */
import { act, renderHook, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import type { ReactNode } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { modelsApi } from '../../shared/api/resources/models';
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

/** 提供真实路由上下文，避免把页面导航行为替换成无条件成功的桩。 */
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

  it('进入模型页时不重播历史失败任务，包括稍后载入的任务记录', () => {
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

  it('只提示本页观察到的加载任务失败，重新进入也不重播', () => {
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
