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
import { readModelDirectory, saveModelDirectory } from './modelDirectoryPreference';
import { setApiBaseUrl } from '../../shared/api/client';
import { isStudio, selectLocalModelDirectory } from '../../studio';
import { DEFAULT_LOAD_CONTEXT } from './loadContextDefaults';

const state = vi.hoisted(() => ({
  ready: false,
  connectionRevision: 0,
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
vi.mock('../../studio', () => ({ isStudio: vi.fn(() => false), selectLocalModelDirectory: vi.fn() }));

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
  state.connectionRevision = 0;
  setApiBaseUrl('');
  localStorage.clear();
  vi.mocked(isStudio).mockReturnValue(false);
  state.refreshRuntime.mockReset().mockResolvedValue(undefined);
  state.setSelectedModel.mockReset();
  vi.spyOn(modelsApi, 'modelArtifacts').mockResolvedValue([artifact]);
});

describe('useModelCatalog', () => {
  it('prepares deletion without removing files and requires explicit confirmation', async () => {
    state.ready = true;
    const plan = { model_id: artifact.id, name: artifact.name, files: ['/models/model.mfq'], total_bytes: 128, token: 'a'.repeat(64) };
    vi.spyOn(modelsApi, 'modelDeletionPlan').mockResolvedValue(plan);
    const remove = vi.spyOn(modelsApi, 'deleteModel').mockResolvedValue({ files: 1, total_bytes: 128 });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await waitFor(() => expect(result.current.artifacts).toEqual([artifact]));
    await act(async () => result.current.prepareModelDeletion(artifact.id));
    expect(result.current.modelDeletion).toEqual(plan);
    expect(remove).not.toHaveBeenCalled();
    act(() => result.current.cancelModelDeletion());
    expect(result.current.modelDeletion).toBeNull();
    expect(remove).not.toHaveBeenCalled();
    await act(async () => result.current.prepareModelDeletion(artifact.id));
    vi.mocked(modelsApi.modelArtifacts).mockResolvedValue([]);
    await act(async () => result.current.confirmModelDeletion());
    expect(remove).toHaveBeenCalledExactlyOnceWith(artifact.id, plan.token);
    expect(result.current.modelDeletion).toBeNull();
    expect(result.current.artifacts).toEqual([]);
  });

  it('connection changes dismiss deletion confirmation and discard old previews', async () => {
    state.ready = true;
    let resolve!: (plan: Awaited<ReturnType<typeof modelsApi.modelDeletionPlan>>) => void;
    vi.spyOn(modelsApi, 'modelDeletionPlan').mockReturnValue(new Promise(done => { resolve = done; }));
    const remove = vi.spyOn(modelsApi, 'deleteModel');
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await waitFor(() => expect(result.current.artifacts).toEqual([artifact]));
    let pending!: Promise<void>;
    act(() => { pending = result.current.prepareModelDeletion(artifact.id); });
    setApiBaseUrl('http://other-server:8090');
    state.connectionRevision++;
    rerender();
    await act(async () => {
      resolve({ model_id: artifact.id, name: artifact.name, files: ['/models/model.mfq'], total_bytes: 128, token: 'a'.repeat(64) });
      await pending;
    });
    expect(result.current.modelDeletion).toBeNull();
    await act(async () => result.current.confirmModelDeletion());
    expect(remove).not.toHaveBeenCalled();
  });

  it('deletion failure preserves the model and confirmation for retry', async () => {
    state.ready = true;
    const plan = { model_id: artifact.id, name: artifact.name, files: ['/models/model.mfq'], total_bytes: 128, token: 'a'.repeat(64) };
    vi.spyOn(modelsApi, 'modelDeletionPlan').mockResolvedValue(plan);
    vi.spyOn(modelsApi, 'deleteModel').mockRejectedValue(new Error('permission denied'));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await waitFor(() => expect(result.current.artifacts).toEqual([artifact]));
    await act(async () => result.current.prepareModelDeletion(artifact.id));
    await act(async () => result.current.confirmModelDeletion());
    expect(result.current.modelDeletion).toEqual(plan);
    expect(result.current.artifacts).toEqual([artifact]);
    expect(result.current.busy).toBe(false);
  });

  it('removes deleted checkpoints on the next fresh scan without a job change', async () => {
    vi.useFakeTimers();
    try {
      state.ready = true;
      const { result, unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
      await act(async () => { await Promise.resolve(); });
      expect(modelsApi.modelArtifacts).toHaveBeenLastCalledWith(true);
      expect(result.current.artifacts).toEqual([artifact]);
      vi.mocked(modelsApi.modelArtifacts).mockResolvedValue([]);
      await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
      expect(result.current.artifacts).toEqual([]);
      expect(result.current.filteredArtifacts).toEqual([]);
      unmount();
      const calls = vi.mocked(modelsApi.modelArtifacts).mock.calls.length;
      await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
      expect(modelsApi.modelArtifacts).toHaveBeenCalledTimes(calls);
    } finally { vi.useRealTimers(); }
  });

  it('rescans on focus or visibility without overlapping catalog reads', async () => {
    vi.useFakeTimers();
    try {
      const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
      state.ready = true;
      let resolve!: (items: ModelArtifact[]) => void;
      vi.mocked(modelsApi.modelArtifacts).mockReturnValue(new Promise(done => { resolve = done; }));
      const { result, unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
      await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
      expect(modelsApi.modelArtifacts).not.toHaveBeenCalled();
      hidden.mockReturnValue(false);
      act(() => { document.dispatchEvent(new Event('visibilitychange')); });
      act(() => { window.dispatchEvent(new Event('focus')); });
      await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
      expect(modelsApi.modelArtifacts).toHaveBeenCalledTimes(1);
      await act(async () => { resolve([artifact]); await Promise.resolve(); });
      expect(result.current.artifacts).toEqual([artifact]);
      vi.mocked(modelsApi.modelArtifacts).mockResolvedValue([]);
      await act(async () => { window.dispatchEvent(new Event('focus')); await Promise.resolve(); });
      expect(modelsApi.modelArtifacts).toHaveBeenCalledTimes(2);
      expect(result.current.artifacts).toEqual([]);
      unmount();
      act(() => { window.dispatchEvent(new Event('focus')); });
      expect(modelsApi.modelArtifacts).toHaveBeenCalledTimes(2);
    } finally { vi.useRealTimers(); }
  });

  it('does not resurrect an old catalog response after changing servers', async () => {
    state.ready = true;
    let resolve!: (items: ModelArtifact[]) => void;
    vi.mocked(modelsApi.modelArtifacts).mockReturnValueOnce(new Promise(done => { resolve = done; }))
      .mockResolvedValue([]);
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    setApiBaseUrl('http://other-server:8090');
    state.connectionRevision++;
    rerender();
    await act(async () => { await Promise.resolve(); });
    await act(async () => { resolve([artifact]); await Promise.resolve(); });
    expect(result.current.artifacts).toEqual([]);
  });

  it('does not overwrite a newly registered catalog with an older poll', async () => {
    state.ready = true;
    let resolve!: (items: ModelArtifact[]) => void;
    vi.mocked(modelsApi.modelArtifacts).mockReturnValueOnce(new Promise(done => { resolve = done; }))
      .mockResolvedValue([artifact]);
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'folder', current_name: 'models',
      current_path: '/models', parent_id: null, model_file_count: 1, data: [] });
    vi.spyOn(modelsApi, 'registerModelDirectory').mockResolvedValue([artifact]);
    vi.spyOn(modelsApi, 'loadModel').mockResolvedValue({ operation_id: 'load', status: 'accepted' });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    await act(async () => result.current.registerCurrentModelDirectory());
    expect(result.current.artifacts).toEqual([artifact]);
    await act(async () => { resolve([{ ...artifact, id: 'deleted', name: 'Deleted model' }]); await Promise.resolve(); });
    expect(result.current.artifacts).toEqual([artifact]);
  });

  it('does not continue registration or save the old folder after changing server', async () => {
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'old-folder', current_name: 'models',
      current_path: '/old/models', parent_id: null, model_file_count: 1, data: [] });
    let resolve!: (items: ModelArtifact[]) => void;
    vi.spyOn(modelsApi, 'registerModelDirectory').mockReturnValue(new Promise((done) => { resolve = done; }));
    const load = vi.spyOn(modelsApi, 'loadModel');
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    let pending!: Promise<void>;
    act(() => { pending = result.current.registerCurrentModelDirectory(); });
    setApiBaseUrl('http://other-server:8090');
    state.connectionRevision++;
    rerender();
    await act(async () => { resolve([artifact]); await pending; });
    expect(modelsApi.modelArtifacts).not.toHaveBeenCalled();
    expect(load).not.toHaveBeenCalled();
    expect(state.setSelectedModel).not.toHaveBeenCalled();
    expect(state.refreshRuntime).not.toHaveBeenCalled();
    expect(readModelDirectory()).toBe('/');
  });

  it('does not load a model when the registered catalog arrives after changing server', async () => {
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'folder', current_name: 'models',
      current_path: '/old/models', parent_id: null, model_file_count: 1, data: [] });
    vi.spyOn(modelsApi, 'registerModelDirectory').mockResolvedValue([artifact]);
    let resolve!: (items: ModelArtifact[]) => void;
    vi.mocked(modelsApi.modelArtifacts).mockReturnValue(new Promise((done) => { resolve = done; }));
    const load = vi.spyOn(modelsApi, 'loadModel');
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    let pending!: Promise<void>;
    await act(async () => { pending = result.current.registerCurrentModelDirectory(); await Promise.resolve(); });
    expect(modelsApi.modelArtifacts).toHaveBeenCalledWith(true);
    setApiBaseUrl('http://other-server:8090');
    state.connectionRevision++;
    rerender();
    await act(async () => { resolve([artifact]); await pending; });
    expect(load).not.toHaveBeenCalled();
    expect(result.current.artifacts).toEqual([]);
    expect(state.setSelectedModel).not.toHaveBeenCalled();
  });

  it('ignores a completed model load after leaving the model page', async () => {
    let resolve!: (value: Awaited<ReturnType<typeof modelsApi.loadModel>>) => void;
    vi.spyOn(modelsApi, 'loadModel').mockReturnValue(new Promise((done) => { resolve = done; }));
    const { result, unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
    let pending!: Promise<void>;
    act(() => { pending = result.current.loadArtifact(artifact.name); });
    unmount();
    await act(async () => { resolve({ operation_id: 'load', status: 'accepted' }); await pending; });
    expect(state.setSelectedModel).not.toHaveBeenCalled();
    expect(state.refreshRuntime).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('does not select an old model if the server changes during the post-load refresh', async () => {
    vi.spyOn(modelsApi, 'loadModel').mockResolvedValue({ operation_id: 'load', status: 'accepted' });
    let resolve!: () => void;
    state.refreshRuntime.mockReturnValue(new Promise<void>((done) => { resolve = done; }));
    const { result, rerender } = renderHook(useModelCatalog, { wrapper: Wrapper });
    let pending!: Promise<void>;
    await act(async () => { pending = result.current.loadArtifact(artifact.name); await Promise.resolve(); });
    expect(state.refreshRuntime).toHaveBeenCalledWith(false);
    setApiBaseUrl('http://other-server:8090');
    state.connectionRevision++;
    rerender();
    await act(async () => { resolve(); await pending; });
    expect(state.setSelectedModel).not.toHaveBeenCalled();
  });

  it('pauses hidden-page polling and resumes immediately without overlapping requests', async () => {
    vi.useFakeTimers();
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    state.ready = true;
    let resolve!: () => void;
    state.refreshRuntime.mockReturnValue(new Promise<void>((done) => { resolve = done; }));
    const { unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(state.refreshRuntime).not.toHaveBeenCalled();
    hidden.mockReturnValue(false);
    act(() => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
    act(() => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
    await act(async () => { resolve(); await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(state.refreshRuntime).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it.each(['load', 'unload'] as const)('does not show a late %s error after leaving the model page', async (operation) => {
    let reject!: (cause: Error) => void;
    const response = new Promise<never>((_resolve, fail) => { reject = fail; });
    vi.spyOn(modelsApi, operation === 'load' ? 'loadModel' : 'unloadModel').mockReturnValue(response);
    const { result, unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
    let pending!: Promise<void>;
    act(() => { pending = operation === 'load' ? result.current.loadArtifact(artifact.name) : result.current.unloadInstance('instance'); });
    unmount();
    await act(async () => { reject(new Error('old server error')); await pending; });
    expect(useToastStore.getState().toasts).toEqual([]);
    expect(state.refreshRuntime).not.toHaveBeenCalled();
  });

  it.each([
    { can_open_directory: true, can_open_in_finder: true },
    { can_open_directory: true, can_open_in_finder: false },
    { can_open_in_finder: true },
  ])('opens the browsed directory without loading, registering, or changing the saved folder (%j)', async (capability) => {
    saveModelDirectory('/chosen/models');
    const open = vi.spyOn(modelsApi, 'openModelDirectory').mockResolvedValue({ opened: true });
    const register = vi.spyOn(modelsApi, 'registerModelDirectory');
    const load = vi.spyOn(modelsApi, 'loadModel');
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'folder', current_name: 'models',
      current_path: '/chosen/models', parent_id: null, model_file_count: 0, data: [], ...capability, files: [] });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    await act(async () => result.current.openCurrentDirectory());
    expect(open).toHaveBeenCalledWith('folder');
    expect(register).not.toHaveBeenCalled();
    expect(load).not.toHaveBeenCalled();
    expect(readModelDirectory()).toBe('/chosen/models');
    expect(result.current.busy).toBe(false);
  });

  it.each([
    { can_open_directory: false, can_open_in_finder: false },
    { can_open_directory: false, can_open_in_finder: true },
  ])('does not launch a file manager when unavailable (%j)', async (capability) => {
    const open = vi.spyOn(modelsApi, 'openModelDirectory').mockResolvedValue({ opened: true });
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'folder', current_name: 'models',
      current_path: '/chosen/models', parent_id: null, model_file_count: 0, data: [], ...capability });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    await act(async () => result.current.openCurrentDirectory());
    expect(open).not.toHaveBeenCalled();
    expect(result.current.busy).toBe(false);
  });

  it('restores the folder path and opens that folder rather than a virtual directory root', async () => {
    saveModelDirectory('/chosen/models');
    const browse = vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({
      current_id: 'folder', current_name: 'models', current_path: '/chosen/models', parent_id: 'parent', model_file_count: 1, data: [],
    });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    expect(result.current.modelFolderPath).toBe('/chosen/models');
    await act(async () => result.current.chooseModelDirectory());
    expect(browse).toHaveBeenCalledWith(null, '/chosen/models');
    act(() => result.current.setModelBrowserOpen(false));
    expect(readModelDirectory()).toBe('/chosen/models');
  });

  it('shows the real root during a pending read and cancellation cannot reopen or block the picker', async () => {
    let resolve!: (value: Awaited<ReturnType<typeof modelsApi.modelDirectories>>) => void;
    const browse = vi.spyOn(modelsApi, 'modelDirectories').mockReturnValue(new Promise((done) => { resolve = done; }));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    let pending!: Promise<void>;
    act(() => { pending = result.current.chooseModelDirectory(); });
    expect(browse).toHaveBeenCalledWith(null, '/');
    expect(result.current.modelBrowserOpen).toBe(true);
    expect(result.current.modelDirectoryPath).toBe('/');
    act(() => result.current.setModelBrowserOpen(false));
    await act(async () => { resolve({ current_id: 'root', current_name: '/', current_path: '/', parent_id: null, model_file_count: 0, data: [] }); await pending; });
    expect(result.current.modelBrowserOpen).toBe(false);
    expect(result.current.busy).toBe(false);
    expect(readModelDirectory()).toBe('/');
  });

  it('remembers a registered folder even if loading the model fails', async () => {
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue({ current_id: 'folder', current_name: 'models', current_path: '/actual/models', parent_id: null, model_file_count: 1, data: [] });
    vi.spyOn(modelsApi, 'registerModelDirectory').mockResolvedValue([artifact]);
    vi.spyOn(modelsApi, 'loadModel').mockRejectedValue(new Error('load failed'));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    await act(async () => result.current.registerCurrentModelDirectory());
    expect(result.current.modelFolderPath).toBe('/actual/models');
    expect(readModelDirectory()).toBe('/actual/models');
  });

  it('keeps artifact file browsing separate from the saved model folder', async () => {
    saveModelDirectory('/chosen/models');
    vi.spyOn(modelsApi, 'modelArtifactDirectory').mockResolvedValue({ current_id: 'files', current_name: 'checkpoint', current_path: '/other/checkpoint', parent_id: null, model_file_count: 1, data: [] });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.openModelFiles('model'));
    expect(result.current.modelFolderPath).toBe('/chosen/models');
    expect(readModelDirectory()).toBe('/chosen/models');
  });

  it('passes and remembers the chosen native folder', async () => {
    vi.mocked(isStudio).mockReturnValue(true);
    saveModelDirectory('/native/start');
    vi.mocked(selectLocalModelDirectory).mockResolvedValue({ path: '/native/selected', names: [artifact.name] });
    vi.spyOn(modelsApi, 'loadModel').mockResolvedValue({ operation_id: 'load', status: 'accepted' });
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.chooseModelDirectory());
    expect(selectLocalModelDirectory).toHaveBeenCalledWith('/native/start');
    expect(result.current.modelFolderPath).toBe('/native/selected');
  });

  it('refreshes remaining memory while visible and stops polling after leaving the page', async () => {
    vi.useFakeTimers();
    try {
      state.ready = true;
      const { unmount } = renderHook(useModelCatalog, { wrapper: Wrapper });
      await act(async () => { await Promise.resolve(); });
      expect(state.refreshRuntime).toHaveBeenCalledWith(true);
      state.refreshRuntime.mockClear();
      await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
      expect(state.refreshRuntime).toHaveBeenCalledTimes(1);
      unmount();
      state.refreshRuntime.mockClear();
      await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
      expect(state.refreshRuntime).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

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
      result.current.setLoadDeviceIds(['2']);
    });
    await act(async () => result.current.loadArtifact(artifact.name));
    expect(load).toHaveBeenCalledWith('Local Model', 32768, 'auto', {
      pin: true,
      idle_ttl_seconds: 300,
      device_ids: ['2'],
      context_management: DEFAULT_LOAD_CONTEXT,
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

  it('opens the actual artifact directory without loading or registering the model', async () => {
    const listing = { current_id: 'actual-dir', current_name: 'S4-L', current_path: '/external/models/S4-L', parent_id: 'parent', model_file_count: 1, data: [] };
    const locate = vi.spyOn(modelsApi, 'modelArtifactDirectory').mockResolvedValue(listing);
    const register = vi.spyOn(modelsApi, 'registerModelDirectory');
    const load = vi.spyOn(modelsApi, 'loadModel');
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.openModelFiles('model-id'));
    expect(locate).toHaveBeenCalledWith('model-id');
    expect(result.current.modelBrowser).toEqual(listing);
    expect(result.current.modelDirectoryPath).toBe('/external/models/S4-L');
    expect(result.current.modelFilesMode).toBe(true);
    expect(result.current.modelBrowserOpen).toBe(true);
    expect(register).not.toHaveBeenCalled();
    expect(load).not.toHaveBeenCalled();
    vi.spyOn(modelsApi, 'modelDirectories').mockResolvedValue(listing);
    await act(async () => result.current.chooseModelDirectory());
    expect(result.current.modelFilesMode).toBe(false);
  });

  it('reports directory lookup failures without opening a stale model folder', async () => {
    vi.spyOn(modelsApi, 'modelArtifactDirectory').mockRejectedValue(new Error('model directory unavailable'));
    const { result } = renderHook(useModelCatalog, { wrapper: Wrapper });
    await act(async () => result.current.openModelFiles('removed-model'));
    expect(result.current.modelBrowserOpen).toBe(false);
    expect(result.current.busy).toBe(false);
    expect(screen.getByRole('alert')).toHaveTextContent('model directory unavailable');
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

  it('reports an interrupted load without requiring an error in the first terminal SSE event', () => {
    const job = { id: 'interrupted-load', kind: 'model.load', status: 'running', progress: .2,
      cancel_requested: false, payload: { model: 'Local Model' },
      created_at: '2026-09-23T10:00:00Z', updated_at: '2026-09-23T10:00:01Z' } as JobResource;
    renderHook(useModelCatalog, { wrapper: Wrapper });
    act(() => useJobStore.getState().setJobs([job]));
    act(() => useJobStore.getState().updateJob(job.id, { status: 'interrupted' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Model loading was interrupted. Please load the model again.');
    act(() => useJobStore.getState().setJobs([{ ...job, status: 'interrupted' }]));
    expect(screen.getAllByRole('alert')).toHaveLength(1);
  });

  it('waits for the failure detail instead of losing the error after a terminal state-only event', () => {
    const job = { id: 'failed-load', kind: 'model.load', status: 'running', progress: .2,
      cancel_requested: false, payload: { model: 'Local Model' },
      created_at: '2026-09-23T10:00:00Z', updated_at: '2026-09-23T10:00:01Z' } as JobResource;
    renderHook(useModelCatalog, { wrapper: Wrapper });
    act(() => useJobStore.getState().setJobs([job]));
    act(() => useJobStore.getState().updateJob(job.id, { status: 'failed' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    act(() => useJobStore.getState().updateJob(job.id, {
      error: { code: 'LOAD_FAILED', message: 'load failed', retryable: false, details: {} },
    }));
    expect(screen.getByRole('alert')).toHaveTextContent('LOAD_FAILED: load failed');
  });
});
