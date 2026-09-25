/** 管理平台启动、共享推理实例与后台任务，不加载具体业务页面的数据。 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { runtimeApi } from '../shared/api/resources/runtime';
import { jobsApi } from '../shared/api/resources/jobs';
import { setApiBaseUrl, setApiToken } from '../shared/api/client';
import type {
  JobResource,
  RuntimeStatus,
  RuntimeModel,
  RuntimeInstance,
  RuntimeCapabilities,
  RealtimeCapabilities,
  VoiceOutputComponentStatus,
} from '../shared/api/types';
import { studioStatus, studioCredential, startLocalStudio, type StudioStatus } from '../studio';
import { isRuntimeReady, runtimeSelectionNames } from '../features/runtime/modelSelection';
import { errorMessage } from './formatters';
import { useSettings } from '../features/settings/SettingsProvider';
import { useJobStore } from '../stores/jobStore';

interface RuntimeContextValue {
  runtime: RuntimeStatus | null;
  models: RuntimeModel[];
  instances: RuntimeInstance[];
  capabilities: RuntimeCapabilities | null;
  realtime: RealtimeCapabilities | null;
  voiceComponent: VoiceOutputComponentStatus | null;
  studio: StudioStatus | null;
  selectedModel: string;
  /** 更新全局选中模型，随后刷新该实例能力；聊天模块负责派生会话。 */
  setSelectedModel: (model: string) => void;
  /** 仅刷新跨页面共享的实例、模型、任务和能力；返回此次请求是否成功。 */
  refreshRuntime: (quiet?: boolean) => Promise<boolean>;
  /** 将新建任务并入共享列表，自动为运行中的任务建立事件订阅。 */
  addJob: (job: JobResource) => void;
  /** 重新读取平台地址与凭据，刷新运行时并返回是否重新连接成功。 */
  reloadService: () => Promise<boolean>;
  /** 重新订阅断开的后台任务事件流。 */
  retryJobStreams: () => void;
  connectionRevision: number;
  ready: boolean;
  loading: boolean;
  connectionError: string | null;
  refreshError: string | null;
  jobStreamErrors: Record<string, string>;
}

const RuntimeContext = createContext<RuntimeContextValue | null>(null);

/**
 * 为页面提供最小共享运行时，任务流跨路由存活，页面数据由各自模块维护。
 *
 * @param props 组件属性，包含子节点
 */
export function RuntimeProvider({ children }: { children: ReactNode }) {
  const { setContextSize } = useSettings();
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [models, setModels] = useState<RuntimeModel[]>([]);
  const [instances, setInstances] = useState<RuntimeInstance[]>([]);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [realtime, setRealtime] = useState<RealtimeCapabilities | null>(null);
  const [voiceComponent, setVoiceComponent] = useState<VoiceOutputComponentStatus | null>(null);
  const [studio, setStudio] = useState<StudioStatus | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [jobStreamErrors, setJobStreamErrors] = useState<Record<string, string>>({});
  const [connectionRevision, setConnectionRevision] = useState(0);
  const mounted = useRef(false);
  const requestVersion = useRef(0);
  const initializationVersion = useRef(0);
  const selectedRef = useRef(selectedModel);
  selectedRef.current = selectedModel;

  const refreshRuntime = useCallback(async (quiet = true): Promise<boolean> => {
    const version = ++requestVersion.current;
    if (!quiet) setLoading(true);
    try {
      const [nextInstances, nextJobs] = await Promise.all([runtimeApi.runtimeInstances(), jobsApi.jobs(100)]);
      const instance = nextInstances.find(
        (item) => item.model === selectedRef.current && item.state !== 'failed',
      );
      const [statusResult, voiceResult] = await Promise.allSettled([
        runtimeApi.runtimeStatus(instance?.id),
        runtimeApi.voiceOutputComponent(),
      ]);
      if (statusResult.status !== 'fulfilled') throw statusResult.reason;
      const status = statusResult.value;
      const [modelResult, capabilityResult, realtimeResult] = await Promise.allSettled([
        isRuntimeReady(status.runtime_state)
          ? runtimeApi.runtimeModels()
          : Promise.resolve<RuntimeModel[]>([]),
        isRuntimeReady(status.runtime_state)
          ? runtimeApi.runtimeCapabilities(instance?.id)
          : Promise.resolve(null),
        isRuntimeReady(status.runtime_state) ? runtimeApi.realtimeCapabilities() : Promise.resolve(null),
      ]);
      if (!mounted.current || version !== requestVersion.current) return false;
      const nextModels = modelResult.status === 'fulfilled' ? modelResult.value : [];
      setInstances(nextInstances);
      useJobStore.getState().setJobs(nextJobs);
      setRuntime(status);
      setModels(nextModels);
      setCapabilities(capabilityResult.status === 'fulfilled' ? capabilityResult.value : null);
      setRealtime(realtimeResult.status === 'fulfilled' ? realtimeResult.value : null);
      if (voiceResult.status === 'fulfilled') setVoiceComponent(voiceResult.value);
      const names = runtimeSelectionNames(nextModels, nextInstances, nextJobs);
      setSelectedModel((current) =>
        names.includes(current)
          ? current
          : typeof status.model === 'string' && names.includes(status.model)
            ? status.model
            : (names[0] ?? ''),
      );
      setRefreshError(null);
      return true;
    } catch (cause) {
      if (mounted.current && version === requestVersion.current) setRefreshError(errorMessage(cause));
      return false;
    } finally {
      if (mounted.current && version === requestVersion.current) setLoading(false);
    }
  }, []);

  const reloadService = useCallback(async () => {
    const version = ++initializationVersion.current;
    ++requestVersion.current;
    setReady(false);
    setLoading(true);
    setConnectionError(null);
    setRefreshError(null);
    setJobStreamErrors({});
    useJobStore.getState().clearJobStreams();
    try {
      let status = await studioStatus();
      if (!mounted.current || version !== initializationVersion.current) return false;
      setStudio(status);
      if (status?.config.mode === 'local' && !status.reachable) {
        await startLocalStudio();
        status = await studioStatus();
      }
      const token = status ? await studioCredential() : '';
      if (!mounted.current || version !== initializationVersion.current) return false;
      setApiBaseUrl(status?.service_url ?? '');
      setApiToken(token);
      setStudio(status);
      setSelectedModel('');
      selectedRef.current = '';
      useJobStore.getState().setJobs([]);
      setInstances([]);
      setModels([]);
      setRuntime(null);
      setCapabilities(null);
      setConnectionRevision((current) => current + 1);
      const refreshed = await refreshRuntime();
      if (mounted.current && version === initializationVersion.current && refreshed) setReady(true);
      return refreshed && mounted.current && version === initializationVersion.current;
    } catch (cause) {
      if (mounted.current && version === initializationVersion.current) {
        setConnectionError(errorMessage(cause));
        setLoading(false);
      }
      return false;
    }
  }, [refreshRuntime]);

  useEffect(() => {
    mounted.current = true;
    void reloadService();
    return () => {
      mounted.current = false;
      ++requestVersion.current;
      ++initializationVersion.current;
    };
  }, [reloadService]);

  useEffect(() => {
    if (ready && selectedModel) void refreshRuntime();
  }, [ready, selectedModel, refreshRuntime]);

  const activeJobIds = useJobStore((state) => state.activeJobIds.slice().sort().join(','));
  useEffect(() => {
    const capacity = Number(runtime?.max_context);
    if (Number.isFinite(capacity) && capacity > 0) setContextSize(Math.floor(capacity));
  }, [runtime?.max_context, setContextSize]);
  /** 订阅活跃任务，并在各任务恢复传输后分别清除其故障。 */
  const retryJobStreams = useCallback(() => {
    if (!ready) return;
    useJobStore.getState().watchActiveJobs({
      onTerminal: () => {
        void refreshRuntime();
      },
      onEvent: (id) => {
        setJobStreamErrors((current) => {
          if (!(id in current)) return current;
          const next = { ...current };
          delete next[id];
          return next;
        });
      },
      onError: (id, cause) => {
        setJobStreamErrors((current) => ({ ...current, [id]: errorMessage(cause) }));
      },
    });
  }, [ready, refreshRuntime]);
  useEffect(() => {
    if (!ready) return;
    retryJobStreams();
    const active = new Set(activeJobIds ? activeJobIds.split(',') : []);
    setJobStreamErrors((current) => {
      if (Object.keys(current).every((id) => active.has(id))) return current;
      return Object.fromEntries(Object.entries(current).filter(([id]) => active.has(id)));
    });
  }, [ready, activeJobIds, connectionRevision, retryJobStreams]);
  useEffect(() => {
    return () => {
      useJobStore.getState().clearJobStreams();
    };
  }, [connectionRevision]);

  const addJob = useCallback(
    (job: JobResource) => useJobStore.getState().addJob(job),
    [],
  );
  const value = useMemo(
    () => ({
      runtime,
      models,
      instances,
      capabilities,
      realtime,
      voiceComponent,
      studio,
      selectedModel,
      setSelectedModel,
      refreshRuntime,
      addJob,
      reloadService,
      retryJobStreams,
      connectionRevision,
      ready,
      loading,
      connectionError,
      refreshError,
      jobStreamErrors,
    }),
    [
      runtime,
      models,
      instances,
      capabilities,
      realtime,
      voiceComponent,
      studio,
      selectedModel,
      setSelectedModel,
      refreshRuntime,
      addJob,
      reloadService,
      retryJobStreams,
      connectionRevision,
      ready,
      loading,
      connectionError,
      refreshError,
      jobStreamErrors,
    ],
  );
  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

/**
 * 读取平台与共享推理状态，并整合后台任务管理；必须位于 RuntimeProvider 内部。
 *
 * @returns 运行时上下文对象，包含平台就绪状态、模型实例列表与后台任务操作
 */
export function useRuntime(): RuntimeContextValue {
  const value = useContext(RuntimeContext);
  if (!value) throw new Error('RuntimeProvider is missing');
  return value;
}
