/** Coordinate runtime snapshots, model selection, and background job recovery. */
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
import { ApiError, browserServiceUrl, getApiToken, setApiBaseUrl, setApiToken } from '../shared/api/client';
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
  setSelectedModel: (model: string) => void;
  refreshRuntime: (quiet?: boolean) => Promise<boolean>;
  addJob: (job: JobResource) => void;
  reloadService: () => Promise<boolean>;
  retryJobStreams: () => void;
  connectionRevision: number;
  ready: boolean;
  loading: boolean;
  connectionError: string | null;
  refreshError: string | null;
  jobStreamErrors: Record<string, string>;
  reloadingInstances: Record<string, number>;
  reloadModelContext: (instanceId: string, contextSize: number) => Promise<RuntimeStatus>;
}

const RuntimeContext = createContext<RuntimeContextValue | null>(null);

/** Keep shared runtime state synchronized with the connected server. */
export function RuntimeProvider({ children }: { children: ReactNode }) {
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
  const [reloadingInstances, setReloadingInstances] = useState<Record<string, number>>({});
  const reloads = useRef(new Set<string>());
  const mounted = useRef(false);
  const requestVersion = useRef(0);
  const initializationVersion = useRef(0);
  const selectedRef = useRef(selectedModel);
  selectedRef.current = selectedModel;

  const refreshRuntime = useCallback(async (quiet = true): Promise<boolean> => {
    const version = ++requestVersion.current;
    if (!quiet) setLoading(true);
    try {
      let [nextInstances, nextJobs] = await Promise.all([runtimeApi.runtimeInstances(), jobsApi.jobs(100)]);
      let instance = nextInstances.find(
        (item) => item.model === selectedRef.current && !['failed', 'unloading'].includes(item.state),
      );
      const [statusResult, voiceResult] = await Promise.allSettled([
        runtimeApi.runtimeStatus(instance?.id).catch(async (cause: unknown) => {
          if (!instance || !(cause instanceof ApiError) || cause.code !== 'runtime_instance_not_found') throw cause;
          // An unload can finish between listing instances and reading their status.
          nextInstances = await runtimeApi.runtimeInstances();
          instance = undefined;
          return runtimeApi.runtimeStatus();
        }),
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

  const reloadModelContext = useCallback(async (instanceId: string, contextSize: number) => {
    if (reloads.current.has(instanceId)) throw new Error('Model reload is already in progress');
    reloads.current.add(instanceId);
    setReloadingInstances((current) => ({ ...current, [instanceId]: contextSize }));
    try {
      const result = await runtimeApi.reloadRuntime(contextSize, instanceId);
      if (mounted.current) {
        setInstances((current) => current.map((item) => item.id === instanceId
          ? { ...item, context_size: result.max_context ?? contextSize }
          : item));
      }
      return result;
    } finally {
      await refreshRuntime(true);
      reloads.current.delete(instanceId);
      if (mounted.current) setReloadingInstances((current) => {
        const next = { ...current };
        delete next[instanceId];
        return next;
      });
    }
  }, [refreshRuntime]);

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
      const token = status ? await studioCredential() : getApiToken();
      if (!mounted.current || version !== initializationVersion.current) return false;
      setApiBaseUrl(status?.service_url ?? browserServiceUrl());
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

  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    let running = false;
    let timer: ReturnType<typeof setTimeout>;
    const syncJobs = async () => {
      if (running || disposed) return;
      running = true;
      clearTimeout(timer);
      try {
        const jobs = await jobsApi.jobs(100);
        if (disposed) return;
        const previous = useJobStore.getState().activeJobIds;
        useJobStore.getState().setJobs(jobs);
        retryJobStreams();
        const active = useJobStore.getState().activeJobIds;
        if (previous.slice().sort().join(',') !== active.slice().sort().join(',')) void refreshRuntime();
      } catch {
        if (!disposed) retryJobStreams();
      } finally {
        running = false;
        if (!disposed) timer = setTimeout(() => void syncJobs(), useJobStore.getState().activeJobIds.length ? 1000 : 5000);
      }
    };
    const resume = () => {
      if (document.visibilityState === 'hidden') return;
      useJobStore.getState().clearJobStreams();
      void refreshRuntime();
      void syncJobs();
    };
    timer = setTimeout(() => void syncJobs(), 1000);
    window.addEventListener('focus', resume);
    document.addEventListener('visibilitychange', resume);
    return () => {
      disposed = true;
      clearTimeout(timer);
      window.removeEventListener('focus', resume);
      document.removeEventListener('visibilitychange', resume);
    };
  }, [ready, connectionRevision, refreshRuntime, retryJobStreams]);

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
      reloadingInstances,
      reloadModelContext,
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
      reloadingInstances,
      reloadModelContext,
    ],
  );
  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

/** Access the shared runtime connection and model state. */
export function useRuntime(): RuntimeContextValue {
  const value = useContext(RuntimeContext);
  if (!value) throw new Error('RuntimeProvider is missing');
  return value;
}
