/** 模型目录控制器负责资产刷新、加载策略和目录注册生命周期。 */
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact, ModelDirectoryList } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage, formatNumber } from '../../app/formatters';
import { isStudio, selectLocalModelDirectory } from '../../studio';
import { runtimeModelNames } from '../runtime/modelSelection';
import { STUDIO_PATHS, labPath } from '../../navigation';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';

/** 为模型页封装模型目录工作流；状态随页面卸载释放。 */
export function useModelCatalog() {
  const {
    runtime,
    models,
    instances,
    studio,
    selectedModel: model,
    setSelectedModel,
    refreshRuntime,
    ready,
  } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const { tr, contextSize } = useSettings();
  const navigate = useNavigate();
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [busy, setBusy] = useState(false);
  const observedActiveLoadJobIds = useRef(new Set<string>());
  const reportedFailedJobIds = useRef(new Set<string>());
  const reportError = (cause: unknown) => {
    toast.error(errorMessage(cause));
  };
  const [modelFilter, setModelFilter] = useState('');
  const [loadPinned, setLoadPinned] = useState(false);
  const [loadIdleTtl, setLoadIdleTtl] = useState<number | null>(null);
  const [modelBrowser, setModelBrowser] = useState<ModelDirectoryList | null>(null);
  const [modelBrowserOpen, setModelBrowserOpen] = useState(false);
  const [modelDirectoryPath, setModelDirectoryPath] = useState('');
  const modelBrowserTriggerRef = useRef<HTMLElement | null>(null);
  const canUseNativeModelPicker = isStudio() && studio?.config.mode !== 'remote';
  const availableModelNames = runtimeModelNames(models, instances);
  const selectModel = setSelectedModel;
  const openStudioPage = (_view: string, page: 'models' | 'quantization') =>
    navigate(labPath(page));
  const artifactRevision = jobs.map((job) => job.id + ':' + job.status).join(',');
  useEffect(() => {
    if (!ready) return;
    let active = true;
    void modelsApi
      .modelArtifacts()
      .then((items) => {
        if (active) setArtifacts(items);
      })
      .catch((cause) => {
        if (active) reportError(cause);
      });
    return () => {
      active = false;
    };
  }, [ready, artifactRevision]);
  useEffect(() => {
    for (const job of jobs) {
      if (job.kind !== 'model.load') continue;
      if (job.status === 'queued' || job.status === 'running' || job.status === 'cancelling') {
        observedActiveLoadJobIds.current.add(job.id);
        continue;
      }
      const wasActive = observedActiveLoadJobIds.current.delete(job.id);
      if (wasActive && job.status === 'failed' && job.error?.message && !reportedFailedJobIds.current.has(job.id)) {
        reportedFailedJobIds.current.add(job.id);
        toast.error(`${job.error.code}: ${job.error.message}`);
      }
    }
  }, [jobs]);
  const filteredInstances = useMemo(
    () =>
      instances.filter((item) =>
        item.model.toLowerCase().includes(modelFilter.trim().toLowerCase()),
      ),
    [instances, modelFilter],
  );
  const filteredArtifacts = useMemo(
    () =>
      artifacts.filter((item) =>
        item.name.toLowerCase().includes(modelFilter.trim().toLowerCase()),
      ),
    [artifacts, modelFilter],
  );
  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function loadArtifact(name: string) {
    if (busy) return;
    setBusy(true);
    try {
      await modelsApi.loadModel(name, contextSize, 2048, {
        pin: loadPinned,
        idle_ttl_seconds: loadIdleTtl,
      });

      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      setSelectedModel(name);
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function finishModelRegistration(names: string[]) {
    const nextArtifacts = await modelsApi.modelArtifacts(true);
    setArtifacts(nextArtifacts);
    const registered = nextArtifacts.filter((item) => names.includes(item.name));
    if (!registered.length) {
      throw new Error(
        tr(
          '所选目录中的模型没有出现在模型目录中。',
          'Models from the selected folder were not registered in the catalog.',
        ),
      );
    }
    if (registered.length === 1) {
      const artifact = registered[0];
      if (!artifact.loadable) {
        throw new Error(
          artifact.error ||
            tr(
              '所选模型不完整或无法加载。',
              'The selected model is incomplete or cannot be loaded.',
            ),
        );
      }
      const loaded =
        instances.some((item) => item.model === artifact.name && item.state !== 'failed') ||
        runtime?.model === artifact.name;
      if (!loaded) {
        await modelsApi.loadModel(artifact.name, contextSize, 2048, {
          pin: loadPinned,
          idle_ttl_seconds: loadIdleTtl,
        });
      }
    }
    setModelBrowserOpen(false);
    navigate(STUDIO_PATHS.models);
    await refreshRuntime(false);
    if (registered.length === 1) setSelectedModel(registered[0].name);
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function openModelDirectory(directoryId?: string | null, path?: string | null) {
    if (busy) return;
    setBusy(true);
    try {
      const listing = await modelsApi.modelDirectories(directoryId, path);
      setModelBrowser(listing);
      setModelDirectoryPath(listing.current_path ?? '');
      setModelBrowserOpen(true);
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function jumpToModelDirectory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = modelDirectoryPath.trim();
    if (!path) return;
    await openModelDirectory(null, path);
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function chooseModelDirectory() {
    if (busy) return;
    modelBrowserTriggerRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!canUseNativeModelPicker) {
      await openModelDirectory();
      return;
    }
    setBusy(true);
    try {
      const names = await selectLocalModelDirectory();
      if (names) await finishModelRegistration(names);
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function registerCurrentModelDirectory() {
    if (busy || !modelBrowser?.current_id) return;
    setBusy(true);
    try {
      const registered = await modelsApi.registerModelDirectory(modelBrowser.current_id);
      await finishModelRegistration(registered.map((item) => item.name));
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  /** 执行模型资源操作，并将错误展示在当前页面。 */
  async function unloadInstance(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      await modelsApi.unloadModel(id);

      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  return {
    runtime,
    model,
    artifacts,
    busy,
    ready,
    instances,
    availableModelNames,
    modelFilter,
    setModelFilter,
    filteredInstances,
    filteredArtifacts,
    loadPinned,
    setLoadPinned,
    loadIdleTtl,
    setLoadIdleTtl,
    modelBrowser,
    modelBrowserOpen,
    setModelBrowserOpen,
    modelDirectoryPath,
    setModelDirectoryPath,
    modelBrowserTriggerRef,
    selectModel,
    openStudioPage,
    chooseModelDirectory,
    jumpToModelDirectory,
    openModelDirectory,
    registerCurrentModelDirectory,
    unloadInstance,
    loadArtifact,
  };
}
