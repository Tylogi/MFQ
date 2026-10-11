import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact, ModelDirectoryList } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage, formatNumber } from '../../app/formatters';
import { isStudio, selectLocalModelDirectory } from '../../studio';
import { runtimeModelNames } from '../runtime/modelSelection';
import { STUDIO_PATHS, labPath } from '../../navigation';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';
import { readModelDirectory, saveModelDirectory } from './modelDirectoryPreference';

export function useModelCatalog() {
  const {
    runtime,
    models,
    instances,
    studio,
    setSelectedModel,
    refreshRuntime,
    ready,
    connectionRevision,
  } = useRuntime();
  const connectionScope = useConnectionScope();
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
  const [modelBrowserOpen, updateModelBrowserOpen] = useState(false);
  const [modelFolderPath, setModelFolderPath] = useState(readModelDirectory);
  const [modelBrowserError, setModelBrowserError] = useState('');
  const [modelBrowserLoading, setModelBrowserLoading] = useState(false);
  const directoryRequest = useRef(0);
  const [modelFilesMode, setModelFilesMode] = useState(false);
  const [modelDirectoryPath, setModelDirectoryPath] = useState(readModelDirectory);
  const modelBrowserTriggerRef = useRef<HTMLElement | null>(null);
  const canUseNativeModelPicker = isStudio() && studio?.config.mode !== 'remote';
  const availableModelNames = runtimeModelNames(models, instances);
  const openStudioPage = (_view: string, page: 'models' | 'quantization') =>
    navigate(labPath(page));
  const artifactRevision = jobs.map((job) => job.id + ':' + job.status).join(',');
  function setModelBrowserOpen(open: boolean) {
    if (!open) {
      directoryRequest.current += 1;
      if (modelBrowserLoading) { setBusy(false); setModelBrowserLoading(false); }
    }
    updateModelBrowserOpen(open);
  }
  function rememberModelDirectory(path: string) {
    saveModelDirectory(path);
    setModelFolderPath(path);
  }
  useEffect(() => {
    directoryRequest.current += 1;
    const path = readModelDirectory();
    setModelFolderPath(path);
    setModelDirectoryPath(path);
    setModelBrowser(null);
    setModelBrowserError('');
    setModelBrowserLoading(false);
    setArtifacts([]);
    updateModelBrowserOpen(false);
    setBusy(false);
  }, [connectionRevision]);
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    let polling = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function update() {
      if (disposed || polling) return;
      clearTimeout(timer);
      polling = true;
      try {
        if (!document.hidden) await refreshRuntime(true);
      } finally {
        polling = false;
        if (!disposed) timer = setTimeout(() => void update(), 5000);
      }
    }
    function visible() { if (!document.hidden) void update(); }
    document.addEventListener('visibilitychange', visible);
    void update();
    return () => { disposed = true; clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision, refreshRuntime]);
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
  }, [ready, connectionRevision, artifactRevision]);
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
  const filteredInstances = instances;
  const filteredArtifacts = useMemo(
    () =>
      artifacts.filter((item) =>
        item.name.toLowerCase().includes(modelFilter.trim().toLowerCase()),
      ),
    [artifacts, modelFilter],
  );
  async function loadArtifact(name: string) {
    const current = connectionScope();
    if (busy) return;
    setBusy(true);
    try {
      await modelsApi.loadModel(name, contextSize, 2048, {
        pin: loadPinned,
        idle_ttl_seconds: loadIdleTtl,
      });
      if (!current()) return;
      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      if (!current()) return;
      setSelectedModel(name);
    } catch (cause) {
      if (current()) reportError(cause);
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function finishModelRegistration(names: string[], current: () => boolean) {
    if (!current()) return;
    const nextArtifacts = await modelsApi.modelArtifacts(true);
    if (!current()) return;
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
        if (!current()) return;
      }
    }
    setModelBrowserOpen(false);
    navigate(STUDIO_PATHS.models);
    await refreshRuntime(false);
    if (!current()) return;
    if (registered.length === 1) setSelectedModel(registered[0].name);
  }

  async function openModelDirectory(directoryId?: string | null, path?: string | null) {
    const current = connectionScope();
    if (busy) return;
    const request = ++directoryRequest.current;
    setModelBrowserError('');
    setModelBrowser(null);
    if (path != null) setModelDirectoryPath(path);
    setModelBrowserOpen(true);
    setModelBrowserLoading(true);
    setBusy(true);
    try {
      const listing = await modelsApi.modelDirectories(directoryId, path);
      if (!current() || request !== directoryRequest.current) return;
      setModelBrowser(listing);
      setModelDirectoryPath(listing.current_path ?? '');
    } catch (cause) {
      if (current() && request === directoryRequest.current) setModelBrowserError(errorMessage(cause));
    } finally {
      if (current() && request === directoryRequest.current) { setBusy(false); setModelBrowserLoading(false); }
    }
  }

  async function jumpToModelDirectory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = modelDirectoryPath.trim();
    if (!path) return;
    await openModelDirectory(null, path);
  }

  async function chooseModelDirectory() {
    const current = connectionScope();
    if (busy) return;
    setModelFilesMode(false);
    modelBrowserTriggerRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!canUseNativeModelPicker) {
      await openModelDirectory(null, modelFolderPath);
      return;
    }
    setBusy(true);
    try {
      const selected = await selectLocalModelDirectory(modelFolderPath);
      if (!current()) return;
      if (selected) {
        rememberModelDirectory(selected.path);
        await finishModelRegistration(selected.names, current);
      }
    } catch (cause) {
      if (current()) reportError(cause);
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function openModelFiles(modelId: string) {
    const current = connectionScope();
    if (busy) return;
    modelBrowserTriggerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const request = ++directoryRequest.current;
    setBusy(true);
    try {
      const listing = await modelsApi.modelArtifactDirectory(modelId);
      if (!current() || request !== directoryRequest.current) return;
      setModelBrowser(listing);
      setModelBrowserError('');
      setModelDirectoryPath(listing.current_path ?? '');
      setModelFilesMode(true);
      setModelBrowserOpen(true);
    } catch (cause) {
      if (current() && request === directoryRequest.current) reportError(cause);
    } finally {
      if (current() && request === directoryRequest.current) setBusy(false);
    }
  }

  async function registerCurrentModelDirectory() {
    const current = connectionScope();
    if (busy || !modelBrowser?.current_id) return;
    setBusy(true);
    try {
      const registered = await modelsApi.registerModelDirectory(modelBrowser.current_id);
      if (!current()) return;
      if (modelBrowser.current_path) rememberModelDirectory(modelBrowser.current_path);
      await finishModelRegistration(registered.map((item) => item.name), current);
    } catch (cause) {
      if (current()) reportError(cause);
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function openCurrentDirectoryInFinder() {
    const current = connectionScope();
    if (busy || !modelBrowser?.current_id || !modelBrowser.can_open_in_finder) return;
    setBusy(true);
    try {
      await modelsApi.openModelDirectoryInFinder(modelBrowser.current_id);
    } catch (cause) {
      if (current()) reportError(cause);
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function unloadInstance(id: string) {
    const current = connectionScope();
    if (busy) return;
    setBusy(true);
    try {
      await modelsApi.unloadModel(id);
      if (!current()) return;
      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
    } catch (cause) {
      if (current()) reportError(cause);
    } finally {
      if (current()) setBusy(false);
    }
  }

  return {
    runtime,
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
    modelFolderPath,
    modelBrowserError,
    modelBrowserLoading,
    modelFilesMode,
    setModelBrowserOpen,
    modelDirectoryPath,
    setModelDirectoryPath,
    modelBrowserTriggerRef,
    openStudioPage,
    chooseModelDirectory,
    jumpToModelDirectory,
    openModelDirectory,
    openModelFiles,
    openCurrentDirectoryInFinder,
    registerCurrentModelDirectory,
    unloadInstance,
    loadArtifact,
  };
}
