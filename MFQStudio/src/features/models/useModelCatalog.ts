/** Manage artifact refresh, load policies, and directory registration lifecycle for the model catalog. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { ModelArtifact, ModelDirectoryList } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage, formatNumber } from '../../app/formatters';
import { isStudio, selectLocalModelDirectory } from '../../studio';
import { runtimeModelNames } from '../runtime/modelSelection';
import { STUDIO_PATHS, labPath } from '../../navigation';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';

/** Encapsulate model-catalog workflows for the models page; state is released when the page unmounts. */
export function useModelCatalog() {
  const {
    runtime,
    models,
    instances,
    studio,
    setSelectedModel,
    refreshRuntime,
    ready,
  } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const { contextSize } = useSettings();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [busy, setBusy] = useState(false);
  const observedModelJobIds = useRef(new Set<string>());
  const submittingUnloads = useRef(new Set<string>());
  const [pendingUnloads, setPendingUnloads] = useState<Record<string, string | null>>({});
  const unloadingInstanceIds = useMemo(() => new Set([
    ...Object.keys(pendingUnloads),
    ...instances.filter((instance) => instance.state === 'unloading').map((instance) => instance.id),
    ...jobs.filter((job) => job.kind === 'model.unload'
      && ['queued', 'running', 'cancelling'].includes(job.status))
      .map((job) => String(job.payload.instance_id)),
  ]), [pendingUnloads, instances, jobs]);
  const reportedModelJobIds = useRef(new Set<string>());
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
      if (!['model.load', 'model.unload'].includes(job.kind)) continue;
      if (['queued', 'running', 'cancelling'].includes(job.status)) {
        observedModelJobIds.current.add(job.id);
        continue;
      }
      const wasActive = observedModelJobIds.current.delete(job.id);
      if (!wasActive || reportedModelJobIds.current.has(job.id)) continue;
      reportedModelJobIds.current.add(job.id);
      if (job.kind === 'model.unload') {
        const id = String(job.payload.instance_id);
        submittingUnloads.current.delete(id);
        setPendingUnloads((current) => {
          if (!(id in current)) return current;
          const next = { ...current };
          delete next[id];
          return next;
        });
        void refreshRuntime(false);
        if (job.status === 'succeeded') toast.success(localized('models:useModelCatalog.modelUnloaded'));
      }
      if (job.kind === 'model.load' && job.status === 'succeeded') {
        toast.success(localized('models:useModelCatalog.modelLoaded'));
      }
      if (job.status === 'failed') {
        if (job.error?.message) {
          toast.error(`${job.error.code}: ${job.error.message}`);
        } else {
          // State events may arrive before the full job error is fetched.
          void jobsApi.getJob(job.id).then((finished) => {
            if (finished.error) toast.error(`${finished.error.code}: ${finished.error.message}`);
            else toast.error(localized('models:useModelCatalog.modelOperationFailedCheckTheTaskRecord'));
          }).catch(() => toast.error(localized('models:useModelCatalog.modelOperationFailedDetailsAreUnavailable')));
        }
      } else if (job.kind === 'model.unload' && job.status !== 'succeeded') {
        toast.error(localized('models:useModelCatalog.modelUnloadWasCancelledOrInterrupted'));
      }
    }
  }, [jobs, refreshRuntime, t]);
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
  /** Run a model catalog operation and display any error on the current page. */
  async function loadArtifact(name: string) {
    if (busy) return;
    setBusy(true);
    try {
      const operation = await modelsApi.loadModel(name, contextSize, 2048, {
        pin: loadPinned,
        idle_ttl_seconds: loadIdleTtl,
      });
      observedModelJobIds.current.add(operation.operation_id);

      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      setSelectedModel(name);
    } catch (cause) {
      reportError(cause);
    } finally {
      setBusy(false);
    }
  }

  /** Run a model catalog operation and display any error on the current page. */
  async function finishModelRegistration(names: string[]) {
    const nextArtifacts = await modelsApi.modelArtifacts(true);
    setArtifacts(nextArtifacts);
    const registered = nextArtifacts.filter((item) => names.includes(item.name));
    if (!registered.length) {
      throw new Error(
        t('models:useModelCatalog.modelsFromTheSelectedFolderWereNotRegisteredInTheCatalog'),
      );
    }
    if (registered.length === 1) {
      const artifact = registered[0];
      if (!artifact.loadable) {
        throw new Error(
          artifact.error ||
            t('models:useModelCatalog.theSelectedModelIsIncompleteOrCannotBeLoaded'),
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

  /** Run a model catalog operation and display any error on the current page. */
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

  /** Run a model catalog operation and display any error on the current page. */
  async function jumpToModelDirectory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = modelDirectoryPath.trim();
    if (!path) return;
    await openModelDirectory(null, path);
  }

  /** Run a model catalog operation and display any error on the current page. */
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

  /** Run a model catalog operation and display any error on the current page. */
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

  /** Track an unload from submission through its terminal job state and suppress duplicate clicks. */
  async function unloadInstance(id: string) {
    if (busy || submittingUnloads.current.has(id) || unloadingInstanceIds.has(id)) return;
    submittingUnloads.current.add(id);
    setPendingUnloads((current) => ({ ...current, [id]: null }));
    setBusy(true);
    let accepted = false;
    try {
      const operation = await modelsApi.unloadModel(id);
      accepted = true;
      observedModelJobIds.current.add(operation.operation_id);
      setPendingUnloads((current) => ({ ...current, [id]: operation.operation_id }));
      navigate(STUDIO_PATHS.models);
      // Fetch even an already-finished job so fast failures cannot escape the active-job watcher.
      try {
        useJobStore.getState().addJob(await jobsApi.getJob(operation.operation_id));
      } catch {
        // The shared job refresh will recover this accepted operation after transient read failures.
      }
      await refreshRuntime(false);
    } catch (cause) {
      if (!accepted) {
        submittingUnloads.current.delete(id);
        setPendingUnloads((current) => {
          const next = { ...current };
          delete next[id];
          return next;
        });
      }
      reportError(cause);
    } finally {
      setBusy(false);
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
    setModelBrowserOpen,
    modelDirectoryPath,
    setModelDirectoryPath,
    modelBrowserTriggerRef,
    openStudioPage,
    chooseModelDirectory,
    jumpToModelDirectory,
    openModelDirectory,
    registerCurrentModelDirectory,
    unloadInstance,
    unloadingInstanceIds,
    loadArtifact,
  };
}
