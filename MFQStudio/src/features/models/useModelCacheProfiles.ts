import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelCacheProfile } from '../../shared/api/types';
import { useSettings } from '../settings/SettingsProvider';

export function useModelCacheProfiles(models: string[], enabled = true) {
  const { ready, connectionRevision } = useRuntime();
  const { tr } = useSettings();
  const names = JSON.stringify([...new Set(models)].sort());
  const [cacheProfiles, setCacheProfiles] = useState<Record<string, ModelCacheProfile | null>>({});
  const [cacheErrors, setCacheErrors] = useState<Record<string, string>>({});
  useEffect(() => {
    setCacheProfiles({}); setCacheErrors({});
    if (!ready || !enabled || names === '[]') return;
    const controller = new AbortController();
    void modelsApi.modelArtifacts().then(async (artifacts) => {
      if (controller.signal.aborted) return;
      await Promise.allSettled((JSON.parse(names) as string[]).map(async (name) => {
        try {
          const artifact = artifacts.find((candidate) => candidate.name === name);
          if (!artifact) throw new Error(tr('模型缓存结构信息不可用', 'Model cache metadata is unavailable'));
          const profile = await modelsApi.modelCacheProfile(artifact.id, controller.signal);
          if (!controller.signal.aborted) setCacheProfiles((current) => ({ ...current, [name]: profile }));
        } catch (cause) {
          if (!controller.signal.aborted) setCacheErrors((current) => ({ ...current, [name]: errorMessage(cause) }));
        }
      }));
    }).catch((cause) => {
      if (!controller.signal.aborted) setCacheErrors(Object.fromEntries((JSON.parse(names) as string[]).map((name) => [name, errorMessage(cause)])));
    });
    return () => controller.abort();
  }, [ready, names, enabled, connectionRevision]);
  return { cacheProfiles, cacheErrors };
}
