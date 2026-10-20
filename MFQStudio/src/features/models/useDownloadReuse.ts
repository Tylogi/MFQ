import { useEffect, useState } from 'react';
import type { HubDownloadReuseRequest } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';

export interface DownloadReuse {
  checking: boolean;
  reusableFiles: Record<string, number>;
}

export function useDownloadReuse(payload: HubDownloadReuseRequest | null, refreshKey: string): DownloadReuse {
  const serialized = payload ? JSON.stringify(payload) : '';
  const [focusRevision, setFocusRevision] = useState(0);
  const key = serialized ? JSON.stringify([serialized, refreshKey, focusRevision]) : '';
  const [result, setResult] = useState({ key: '', reusableFiles: {} as Record<string, number> });
  useEffect(() => {
    if (!serialized) return;
    const refresh = () => setFocusRevision(value => value + 1);
    window.addEventListener('focus', refresh);
    return () => window.removeEventListener('focus', refresh);
  }, [serialized]);
  useEffect(() => {
    if (!serialized) return;
    const controller = new AbortController();
    void modelsApi.downloadReuse(JSON.parse(serialized) as HubDownloadReuseRequest, controller.signal)
      .then(value => {
        if (!controller.signal.aborted) setResult({ key, reusableFiles: value.reusable_files });
      })
      .catch(() => {
        if (!controller.signal.aborted) setResult({ key, reusableFiles: {} });
      });
    return () => controller.abort();
  }, [key, serialized]);
  return { checking: !!key && result.key !== key,
    reusableFiles: result.key === key ? result.reusableFiles : {} };
}
