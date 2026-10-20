import { useEffect, useState } from 'react';
import { getApiBaseUrl, getApiToken } from '../../shared/api/client';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { ServingStatistics } from '../../shared/api/contracts/runtime';
import { errorMessage } from '../../app/formatters';

export function useServingStatistics(scope: 'session' | 'alltime', revision: number) {
  const [data, setData] = useState<ServingStatistics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let disposed = false;
    let running = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    const endpoint = getApiBaseUrl();
    const credential = getApiToken();
    const current = () => !disposed && endpoint === getApiBaseUrl() && credential === getApiToken();
    setData(null);
    setError(null);
    async function poll() {
      if (!current() || running || document.hidden) return;
      running = true;
      clearTimeout(timer);
      controller = new AbortController();
      try {
        const next = await runtimeApi.servingStatistics(scope, controller.signal);
        if (current() && !controller.signal.aborted) { setData(next); setError(null); }
      } catch (cause) {
        if (current() && !controller.signal.aborted) setError(errorMessage(cause));
      } finally {
        running = false;
        if (current() && !document.hidden) timer = setTimeout(() => void poll(), 500);
      }
    }
    function visible() {
      clearTimeout(timer);
      if (document.hidden) controller?.abort();
      else void poll();
    }
    void poll();
    document.addEventListener('visibilitychange', visible);
    window.addEventListener('focus', visible);
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener('visibilitychange', visible);
      window.removeEventListener('focus', visible);
    };
  }, [scope, revision]);
  return { data: data?.scope === scope ? data : null, error };
}
