import { useCallback, useEffect, useRef } from 'react';
import { getApiBaseUrl } from '../shared/api/client';
import { useRuntime } from './RuntimeProvider';

export function useConnectionScope() {
  const { connectionRevision } = useRuntime();
  const current = useRef({ active: true, revision: connectionRevision });
  current.current.revision = connectionRevision;
  useEffect(() => {
    current.current.active = true;
    return () => { current.current.active = false; };
  }, []);
  return useCallback(() => {
    const revision = current.current.revision;
    const endpoint = getApiBaseUrl();
    return () => current.current.active && current.current.revision === revision && getApiBaseUrl() === endpoint;
  }, []);
}
