import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeConcurrencyPolicy } from '../../shared/api/types';
import { errorMessage } from '../../app/formatters';

/** Fetch once per connection and discard responses from a previous server. */
export function useConcurrencyPolicy() {
  const { connectionRevision } = useRuntime();
  const [state, setState] = useState<{ revision: number; policy: RuntimeConcurrencyPolicy | null; error: string }>();
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    void runtimeApi.concurrencyPolicy(controller.signal).then(policy => {
      if (active) setState({ revision: connectionRevision, policy, error: '' });
    }).catch(cause => {
      if (active) setState({ revision: connectionRevision, policy: null, error: errorMessage(cause) });
    });
    return () => { active = false; controller.abort(); };
  }, [connectionRevision]);
  return state && state.revision === connectionRevision ? state : { policy: null, error: '' };
}
