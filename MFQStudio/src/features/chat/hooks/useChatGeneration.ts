/** Connect the generation controller to React; the app subscribes only to phases while messages subscribe to text separately. */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { GenerationController, type GenerationCallbacks } from '../state/generationController';
/** Reuse a single controller and cancel async work on unmount; callbacks always use the latest rendered implementation. */
export function useChatGeneration(callbacks: GenerationCallbacks) {
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;
  const [controller] = useState(
    () =>
      new GenerationController({
        onSynchronized: (value) => callbacksRef.current.onSynchronized(value),
        onSessionState: (sessionId, state, revision) =>
          callbacksRef.current.onSessionState(sessionId, state, revision),
      }),
  );
  const phase = useSyncExternalStore(controller.subscribe, controller.getPhase);
  useEffect(() => () => controller.reset(), [controller]);
  return { controller, phase };
}
