/** 将生成控制器接入 React，应用层只订阅阶段，消息层单独订阅文本。 */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { GenerationController, type GenerationCallbacks } from '../state/generationController';

/** 复用单个控制器并在卸载时取消异步工作；回调始终使用最近一次渲染的实现。 */
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
