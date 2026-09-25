/** 隔离高频语音音量更新，仅通知直接订阅音量的组件。 */
import { useSyncExternalStore } from 'react';

let level = 0;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** 获取当前音量快照，供不订阅的兼容调用方读取。 */
export function getVoiceLevel(): number {
  return level;
}

/** 仅在音量变化时通知订阅者，不更新语音会话的 React 状态。 */
export function setVoiceLevel(nextLevel: number): void {
  if (Object.is(level, nextLevel)) return;
  level = nextLevel;
  listeners.forEach((listener) => listener());
}

/** 清除当前控制器留下的音量读数。 */
export function resetVoiceLevel(): void {
  setVoiceLevel(0);
}

/** 精准订阅语音音量；ChatToolbar 可直接调用以更新音量指示器。 */
export function useVoiceLevel(): number {
  return useSyncExternalStore(subscribe, getVoiceLevel, () => 0);
}
