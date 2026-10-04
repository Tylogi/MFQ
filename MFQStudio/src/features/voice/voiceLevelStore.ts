/** Isolate high-frequency voice-level updates and notify only components subscribed to the level. */
import { useSyncExternalStore } from 'react';

let level = 0;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Get the current volume snapshot for compatible callers that do not subscribe. */
export function getVoiceLevel(): number {
  return level;
}

/** Notify subscribers only when the volume changes, without updating React voice-session state. */
export function setVoiceLevel(nextLevel: number): void {
  if (Object.is(level, nextLevel)) return;
  level = nextLevel;
  listeners.forEach((listener) => listener());
}

/** Clear the volume reading left by the current controller. */
export function resetVoiceLevel(): void {
  setVoiceLevel(0);
}

/** Subscribe specifically to voice volume; ChatToolbar can call this directly to update its volume indicator. */
export function useVoiceLevel(): number {
  return useSyncExternalStore(subscribe, getVoiceLevel, () => 0);
}
