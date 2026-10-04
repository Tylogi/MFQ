/** Expose the current chat mode to other domains without exposing session-store internals. */
import type { SessionMode } from '../../../shared/api/types';
import { useConversationSelector } from '../state/conversationStore';

/**
* Read the active session mode; use text mode for settings defaults when chat is not open or no session is active.
 *
* @returns The loaded active-session mode, or the default text mode
 */
export function useActiveSessionMode(): SessionMode {
  return useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId)?.mode ?? 'text',
  );
}
