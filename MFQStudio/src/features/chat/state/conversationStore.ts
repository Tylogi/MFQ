/** Preserve sessions and history across routes, using revision tokens to isolate stale async writes. */
import { create } from 'zustand';
import type { Dispatch, SetStateAction } from 'react';
import type { Message, ResponseResource, Session } from '../../../shared/api/types';

type Responses = Record<string, ResponseResource>;
/** Session business state and semantic actions for the chat domain. */
export interface ConversationState {
  sessions: Session[];
  activeId: string | null;
  messages: Message[];
  responses: Responses;
  historyLoadedId: string | null;
  epoch: number;
/** Clear connection-related data and invalidate previously issued asynchronous requests. */
  reset: () => void;
/** Support functional session-list updates for existing callers. */
  setSessions: Dispatch<SetStateAction<Session[]>>;
/** Clear the previous session history when switching sessions. */
  setActiveId: Dispatch<SetStateAction<string | null>>;
/** Support functional message updates for existing callers. */
  setMessages: Dispatch<SetStateAction<Message[]>>;
/** Support functional response updates for existing callers. */
  setResponses: Dispatch<SetStateAction<Responses>>;
/** Apply the first session only while the list request still belongs to the current connection. */
  loadSessions: (epoch: number, sessions: Session[]) => boolean;
/** Clear stale history for the current session while awaiting the new request. */
  beginHistory: (id: string) => void;
/** Write history only if both the connection and session remain unchanged. */
  applyHistory: (
    epoch: number,
    id: string,
    messages: Message[],
    responses: ResponseResource[],
  ) => void;
/** Sync the session revision returned by generation, replacing message history only when the target session is active. */
  applySynchronized: (session: Session, messages: Message[], responses: ResponseResource[]) => void;
}
/** Index server responses by their output-message IDs. */
function indexResponses(responses: ResponseResource[]): Responses {
  return Object.fromEntries(
    responses
      .filter((response) => response.output_message_id)
      .map((response) => [response.output_message_id!, response]),
  );
}
/** Shared chat-session store; components can subscribe to one field to avoid unrelated updates. */
export const useConversationStore = create<ConversationState>()((set) => ({
  sessions: [],
  activeId: null,
  messages: [],
  responses: {},
  historyLoadedId: null,
  epoch: 0,
  reset: () =>
    set((state) => ({
      sessions: [],
      activeId: null,
      messages: [],
      responses: {},
      historyLoadedId: null,
      epoch: state.epoch + 1,
    })),
  setSessions: (value) =>
    set((state) => ({
      sessions: typeof value === 'function' ? value(state.sessions) : value,
    })),
  setActiveId: (value) =>
    set((state) => {
      const activeId = typeof value === 'function' ? value(state.activeId) : value;
      return activeId === state.activeId
        ? state
        : {
            activeId,
            messages: [],
            responses: {},
            historyLoadedId: null,
          };
    }),
  setMessages: (value) =>
    set((state) => ({
      messages: typeof value === 'function' ? value(state.messages) : value,
    })),
  setResponses: (value) =>
    set((state) => ({
      responses: typeof value === 'function' ? value(state.responses) : value,
    })),
  loadSessions: (epoch, sessions) => {
    if (useConversationStore.getState().epoch !== epoch) return false;
    set({
      sessions,
      activeId: sessions[0]?.id ?? null,
      messages: [],
      responses: {},
      historyLoadedId: null,
    });
    return true;
  },
  beginHistory: (id) =>
    set((state) =>
      state.activeId === id ? { messages: [], responses: {}, historyLoadedId: null } : state,
    ),
  applyHistory: (epoch, id, messages, responses) =>
    set((state) =>
      state.epoch === epoch && state.activeId === id
        ? { messages, responses: indexResponses(responses), historyLoadedId: id }
        : state,
    ),
  applySynchronized: (session, messages, responses) =>
    set((state) => ({
      sessions: state.sessions.map((item) => (item.id === session.id ? session : item)),
      ...(state.activeId === session.id ? { messages, responses: indexResponses(responses) } : {}),
    })),
}));
/** Subscribe to a single session field by selector for pages incrementally removing reactive Context propagation. */
export function useConversationSelector<T>(selector: (state: ConversationState) => T): T {
  return useConversationStore(selector);
}
/** Stable actions for parent business callbacks to invoke without subscribing to session entities. */
export const conversationActions = {
  reset: useConversationStore.getState().reset,
  applySynchronized: useConversationStore.getState().applySynchronized,
  applySync: useConversationStore.getState().applySynchronized,
  setSessions: useConversationStore.getState().setSessions,
  setActiveId: useConversationStore.getState().setActiveId,
  setMessages: useConversationStore.getState().setMessages,
  setResponses: useConversationStore.getState().setResponses,
};
