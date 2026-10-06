/** Isolate input drafts by session, keep them in memory, and prevent typing from rerendering the app shell. */
import { create } from 'zustand';

interface DraftState {
  drafts: Record<string, string>;
/** Update a session draft; remove the entry when empty and do not persist potentially private input. */
  setDraft: (sessionId: string, text: string) => void;
}
/** Subscribe to a session’s draft string only in the input component, preserving unsent text across page changes. */
export const useDraftStore = create<DraftState>()((set) => ({
  drafts: {},
  setDraft: (sessionId, text) =>
    set((state) => {
      if ((state.drafts[sessionId] ?? '') === text) return state;
      const drafts = { ...state.drafts };
      if (text) drafts[sessionId] = text;
      else delete drafts[sessionId];
      return { drafts };
    }),
}));
