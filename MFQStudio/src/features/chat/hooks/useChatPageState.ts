/** Local interaction state for the chat route; generation and session lifecycles remain cross-route. */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useChatAutoScroll } from './useChatAutoScroll';
import type { EditDraft } from '../SavedMessageList';

/** Manage interactions that exist only on the chat page, such as the sidebar, scrolling, and edit drafts. */
export function useChatPageState() {
  const chat = useChat();
  const navigate = useNavigate();
  const activeId = useConversationSelector((state) => state.activeId);
  const active = useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId) ?? null,
  );
  const [editDraft, setEditDraft] = useState<EditDraft | null>(null);
  const [chatSessionsOpen, setChatSessionsOpen] = useState(
    () => window.matchMedia('(min-width: 681px)').matches,
  );
  const scroll = useChatAutoScroll(activeId, true);

  useEffect(() => setEditDraft(null), [activeId]);
  useEffect(() => {
    const viewport = window.matchMedia('(max-width: 680px)');
    const close = () => {
      if (viewport.matches) setChatSessionsOpen(false);
    };
    viewport.addEventListener('change', close);
    return () => viewport.removeEventListener('change', close);
  }, []);

  /** Save the current edit and exit editing only if the draft has not changed further. */
  async function saveCurrentEdit(message: Parameters<typeof chat.messageActions.saveEdit>[0]) {
    const draft = editDraft;
    if (!draft || draft.messageId !== message.id) return;
    await chat.messageActions.saveEdit(message, draft.text, () => {
      setEditDraft((current) =>
        current?.messageId === draft.messageId && current.text === draft.text ? null : current,
      );
    });
  }

  /** Switch historical sessions and collapse the sidebar on mobile. */
  function selectSession(id: string) {
    if (chat.busy || chat.recoveryNeeded) return;
    chat.conversation.selectSession(id);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }

  /** Create a session and collapse the sidebar on mobile. */
  async function createSession() {
    if (chat.busy || chat.recoveryNeeded) return;
    await chat.conversation.createSession(active?.mode);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }

  /** Navigate to the model catalog to load a model. */
  function chooseModelDirectory() {
    navigate('/models');
  }

  /** Apply an explicit chat model change through the session lifecycle. */
  function selectModel(value: string) {
    if (chat.busy || chat.recoveryNeeded) return;
    if (chat.inference.availableModelNames.includes(value)) chat.conversation.changeSessionModel(value);
  }

  return {
    chat,
    activeId,
    active,
    editDraft,
    setEditDraft,
    saveCurrentEdit,
    chatSessionsOpen,
    setChatSessionsOpen,
    scroll,
    selectSession,
    createSession,
    chooseModelDirectory,
    selectModel,
  };
}

export type ChatPageState = ReturnType<typeof useChatPageState>;
