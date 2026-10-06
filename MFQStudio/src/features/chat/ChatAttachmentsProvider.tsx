/** Preserve pending attachments across routes, isolating preview-list updates from chat-generation lifecycle. */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import type { PendingAttachment } from './attachments';
import { useChatAttachments } from './hooks/useChatAttachments';
import { useConversationSelector } from './state/conversationStore';

type AttachmentActions = Pick<
  ReturnType<typeof useChatAttachments>,
  | 'getAttachments'
  | 'selectAttachments'
  | 'removeAttachment'
  | 'clearAttachments'
  | 'uploadAttachments'
>;

const ActionsContext = createContext<AttachmentActions | null>(null);
const AttachmentsContext = createContext<PendingAttachment[] | null>(null);
/** Preserve event identity so repeating a dismissed validation error displays it again. */
interface AttachmentError {
  message: string;
}
const ErrorContext = createContext<AttachmentError | null>(null);
/** Maintain attachment and preview-URL lifecycles, releasing stale previews on session or service changes and unmount. */
export function ChatAttachmentsProvider({ children }: { children: ReactNode }) {
  const activeId = useConversationSelector((state) => state.activeId);
  const { connectionRevision } = useRuntime();
  const { tr } = useSettings();
  const [error, setError] = useState<AttachmentError | null>(null);
  const reportError = useCallback((message: string | null) => {
    setError(message ? { message } : null);
  }, []);
  const {
    attachments,
    getAttachments,
    selectAttachments,
    removeAttachment,
    clearAttachments,
    uploadAttachments,
  } = useChatAttachments(activeId, connectionRevision, reportError, tr);
  useEffect(() => setError(null), [activeId, connectionRevision]);
  const actions = useMemo(
    () => ({
      getAttachments,
      selectAttachments,
      removeAttachment,
      clearAttachments,
      uploadAttachments,
    }),
    [getAttachments, selectAttachments, removeAttachment, clearAttachments, uploadAttachments],
  );

  return (
    <ActionsContext.Provider value={actions}>
      <AttachmentsContext.Provider value={attachments}>
        <ErrorContext.Provider value={error}>{children}</ErrorContext.Provider>
      </AttachmentsContext.Provider>
    </ActionsContext.Provider>
  );
}
/** Let send flows access stable attachment actions without subscribing to the attachment list. */
export function useChatAttachmentActions(): AttachmentActions {
  const actions = useContext(ActionsContext);
  if (!actions) throw new Error('ChatAttachmentsProvider is missing');
  return actions;
}
/** Subscribe to the pending attachment list only in the input area. */
export function useChatAttachmentList(): PendingAttachment[] {
  const attachments = useContext(AttachmentsContext);
  if (!attachments) throw new Error('ChatAttachmentsProvider is missing');
  return attachments;
}
/** Read attachment-validation errors in the chat error area without subscribing to the attachment list. */
export function useChatAttachmentError(): AttachmentError | null {
  return useContext(ErrorContext);
}
