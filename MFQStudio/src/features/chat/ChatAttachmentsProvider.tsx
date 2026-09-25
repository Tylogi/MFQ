/** 跨路由保管待发送附件，隔离预览列表的更新与聊天生成生命周期。 */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
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
const ErrorContext = createContext<string | null>(null);

/** 维持附件及预览 URL 的生命周期；切会话、切服务或卸载时释放旧预览。 */
export function ChatAttachmentsProvider({ children }: { children: ReactNode }) {
  const activeId = useConversationSelector((state) => state.activeId);
  const { connectionRevision } = useRuntime();
  const [error, setError] = useState<string | null>(null);
  const {
    attachments,
    getAttachments,
    selectAttachments,
    removeAttachment,
    clearAttachments,
    uploadAttachments,
  } = useChatAttachments(activeId, connectionRevision, setError);
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

/** 发送流程读取稳定附件操作，无需订阅附件列表。 */
export function useChatAttachmentActions(): AttachmentActions {
  const actions = useContext(ActionsContext);
  if (!actions) throw new Error('ChatAttachmentsProvider is missing');
  return actions;
}

/** 仅输入框订阅当前待发送的附件列表。 */
export function useChatAttachmentList(): PendingAttachment[] {
  const attachments = useContext(AttachmentsContext);
  if (!attachments) throw new Error('ChatAttachmentsProvider is missing');
  return attachments;
}

/** 聊天错误区域读取附件校验错误，不订阅附件列表。 */
export function useChatAttachmentError(): string | null {
  return useContext(ErrorContext);
}
