/** 聊天路由的局部交互状态，不接管跨路由的生成与会话生命周期。 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useChatAutoScroll } from './useChatAutoScroll';
import type { EditDraft } from '../SavedMessageList';

/** 组织侧栏、滚动和编辑草稿等仅在聊天页存在的交互。 */
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

  /** 保存当前编辑，并仅在草稿未被继续修改时退出编辑。 */
  async function saveCurrentEdit(message: Parameters<typeof chat.messageActions.saveEdit>[0]) {
    const draft = editDraft;
    if (!draft || draft.messageId !== message.id) return;
    await chat.messageActions.saveEdit(message, draft.text, () => {
      setEditDraft((current) =>
        current?.messageId === draft.messageId && current.text === draft.text ? null : current,
      );
    });
  }

  /** 切换历史会话，移动端同时折叠侧栏。 */
  function selectSession(id: string) {
    if (chat.busy) return;
    chat.conversation.selectSession(id);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }

  /** 新建会话，移动端同时折叠侧栏。 */
  async function createSession() {
    await chat.conversation.createSession(active?.mode);
    if (window.matchMedia('(max-width: 680px)').matches) setChatSessionsOpen(false);
  }

  /** 跳转到模型目录以加载模型。 */
  function chooseModelDirectory() {
    navigate('/models');
  }

  /** 生成期间禁止改变选中模型。 */
  function selectModel(value: string) {
    if (!chat.busy) chat.inference.setSelectedModel(value);
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
