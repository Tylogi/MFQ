/** Render chat sessions and guard navigation while generation recovery is pending. */
import type { UIEvent } from 'react';
import { useSettings } from '../../settings/SettingsProvider';
import { Icon } from '../../../app/display';
import { useConversationSelector } from '../state/conversationStore';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ModelVendorMark } from '../../../app/ModelVendorMark';

/** Display session selection, creation, and deletion controls. */
export function ChatSessionSidebar({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
  const sessions = useConversationSelector((state) => state.sessions);
  const { activeId, chatSessionsOpen, selectSession, createSession, chat } = page;
  const { busy, recoveryNeeded, conversation, deleteConversation } = chat;
  /** Prefetch older chats near the bottom, matching runtime history's scroll threshold. */
  function onScroll(event: UIEvent<HTMLDivElement>) {
    const node = event.currentTarget;
    if (conversation.hasMoreSessions && !conversation.loadingSessions && !busy && !recoveryNeeded
      && node.scrollHeight - node.scrollTop - node.clientHeight < 100) {
      void conversation.loadMoreSessions();
    }
  }
  return (
    <aside
      className={'chat-session-sidebar' + (chatSessionsOpen ? ' open' : '')}
      aria-label={tr('会话列表', 'Conversations')}
    >
      <div className="chat-session-sidebar-header">
        <strong>{tr('对话', 'Chats')}</strong>
        <button
          aria-label={tr('新建会话', 'New chat')}
          className="chat-icon-button"
          disabled={busy || recoveryNeeded || !conversation.modelAvailable}
          onClick={() => void createSession()}
          title={tr('新建会话', 'New chat')}
          type="button"
        >
          <Icon name="plus" size={15} />
        </button>
      </div>
      <div className="chat-session-list" onScroll={onScroll} role="region"
        aria-label={tr('会话历史', 'Conversation history')} tabIndex={0}>
        {sessions.length ? (
          sessions.map((session) => (
            <div className={'chat-session-row' + (session.id === activeId ? ' active' : '')} key={session.id}>
              <button
                aria-current={session.id === activeId ? 'page' : undefined}
                className="chat-session-select-button"
                disabled={busy || recoveryNeeded}
                onClick={() => selectSession(session.id)}
                title={session.title || tr('未命名会话', 'Untitled chat')}
                type="button"
              >
                <strong>{session.title || tr('未命名会话', 'Untitled chat')}</strong>
                <small>{session.model}</small>
                <ModelVendorMark name={session.model} size={18} />
              </button>
              <button
                aria-label={tr(`删除对话：${session.title || '未命名会话'}`, `Delete chat: ${session.title || 'Untitled chat'}`)}
                className="chat-session-delete-button"
                disabled={busy || recoveryNeeded}
                onClick={() => void deleteConversation(session.id)}
                title={tr('删除对话', 'Delete chat')}
                type="button"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>
          ))
        ) : (
          <p>{conversation.loadingSessions ? tr('正在加载…', 'Loading…') : tr('暂无会话', 'No conversations yet')}</p>
        )}
        {sessions.length > 0 && conversation.loadingSessions && (
          <p role="status">{tr('正在加载…', 'Loading…')}</p>
        )}
        {sessions.length > 0 && !conversation.loadingSessions && !conversation.hasMoreSessions && (
          <p>{tr('没有更早的会话了', 'No older conversations')}</p>
        )}
      </div>
    </aside>
  );
}
