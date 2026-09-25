/** 聊天会话侧栏展示历史列表并转发页面级会话操作。 */
import { useSettings } from '../../settings/SettingsProvider';
import { Icon } from '../../../app/display';
import { useConversationSelector } from '../state/conversationStore';
import type { ChatPageState } from '../hooks/useChatPageState';

/** 展示会话列表与新建入口。 */
export function ChatSessionSidebar({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
  const sessions = useConversationSelector((state) => state.sessions);
  const { activeId, chatSessionsOpen, selectSession, createSession, chat } = page;
  const { busy, conversation, deleteConversation } = chat;
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
          disabled={busy || conversation.transitioning || !conversation.modelAvailable}
          onClick={() => void createSession()}
          title={tr('新建会话', 'New chat')}
          type="button"
        >
          <Icon name="plus" size={15} />
        </button>
      </div>
      <div className="chat-session-list">
        {sessions.length ? (
          sessions.map((session) => (
            <div className={'chat-session-row' + (session.id === activeId ? ' active' : '')} key={session.id}>
              <button
                aria-current={session.id === activeId ? 'page' : undefined}
                className="chat-session-select-button"
                disabled={busy || conversation.transitioning}
                onClick={() => selectSession(session.id)}
                title={session.title || tr('未命名会话', 'Untitled chat')}
                type="button"
              >
                <strong>{session.title || tr('未命名会话', 'Untitled chat')}</strong>
                <small>{session.model}</small>
              </button>
              <button
                aria-label={tr(`删除对话：${session.title || '未命名会话'}`, `Delete chat: ${session.title || 'Untitled chat'}`)}
                className="chat-session-delete-button"
                disabled={busy || conversation.transitioning}
                onClick={() => void deleteConversation(session.id)}
                title={tr('删除对话', 'Delete chat')}
                type="button"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>
          ))
        ) : (
          <p>{tr('暂无会话', 'No conversations yet')}</p>
        )}
      </div>
    </aside>
  );
}
