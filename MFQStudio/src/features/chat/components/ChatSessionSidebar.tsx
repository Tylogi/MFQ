/** Render chat sessions and guard navigation while generation recovery is pending. */
import { useTranslation } from 'react-i18next';
import type { UIEvent } from 'react';
import { Icon } from '../../../app/display';
import { useConversationSelector } from '../state/conversationStore';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ModelVendorMark } from '../../../app/ModelVendorMark';

/** Display session selection, creation, and deletion controls. */
export function ChatSessionSidebar({ page }: { page: ChatPageState }) {
  const { t } = useTranslation();
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
      aria-label={t('chat:chatSessionSidebar.conversations')}
    >
      <div className="chat-session-sidebar-header">
        <strong>{t('chat:chatSessionSidebar.chats')}</strong>
        <button
          aria-label={t('chat:chatSessionSidebar.newChat')}
          className="chat-icon-button"
          disabled={busy || recoveryNeeded || !conversation.modelAvailable}
          onClick={() => void createSession()}
          title={t('chat:chatSessionSidebar.newChat')}
          type="button"
        >
          <Icon name="plus" size={15} />
        </button>
      </div>
      <div className="chat-session-list" onScroll={onScroll} role="region"
        aria-label={t('chat:chatSessionSidebar.conversationHistory')} tabIndex={0}>
        {sessions.length ? (
          sessions.map((session) => (
            <div className={'chat-session-row' + (session.id === activeId ? ' active' : '')} key={session.id}>
              <button
                aria-current={session.id === activeId ? 'page' : undefined}
                className="chat-session-select-button"
                disabled={busy || recoveryNeeded}
                onClick={() => selectSession(session.id)}
                title={session.title || t('common:untitledChat')}
                type="button"
              >
                <strong>{session.title || t('common:untitledChat')}</strong>
                <small>{session.model}</small>
                <ModelVendorMark name={session.model} size={18} />
              </button>
              <button
                aria-label={t('chat:chatSessionSidebar.deleteChat', { title: (session.title || t('common:untitledChat')) })}
                className="chat-session-delete-button"
                disabled={busy || recoveryNeeded}
                onClick={() => void deleteConversation(session.id)}
                title={t('chat:chatSessionSidebar.deleteChat2')}
                type="button"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>
          ))
        ) : (
          <p>{conversation.loadingSessions ? t('common:loading') : t('chat:chatSessionSidebar.noConversationsYet')}</p>
        )}
        {sessions.length > 0 && conversation.loadingSessions && (
          <p role="status">{t('common:loading')}</p>
        )}
        {sessions.length > 0 && !conversation.loadingSessions && !conversation.hasMoreSessions && (
          <p>{t('chat:chatSessionSidebar.noOlderConversations')}</p>
        )}
      </div>
    </aside>
  );
}
