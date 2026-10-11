import { useSettings } from '../../settings/SettingsProvider';
import { Icon } from '../../../app/display';
import { useConversationSelector } from '../state/conversationStore';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ModelVendorMark } from '../../../app/ModelVendorMark';
import { useEffect, useRef, useState } from 'react';
import { CaretRightIcon, XIcon } from '@phosphor-icons/react';

export function ChatSessionSidebar({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
  const sessions = useConversationSelector((state) => state.sessions);
  const { activeId, chatSessionsOpen, setChatSessionsOpen, selectSession, createSession, chat } = page;
  const { busy, conversation, deleteConversation, deleteAllConversations } = chat;
  const [editingId, setEditingId] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const nameInput = useRef<HTMLInputElement>(null);
  const blocked = busy || conversation.transitioning;
  useEffect(() => {
    if (editingId && (!chatSessionsOpen || !sessions.some((session) => session.id === editingId))) setEditingId(null);
  }, [editingId, chatSessionsOpen, sessions]);
  useEffect(() => {
    if (editingId) {
      nameInput.current?.focus();
      nameInput.current?.select();
    }
  }, [editingId]);
  async function saveName() {
    if (!editingId || blocked || !title.trim()) return;
    if (await conversation.renameSession(editingId, title)) setEditingId(null);
  }
  return (
    <aside
      id="chat-conversations"
      className={'chat-session-sidebar' + (chatSessionsOpen ? ' open' : '')}
      aria-label={tr('会话列表', 'Conversations')}
    >
      <div className="chat-session-sidebar-header">
        <button className="chat-session-disclosure" aria-controls="chat-conversations"
          aria-expanded={chatSessionsOpen} aria-label={tr('收起会话列表', 'Collapse conversations')}
          onClick={() => setChatSessionsOpen(false)} type="button">
          <CaretRightIcon size={12} weight="fill" aria-hidden="true" /><strong>{tr('对话', 'Chats')}</strong>
        </button>
        <div className="chat-session-header-actions">
        <button aria-label={tr('删除全部对话', 'Delete all conversations')} className="chat-icon-button chat-delete-all"
          disabled={blocked || !sessions.length} onClick={() => void deleteAllConversations()}
          title={tr('删除全部对话', 'Delete all conversations')} type="button"><Icon name="trash" size={15} /></button>
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
      </div>
      <div className="chat-session-list" role="list" aria-label={tr('对话记录', 'Conversation history')} tabIndex={0}>
        {sessions.length ? (
          sessions.map((session) => (
            <div className={'chat-session-row' + (session.id === activeId ? ' active' : '')} key={session.id} role="listitem">
              {editingId === session.id ? <form className="chat-session-rename-form" onSubmit={(event) => { event.preventDefault(); void saveName(); }}>
                <input ref={nameInput} aria-label={tr('对话名称', 'Conversation name')} disabled={blocked}
                  maxLength={512} required value={title} onChange={(event) => setTitle(event.target.value)}
                  onKeyDown={(event) => { if (event.key === 'Escape') { event.preventDefault(); setEditingId(null); } }} />
                <button aria-label={tr('保存名称', 'Save name')} className="chat-session-action-button"
                  disabled={blocked || !title.trim()} title={tr('保存名称', 'Save name')} type="submit"><Icon name="check" size={14} /></button>
                <button aria-label={tr('取消改名', 'Cancel rename')} className="chat-session-action-button"
                  disabled={blocked} onClick={() => setEditingId(null)} title={tr('取消改名', 'Cancel rename')} type="button"><XIcon size={14} aria-hidden="true" /></button>
              </form> : <>
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
                <ModelVendorMark name={session.model} size={18} />
              </button>
              <button aria-label={tr(`重命名对话：${session.title || '未命名会话'}`, `Rename chat: ${session.title || 'Untitled chat'}`)}
                className="chat-session-action-button" disabled={blocked}
                onClick={() => { setTitle(session.title || ''); setEditingId(session.id); }}
                title={tr('重命名对话', 'Rename chat')} type="button"><Icon name="edit" size={14} /></button>
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
              </>}
            </div>
          ))
        ) : (
          <p>{tr('暂无会话', 'No conversations yet')}</p>
        )}
      </div>
    </aside>
  );
}
