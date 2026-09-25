/** 聊天页头展示当前会话、模型和运行状态。 */
import { useSettings } from '../../settings/SettingsProvider';
import { useConversationSelector } from '../state/conversationStore';
import { Icon } from '../../../app/display';
import { formatNumber } from '../../../app/formatters';
import type { ChatPageState } from '../hooks/useChatPageState';

/** 渲染会话标题、模型切换和清空入口。 */
export function ChatPageHeader({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
  const messages = useConversationSelector((state) => state.messages);
  const { active, activeId, chat, chatSessionsOpen, setChatSessionsOpen, selectModel } = page;
  const { conversation, inference, voice, busy, clearActiveConversation } = chat;
  const currentVoiceMessages = voice.voiceMessages.filter((message) => message.sessionId === activeId);
  return (
    <header className="chat-screen-header">
      <div className="chat-screen-title">
        <button
          aria-expanded={chatSessionsOpen}
          aria-label={chatSessionsOpen
            ? tr('收起会话列表', 'Collapse conversations')
            : tr('展开会话列表', 'Expand conversations')}
          className="chat-sidebar-toggle"
          onClick={() => setChatSessionsOpen((open) => !open)}
          title={chatSessionsOpen
            ? tr('收起会话列表', 'Collapse conversations')
            : tr('展开会话列表', 'Expand conversations')}
          type="button"
        >
          <span aria-hidden="true">{chatSessionsOpen ? '‹' : '›'}</span>
        </button>
        <h1>{active?.title || tr('对话', 'Chat')}</h1>
      </div>
      <div className="chat-screen-actions">
        <div className="chat-model-summary">
          {inference.availableModelNames.length > 1 ? (
            <select
              aria-label={tr('对话模型', 'Chat model')}
              disabled={busy || conversation.transitioning}
              onChange={(event) => selectModel(event.target.value)}
              value={inference.selectedModel}
            >
              {inference.availableModelNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          ) : (
            <strong>{inference.selectedModel || tr('尚未加载模型', 'No model loaded')}</strong>
          )}
          <small>
            {tr(
              '最多 ' + formatNumber(inference.effectiveSettings.maxTokens) + ' tokens',
              formatNumber(inference.effectiveSettings.maxTokens) + ' max tokens',
            )}{' '}
            · {tr('温度', 'temperature')} {formatNumber(inference.effectiveSettings.temperature, 2)} ·{' '}
            {tr('流式', 'streaming')}
          </small>
        </div>
        <span className={'runtime-status-pill ' + (conversation.conversationReady ? 'running' : 'stopped')}>
          <i />
          {conversation.conversationReady
            ? tr('就绪', 'Ready')
            : inference.selectedModelLoading
              ? tr('加载中', 'Loading')
              : tr('空闲', 'Idle')}
        </span>
        <button
          aria-label={tr('清空对话', 'Clear conversation')}
          className="chat-icon-button"
          disabled={!conversation.conversationReady || busy || (!messages.length && !currentVoiceMessages.length)}
          onClick={() => void clearActiveConversation()}
          title={tr('清空对话', 'Clear conversation')}
          type="button"
        >
          <Icon name="trash" size={14} />
        </button>
      </div>
    </header>
  );
}
