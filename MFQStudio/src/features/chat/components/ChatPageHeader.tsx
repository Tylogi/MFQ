/** Render the active chat title, model selector, and conversation controls. */
import { useTranslation } from 'react-i18next';
import { useConversationSelector } from '../state/conversationStore';
import { Icon } from '../../../app/display';
import { formatNumber } from '../../../app/formatters';
import type { ChatPageState } from '../hooks/useChatPageState';
import { ModelVendorMark } from '../../../app/ModelVendorMark';
import { useRef, useState } from 'react';
import { Dialog } from '../../../shared/ui/Dialog';
import { InferenceDefaultsPanel } from '../../connections/InferenceDefaultsPanel';

/** Display active-session controls and prevent model changes during recovery. */
export function ChatPageHeader({ page }: { page: ChatPageState }) {
  const { t } = useTranslation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsTrigger = useRef<HTMLButtonElement>(null);
  const messages = useConversationSelector((state) => state.messages);
  const { active, activeId, chat, chatSessionsOpen, setChatSessionsOpen, selectModel } = page;
  const { conversation, inference, voice, busy, recoveryNeeded, clearActiveConversation } = chat;
  const currentVoiceMessages = voice.voiceMessages.filter((message) => message.sessionId === activeId);
  return (
    <header className="chat-screen-header">
      <div className="chat-screen-title">
        <button
          aria-expanded={chatSessionsOpen}
          aria-label={chatSessionsOpen
            ? t('chat:chatPageHeader.collapseConversations')
            : t('chat:chatPageHeader.expandConversations')}
          className="chat-sidebar-toggle"
          onClick={() => setChatSessionsOpen((open) => !open)}
          title={chatSessionsOpen
            ? t('chat:chatPageHeader.collapseConversations')
            : t('chat:chatPageHeader.expandConversations')}
          type="button"
        >
          <span aria-hidden="true">{chatSessionsOpen ? '‹' : '›'}</span>
        </button>
        <h1>{active?.title || t('chat:chatPageHeader.chat')}</h1>
      </div>
      <div className="chat-screen-actions">
        <div className="chat-model-summary">
          {inference.availableModelNames.length > 1 ? (
            <select
              aria-label={t('chat:chatPageHeader.chatModel')}
              disabled={busy || recoveryNeeded}
              onChange={(event) => selectModel(event.target.value)}
              value={inference.selectedModel}
            >
              {inference.availableModelNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          ) : (
            <strong>{inference.selectedModel || t('chat:chatPageHeader.noModelLoaded')}</strong>
          )}
          <small>
            {t('chat:chatPageHeader.maxTokens', { tokens: formatNumber(inference.effectiveSettings.maxTokens) })}{' '}
            · {t('chat:chatPageHeader.temperature')} {formatNumber(inference.effectiveSettings.temperature, 2)} ·{' '}
            {t('chat:chatPageHeader.streaming')}
          </small>
        </div>
        <ModelVendorMark name={inference.selectedModel}
          architecture={inference.selectedModel === inference.runtime?.model ? inference.runtime.model_capabilities?.architecture_family || inference.runtime.model_type : undefined} />
        <span className={'runtime-status-pill ' + (conversation.conversationReady ? 'running' : 'stopped')}>
          <i />
          {conversation.conversationReady
            ? t('chat:chatPageHeader.ready')
            : inference.selectedModelLoading
              ? t('chat:chatPageHeader.loading')
              : t('chat:chatPageHeader.idle')}
        </span>
        <button
          ref={settingsTrigger} aria-label={t('chat:chatPageHeader.chatSettings')} className="chat-icon-button"
          onClick={() => setSettingsOpen(true)} type="button"><Icon name="settings" size={16} /></button>
        <Dialog open={settingsOpen} onOpenChange={setSettingsOpen} title={t('chat:chatPageHeader.chatSettings')}
          closeLabel={t('common:close')} className="chat-settings-dialog" returnFocusRef={settingsTrigger}>
          <div className="server-page"><InferenceDefaultsPanel /></div>
        </Dialog>
        <button
          aria-label={t('chat:chatPageHeader.clearConversation')}
          className="chat-icon-button"
          disabled={!conversation.conversationReady || busy || recoveryNeeded || (!messages.length && !currentVoiceMessages.length)}
          onClick={() => void clearActiveConversation()}
          title={t('chat:chatPageHeader.clearConversation')}
          type="button"
        >
          <Icon name="trash" size={14} />
        </button>
      </div>
    </header>
  );
}
