/** Compose the existing editor, scroll controls, and voice-component installation prompt in the chat input area. */
import { useTranslation } from 'react-i18next';
import { ArrowDownIcon } from '@phosphor-icons/react';
import { useJobStore } from '../../../stores/jobStore';
import { formatNumber, errorMessage } from '../../../app/formatters';
import { ChatComposer } from './ChatComposer';
import { ChatToolbar } from './ChatToolbar';
import type { ChatPageState } from '../hooks/useChatPageState';
/** Display input and installation state while leaving send actions to the chat domain. */
export function ChatInputArea({ page }: { page: ChatPageState }) {
  const { t } = useTranslation();
  const { chat, active, activeId, scroll } = page;
  const {
    conversation, inference, generation, generationPhase, busy, recoveryNeeded,
    voiceComponentBusy, installOrEnableVoiceOutput, send,
  } = chat;
  const voiceComponentJob = useJobStore((state) =>
    state.jobs.find((job) =>
      job.kind === 'component.voice_output.install' &&
      ['queued', 'running', 'cancelling'].includes(job.status)),
  );
  const voiceComponent = inference.voiceComponent;
  const needsVoiceOutputComponent =
    active?.mode === 'full_duplex' &&
    Boolean(inference.capabilities?.model_capabilities.features.audio_output) &&
    !inference.realtimeAvailable;
  return (
    <div className="composer-region">
      {!scroll.following && (
        <button
          className="chat-scroll-bottom"
          onClick={scroll.scrollToBottom}
          aria-label={t('chat:chatInputArea.scrollToBottom')}
          title={t('chat:chatInputArea.scrollToBottom')}
          type="button"
        >
          <ArrowDownIcon size={16} aria-hidden="true" />
        </button>
      )}
      {needsVoiceOutputComponent && voiceComponent && (
        <div className="voice-component-banner">
          <div>
            <strong>{voiceComponent.ready
              ? t('chat:chatInputArea.voiceComponentDownloaded')
              : t('chat:chatInputArea.thisModelNeedsTheVoiceOutputComponent')}</strong>
            <span>{voiceComponentJob
              ? t('chat:chatInputArea.downloadingAndVerifying', { progress: formatNumber(voiceComponentJob.progress * 100) })
              : voiceComponent.error
                ? voiceComponent.error
                : t('chat:chatInputArea.token2wavIsInstalledOnceAndSharedByAllCompatibleModels')}</span>
          </div>
          {voiceComponentJob && <progress max={1} value={voiceComponentJob.progress} />}
          <button
            disabled={voiceComponentBusy || Boolean(voiceComponentJob)}
            onClick={() => void installOrEnableVoiceOutput()}
            type="button"
          >
            {voiceComponentJob
              ? t('chat:chatInputArea.downloading')
              : voiceComponent.ready
                ? t('chat:chatInputArea.enableVoiceOutput')
                : t('chat:chatInputArea.downloadGb', { size: formatNumber(voiceComponent.total_bytes / 1e9, 2) })}
          </button>
        </div>
      )}
      {conversation.error && (
        <div className="error-banner" role="alert">
          <span>{conversation.error}</span>
          <button
            aria-label={t('chat:chatInputArea.dismissError')}
            onClick={() => conversation.setError(null)}
            type="button"
          >
            ×
          </button>
        </div>
      )}
      <ChatComposer
        sessionId={activeId ?? 'new'}
        ready={conversation.conversationReady}
        busy={busy}
        recoveryNeeded={recoveryNeeded}
        phase={generationPhase}
        placeholder={conversation.conversationReady
          ? t('chat:chatInputArea.messageMfq')
          : conversation.modelAvailable
            ? t('chat:chatInputArea.loadingConversation')
            : t('chat:chatInputArea.loadAModelFirst')}
        attachmentAccept={inference.attachmentAccept}
        t={t}
        onSend={send}
        onStop={generation.stop}
        onError={(cause) => conversation.setError(errorMessage(cause))}
        toolbar={<ChatToolbar />}
      />
      <p>{t('chat:chatInputArea.modelOutputMayBeInaccurateVerifyImportantInformation')}</p>
    </div>
  );
}
