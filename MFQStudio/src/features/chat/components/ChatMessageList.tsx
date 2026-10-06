/** Compose the welcome state, history, voice, and streaming response in the chat message area. */
import { useTranslation } from 'react-i18next';
import { lazy, Suspense } from 'react';
import { useChatTools } from '../ChatToolsProvider';
import { useConversationSelector } from '../state/conversationStore';
import { Icon } from '../../../app/display';
import { AudioClip } from '../../voice/AudioClip';
import { renderMarkdown } from '../MessageMarkdown';
import { StreamingMessage } from './StreamingMessage';
import type { ChatPageState } from '../hooks/useChatPageState';

const SavedMessageList = lazy(() =>
  import('../SavedMessageList').then((module) => ({ default: module.SavedMessageList })),
);
/** Display current-session messages while preserving lazy loading of saved-message components. */
export function ChatMessageList({ page }: { page: ChatPageState }) {
  const { t } = useTranslation();
  const { mcpTools } = useChatTools();
  const messages = useConversationSelector((state) => state.messages);
  const responses = useConversationSelector((state) => state.responses);
  const {
    active, activeId, chat, editDraft, setEditDraft, saveCurrentEdit,
    createSession, chooseModelDirectory, scroll,
  } = page;
  const { voice, generation, inference, messageActions, busy, recoveryNeeded } = chat;
  const currentVoiceMessages = voice.voiceMessages.filter((message) => message.sessionId === activeId);
  const live = generation.getSnapshot().live;
  return (
    <div className="message-scroller" onScroll={scroll.handleScroll} ref={scroll.scrollerRef}>
      <div className="message-list">
        {!messages.length && !currentVoiceMessages.length && !live && (
          <div className="welcome">
            <Icon name="chat" size={34} />
            {!chat.conversation.modelAvailable ? (
              <>
                <h1>{inference.selectedModelLoading
                  ? t('chat:chatMessageList.modelLoading')
                  : t('chat:chatMessageList.noModelLoaded')}</h1>
                <p>{inference.selectedModelLoading
                  ? t('chat:chatMessageList.chatBecomesAvailableAsSoonAsLoadingCompletes')
                  : t('chat:chatMessageList.chooseALocalCheckpointToUseTheInferencePlayground')}</p>
                {!inference.selectedModelLoading && (
                  <button className="open-model-primary" disabled={busy} onClick={chooseModelDirectory} type="button">
                    <Icon name="folder" />{t('chat:chatMessageList.chooseModel')}
                  </button>
                )}
              </>
            ) : !active ? (
              <>
                <h1>{t('chat:chatMessageList.startAConversation')}</h1>
                <p>{t('chat:chatMessageList.connectedToYourConfiguredMfqService')}</p>
                <button className="open-model-primary" disabled={busy || recoveryNeeded} onClick={() => void createSession()} type="button">
                  {t('chat:chatMessageList.startChat')}
                </button>
              </>
            ) : (
              <>
                <h1>{t('chat:chatMessageList.startAConversation')}</h1>
                <p>{t('chat:chatMessageList.connectedToYourConfiguredMfqService')}</p>
              </>
            )}
          </div>
        )}
        <Suspense fallback={<p role="status">{t('chat:chatMessageList.loadingMessages')}</p>}>
          <SavedMessageList
            messages={messages}
            responses={responses}
            mcpTools={mcpTools}
            busy={busy}
            t={t}
            editDraft={editDraft}
            setEditDraft={setEditDraft}
            actions={{ saveEdit: saveCurrentEdit, copyMessage: messageActions.copyMessage,
              regenerate: messageActions.regenerate, executeToolCalls: messageActions.executeToolCalls }}
          />
        </Suspense>
        {currentVoiceMessages.map((message) => (
          <article className={`message message-${message.role}`} key={message.id}>
            <div className="message-body">
              {message.text && renderMarkdown(message.text, false, message.role === 'assistant')}
              {message.audioId && <AudioClip audioId={message.audioId} />}
            </div>
          </article>
        ))}
        {voice.liveVoice?.sessionId === activeId && voice.liveVoice.text && (
          <article className="message message-assistant live-message">
            <div className="message-body">{renderMarkdown(voice.liveVoice.text, true, true)}</div>
          </article>
        )}
        <StreamingMessage controller={generation} sessionId={activeId} t={t} />
      </div>
    </div>
  );
}
