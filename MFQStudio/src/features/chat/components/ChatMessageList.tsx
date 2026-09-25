/** 聊天消息区组合欢迎状态、历史、语音和流式回复。 */
import { lazy, Suspense } from 'react';
import { useSettings } from '../../settings/SettingsProvider';
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

/** 展示当前会话消息并维持历史消息组件的按需加载。 */
export function ChatMessageList({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
  const { mcpTools } = useChatTools();
  const messages = useConversationSelector((state) => state.messages);
  const responses = useConversationSelector((state) => state.responses);
  const {
    active, activeId, chat, editDraft, setEditDraft, saveCurrentEdit,
    createSession, chooseModelDirectory, scroll,
  } = page;
  const { voice, generation, inference, messageActions, busy } = chat;
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
                  ? tr('模型加载中', 'Model loading')
                  : tr('尚未加载模型', 'No model loaded')}</h1>
                <p>{inference.selectedModelLoading
                  ? tr('加载完成后即可开始对话。', 'Chat becomes available as soon as loading completes.')
                  : tr('选择本地检查点后即可开始对话。', 'Choose a local checkpoint to use the inference playground.')}</p>
                {!inference.selectedModelLoading && (
                  <button className="open-model-primary" disabled={busy} onClick={chooseModelDirectory} type="button">
                    <Icon name="folder" />{tr('选择模型', 'Choose model')}
                  </button>
                )}
              </>
            ) : !active ? (
              <>
                <h1>{tr('开始对话', 'Start a conversation')}</h1>
                <p>{tr('连接到当前配置的 MFQ 服务。', 'Connected to your configured MFQ service.')}</p>
                <button className="open-model-primary" disabled={busy} onClick={() => void createSession()} type="button">
                  {tr('开始对话', 'Start chat')}
                </button>
              </>
            ) : (
              <>
                <h1>{tr('开始对话', 'Start a conversation')}</h1>
                <p>{tr('连接到当前配置的 MFQ 服务。', 'Connected to your configured MFQ service.')}</p>
              </>
            )}
          </div>
        )}
        <Suspense fallback={<p role="status">{tr('正在加载消息…', 'Loading messages…')}</p>}>
          <SavedMessageList
            messages={messages}
            responses={responses}
            mcpTools={mcpTools}
            busy={busy}
            tr={tr}
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
        <StreamingMessage controller={generation} sessionId={activeId} tr={tr} />
      </div>
    </div>
  );
}
