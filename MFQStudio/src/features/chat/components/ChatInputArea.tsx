/** 聊天输入区组合已有编辑器、滚动入口及语音组件安装提示。 */
import { ArrowDownIcon } from '@phosphor-icons/react';
import { useSettings } from '../../settings/SettingsProvider';
import { useJobStore } from '../../../stores/jobStore';
import { formatNumber, errorMessage } from '../../../app/formatters';
import { ChatComposer } from './ChatComposer';
import { ChatToolbar } from './ChatToolbar';
import type { ChatPageState } from '../hooks/useChatPageState';

/** 展示输入和安装状态，发送操作仍交由聊天领域处理。 */
export function ChatInputArea({ page }: { page: ChatPageState }) {
  const { tr } = useSettings();
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
          aria-label={tr('回到底部', 'Scroll to bottom')}
          title={tr('回到底部', 'Scroll to bottom')}
          type="button"
        >
          <ArrowDownIcon size={16} aria-hidden="true" />
        </button>
      )}
      {needsVoiceOutputComponent && voiceComponent && (
        <div className="voice-component-banner">
          <div>
            <strong>{voiceComponent.ready
              ? tr('语音组件已下载', 'Voice component downloaded')
              : tr('此模型还缺少语音输出组件', 'This model needs the voice output component')}</strong>
            <span>{voiceComponentJob
              ? tr(`正在下载并校验 · ${formatNumber(voiceComponentJob.progress * 100)}%`,
                `Downloading and verifying · ${formatNumber(voiceComponentJob.progress * 100)}%`)
              : voiceComponent.error
                ? voiceComponent.error
                : tr('Token2Wav 独立安装，不会重复占用每个模型的空间。',
                  'Token2Wav is installed once and shared by all compatible models.')}</span>
          </div>
          {voiceComponentJob && <progress max={1} value={voiceComponentJob.progress} />}
          <button
            disabled={voiceComponentBusy || Boolean(voiceComponentJob)}
            onClick={() => void installOrEnableVoiceOutput()}
            type="button"
          >
            {voiceComponentJob
              ? tr('正在下载…', 'Downloading…')
              : voiceComponent.ready
                ? tr('启用语音输出', 'Enable voice output')
                : tr(`下载组件 · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`,
                  `Download · ${formatNumber(voiceComponent.total_bytes / 1e9, 2)} GB`)}
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
          ? tr('向模型发送消息', 'Message MFQ')
          : conversation.modelAvailable
            ? tr('正在加载会话', 'Loading conversation')
            : tr('请先加载模型', 'Load a model first')}
        attachmentAccept={inference.attachmentAccept}
        tr={tr}
        onSend={send}
        onStop={generation.stop}
        onError={(cause) => conversation.setError(errorMessage(cause))}
        toolbar={<ChatToolbar />}
      />
      <p>{tr('模型输出可能存在错误，请核对重要信息。',
        'Model output may be inaccurate. Verify important information.')}</p>
    </div>
  );
}
