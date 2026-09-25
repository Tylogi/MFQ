/** 按模型能力提供聊天模式、语音播放和即时推理开关。 */
import type { CSSProperties } from 'react';
import type { SessionMode } from '../../../shared/api/types';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useVoiceLevel } from '../../voice/voiceLevelStore';
import { useSettings } from '../../settings/SettingsProvider';
import { Icon } from '../../../app/display';
import { Switch } from '../../../shared/ui/Switch';
const MODE_LABELS: Record<SessionMode, [string, string]> = {
  text: ['文本', 'Text'],
  voice: ['语音', 'Voice'],
  full_duplex: ['全双工', 'Full duplex'],
};
/** 展示当前会话允许的输入与推理选项，所有操作写入所属领域。 */
export function ChatToolbar() {
  const { settings, updateSettings, tr, english } = useSettings();
  const active = useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId) ?? null,
  );
  const voiceLevel = useVoiceLevel();
  const {
    conversation: { conversationReady },
    inference,
    voice,
    busy,
    selectInteractionMode,
    toggleVoice,
  } = useChat();
  const {
    capabilities,
    realtimeAvailable,
    visionSupported,
    visionAvailable,
    effectiveSettings,
    mtpSupported,
    mtpAvailable,
    thinkingSupported,
    reasoningValues,
    updateGlobalInference,
  } = inference;
  const { voiceState } = voice;
  const mode = active?.mode ?? 'text';
  return (
    <>
      {capabilities &&
        (capabilities.model_capabilities.features.audio_input ||
          capabilities.model_capabilities.features.full_duplex) && (
          <select
            aria-label={tr('交互模式', 'Interaction mode')}
            disabled={!conversationReady || busy || voiceState !== 'idle'}
            onChange={(event) => void selectInteractionMode(event.target.value as SessionMode)}
            value={active?.mode ?? mode}
          >
            {(['text', 'voice', 'full_duplex'] as SessionMode[]).map((item) => {
              const feature = capabilities.model_capabilities.features;
              const disabled =
                item === 'voice'
                  ? !feature.audio_input
                  : item === 'full_duplex'
                    ? !feature.full_duplex
                    : false;
              return (
                <option disabled={disabled} key={item} value={item}>
                  {MODE_LABELS[item][english ? 1 : 0]}
                </option>
              );
            })}
          </select>
        )}
      {realtimeAvailable && (
        <button
          aria-label={tr('语音输入', 'Voice input')}
          aria-pressed={voiceState !== 'idle' && voiceState !== 'error'}
          className="voice-button"
          disabled={!conversationReady || active?.mode === 'text' || busy}
          onClick={() => void toggleVoice()}
          style={{ '--voice-level': voiceLevel } as CSSProperties}
          title={
            active?.mode === 'text'
              ? tr('请先选择语音或全双工模式', 'Select voice or full duplex mode first')
              : voiceState === 'processing'
                ? tr('语音处理中', 'Processing voice')
                : tr('语音输入', 'Voice input')
          }
          type="button"
        >
          <span />
        </button>
      )}
      {realtimeAvailable && active?.mode !== 'text' && (
        <Switch
          label={tr('语音播放', 'Voice playback')}
          checked={settings.playbackEnabled}
          onCheckedChange={(playbackEnabled) => updateSettings({ playbackEnabled })}
        />
      )}
      {active?.mode === 'text' && visionSupported && (
        <button
          aria-label={tr('视觉输入', 'Vision input')}
          aria-pressed={visionAvailable && effectiveSettings.enableVision}
          disabled={!visionAvailable}
          onClick={() => updateGlobalInference({ enableVision: !effectiveSettings.enableVision })}
          title={
            visionAvailable
              ? tr('视觉输入', 'Vision input')
              : tr('当前模型文件没有视觉权重', 'The current model artifact has no vision weights')
          }
          type="button"
        >
          <Icon name="image" />
          {tr('视觉', 'Vision')}
        </button>
      )}
      {active?.mode === 'text' && mtpSupported && (
        <button
          aria-label="MTP"
          aria-pressed={mtpAvailable && effectiveSettings.enableMtp}
          disabled={!mtpAvailable}
          onClick={() => updateGlobalInference({ enableMtp: !effectiveSettings.enableMtp })}
          title={
            mtpAvailable
              ? 'MTP'
              : tr(
                  '当前模型文件没有完整 MTP 权重',
                  'The current model artifact has no complete MTP head',
                )
          }
          type="button"
        >
          <Icon name="text-forward" />
          MTP
        </button>
      )}
      {active?.mode === 'text' && (
        <button
          aria-pressed={thinkingSupported && effectiveSettings.enableThinking}
          disabled={!thinkingSupported}
          onClick={() =>
            updateGlobalInference({ enableThinking: !effectiveSettings.enableThinking })
          }
          type="button"
        >
          <Icon name="lightbulb" />
          {tr('思考', 'Thinking')}
        </button>
      )}
      {active?.mode === 'text' &&
        thinkingSupported &&
        effectiveSettings.enableThinking &&
        reasoningValues.length > 0 && (
          <select
            aria-label={tr('思考档位', 'Reasoning effort')}
            onChange={(event) => updateGlobalInference({ reasoningEffort: event.target.value })}
            value={effectiveSettings.reasoningEffort}
          >
            <option value="">{tr('标准', 'Standard')}</option>
            {reasoningValues.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        )}
      <span className="composer-hint">
        {voiceState !== 'idle'
          ? voiceState
          : tr('Enter 发送 · Shift+Enter 换行', 'Enter to send · Shift+Enter for newline')}
      </span>
    </>
  );
}
