/** Provide chat-mode, voice-playback, and immediate-inference switches based on model capabilities. */
import { useTranslation } from 'react-i18next';
import type { CSSProperties } from 'react';
import type { SessionMode } from '../../../shared/api/types';
import { useChat } from '../ChatProvider';
import { useConversationSelector } from '../state/conversationStore';
import { useVoiceLevel } from '../../voice/voiceLevelStore';
import { useSettings } from '../../settings/SettingsProvider';
import { Icon } from '../../../app/display';
import { Switch } from '../../../shared/ui/Switch';
const MODE_LABELS = {
  text: 'chat:modes.text',
  voice: 'chat:modes.voice',
  full_duplex: 'chat:modes.fullDuplex',
} as const satisfies Record<SessionMode, string>;
/** Display input and inference options allowed by the active session, routing each action to its owning domain. */
export function ChatToolbar() {
  const { settings, updateSettings } = useSettings();
  const { t } = useTranslation();
  const active = useConversationSelector(
    (state) => state.sessions.find((session) => session.id === state.activeId) ?? null,
  );
  const voiceLevel = useVoiceLevel();
  const {
    conversation: { conversationReady },
    inference,
    voice,
    busy,
    recoveryNeeded,
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
            aria-label={t('chat:chatToolbar.interactionMode')}
            disabled={!conversationReady || busy || recoveryNeeded || voiceState !== 'idle'}
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
                  {t(MODE_LABELS[item])}
                </option>
              );
            })}
          </select>
        )}
      {realtimeAvailable && (
        <button
          aria-label={t('chat:chatToolbar.voiceInput')}
          aria-pressed={voiceState !== 'idle' && voiceState !== 'error'}
          className="voice-button"
          disabled={!conversationReady || active?.mode === 'text' || busy || recoveryNeeded}
          onClick={() => void toggleVoice()}
          style={{ '--voice-level': voiceLevel } as CSSProperties}
          title={
            active?.mode === 'text'
              ? t('chat:chatToolbar.selectVoiceOrFullDuplexModeFirst')
              : voiceState === 'processing'
                ? t('chat:chatToolbar.processingVoice')
                : t('chat:chatToolbar.voiceInput')
          }
          type="button"
        >
          <span />
        </button>
      )}
      {realtimeAvailable && active?.mode !== 'text' && (
        <Switch
          label={t('chat:chatToolbar.voicePlayback')}
          checked={settings.playbackEnabled}
          onCheckedChange={(playbackEnabled) => updateSettings({ playbackEnabled })}
        />
      )}
      {active?.mode === 'text' && visionSupported && (
        <button
          aria-label={t('chat:chatToolbar.visionInput')}
          aria-pressed={visionAvailable && effectiveSettings.enableVision}
          disabled={!visionAvailable}
          onClick={() => updateGlobalInference({ enableVision: !effectiveSettings.enableVision })}
          title={
            visionAvailable
              ? t('chat:chatToolbar.visionInput')
              : t('chat:chatToolbar.theCurrentModelArtifactHasNoVisionWeights')
          }
          type="button"
        >
          <Icon name="image" />
          {t('chat:chatToolbar.vision')}
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
              : t('chat:chatToolbar.theCurrentModelArtifactHasNoCompleteMtpHead')
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
          {t('chat:chatToolbar.thinking')}
        </button>
      )}
      {active?.mode === 'text' &&
        thinkingSupported &&
        effectiveSettings.enableThinking &&
        reasoningValues.length > 0 && (
          <select
            aria-label={t('chat:chatToolbar.reasoningEffort')}
            onChange={(event) => updateGlobalInference({ reasoningEffort: event.target.value })}
            value={effectiveSettings.reasoningEffort}
          >
            <option value="">{t('chat:chatToolbar.standard')}</option>
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
          : t('chat:chatToolbar.enterToSendShiftEnterForNewline')}
      </span>
    </>
  );
}
