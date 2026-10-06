/** Resolve chat inference capabilities and request parameters from the selected instance, session mode, and user settings. */
import { useCallback, useMemo } from 'react';
import { useRuntime } from '../../../app/RuntimeProvider';
import { useSettings } from '../../settings/SettingsProvider';
import { modeTemplateSettings, type GenerationSettings } from '../../settings/configuration';
import { DOCUMENT_ACCEPT } from '../attachments';
import { runtimeModelNames } from '../../runtime/modelSelection';
import type { SessionMode, SamplingParams } from '../../../shared/api/types';
import { useJobStore } from '../../../stores/jobStore';

const EMPTY_REASONING_VALUES: string[] = [];
/** Return chat-specific derived capabilities, recalculating model defaults when the instance changes. */
export function useChatInference(mode: SessionMode) {
  const runtimeContext = useRuntime();
  const {
    runtime,
    realtime,
    capabilities,
    selectedModel: model,
    instances,
    models,
  } = runtimeContext;
  const modelLoadingJob = useJobStore((state) =>
    state.jobs.some(
      (job) =>
        job.kind === 'model.load' &&
        job.payload.model === model &&
        ['queued', 'running', 'cancelling'].includes(job.status),
    ),
  );
  const { settings, updateSettings } = useSettings();
  const effectiveSettings = useMemo(
    () =>
      settings.inheritModelDefaults
        ? modeTemplateSettings(settings, mode, runtime, realtime)
        : settings,
    [settings, mode, runtime, realtime],
  );
  const instance = instances.find((item) => item.model === model && item.state !== 'failed');
  const thinkingSupported = runtime?.chat_template_capabilities?.thinking?.supported === true;
  const reasoningValues =
    runtime?.chat_template_capabilities?.reasoning_effort?.values ?? EMPTY_REASONING_VALUES;
  const visionSupported = Boolean(
    capabilities?.model_capabilities.features.image_input ||
    capabilities?.model_capabilities.features.video_input,
  );
  const visionAvailable = Boolean(capabilities?.vision_available || visionSupported);
  const mtpSupported =
    instance?.mtp_supported ??
    (capabilities?.model === model && capabilities?.model_capabilities.features.mtp === true);
  const mtpAvailable =
    instance?.mtp_available ??
    (capabilities?.model === model && capabilities?.mtp_available === true);
  const attachmentAccept = [
    effectiveSettings.enableVision && capabilities?.model_capabilities.features.image_input
      ? 'image/*'
      : '',
    effectiveSettings.enableVision && capabilities?.model_capabilities.features.video_input
      ? 'video/*'
      : '',
    capabilities?.model_capabilities.features.audio_input ? 'audio/*' : '',
    DOCUMENT_ACCEPT,
  ]
    .filter(Boolean)
    .join(',');
  const sampling: SamplingParams = useMemo(
    () => ({
      max_tokens: effectiveSettings.maxTokens,
      temperature: effectiveSettings.temperature,
      top_k: effectiveSettings.topK,
      top_p: effectiveSettings.topP,
      presence_penalty: effectiveSettings.presencePenalty,
      frequency_penalty: effectiveSettings.frequencyPenalty,
      repetition_penalty: effectiveSettings.repetitionPenalty,
      seed: effectiveSettings.seed,
      enable_thinking: thinkingSupported && effectiveSettings.enableThinking,
      enable_vision: effectiveSettings.enableVision,
      enable_mtp: mtpSupported && mtpAvailable && effectiveSettings.enableMtp,
      reasoning_effort: effectiveSettings.reasoningEffort || null,
    }),
    [effectiveSettings, thinkingSupported, mtpSupported, mtpAvailable],
  );
/** Immediately update the current inference setting and disable default inheritance. */
  const updateGlobalInference = useCallback(
    (patch: Partial<GenerationSettings>) => {
      updateSettings({
        ...effectiveSettings,
        ...patch,
        inheritModelDefaults: false,
        preset: 'custom',
      });
    },
    [updateSettings, effectiveSettings],
  );
  const availableModelNames = useMemo(
    () => runtimeModelNames(models, instances),
    [models, instances],
  );
  return useMemo(
    () => ({
      ...runtimeContext,
      effectiveSettings,
      sampling,
      thinkingSupported,
      reasoningValues,
      visionSupported,
      visionAvailable,
      mtpSupported,
      mtpAvailable,
      attachmentAccept,
      updateGlobalInference,
      availableModelNames,
      selectedModelLoading: instance?.state === 'loading' || modelLoadingJob,
      realtimeAvailable: realtime?.available === true,
    }),
    [
      runtimeContext,
      effectiveSettings,
      sampling,
      thinkingSupported,
      reasoningValues,
      visionSupported,
      visionAvailable,
      mtpSupported,
      mtpAvailable,
      attachmentAccept,
      updateGlobalInference,
      availableModelNames,
      instance?.state,
      modelLoadingJob,
      realtime?.available,
    ],
  );
}
