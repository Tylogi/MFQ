/** 管理生成预设的本地校验、设置快照及服务资源转换。 */
import { GenerationPresetResource, SessionMode } from '../../shared/api/types';
import { GenerationSettings, DEFAULT_SETTINGS } from './configuration';

export type StoredPresetSettings = Pick<
  GenerationSettings,
  | "systemPrompt"
  | "maxTokens"
  | "temperature"
  | "topP"
  | "topK"
  | "repetitionPenalty"
  | "presencePenalty"
  | "frequencyPenalty"
  | "enableThinking"
  | "enableVision"
  | "enableMtp"
  | "reasoningEffort"
  | "excludeReasoning"
  | "seed"
>;

export interface StoredPreset {
  id?: string;
  name: string;
  settings: StoredPresetSettings;
  inheritGlobalSettings: boolean;
  contextSize: number;
  model?: string | null;
  mode?: SessionMode | null;
  icon?: string;
  updatedAt: string;
}

export const STORED_PRESETS_KEY = "mfq.studio.presets.v1";

/** 截取可持久化的推理参数，不包含界面偏好。 */
export function presetSnapshot(settings: GenerationSettings): StoredPresetSettings {
  return {
    systemPrompt: settings.systemPrompt,
    maxTokens: settings.maxTokens,
    temperature: settings.temperature,
    topP: settings.topP,
    topK: settings.topK,
    repetitionPenalty: settings.repetitionPenalty,
    presencePenalty: settings.presencePenalty,
    frequencyPenalty: settings.frequencyPenalty,
    enableThinking: settings.enableThinking,
    enableVision: settings.enableVision,
    enableMtp: settings.enableMtp,
    reasoningEffort: settings.reasoningEffort,
    excludeReasoning: settings.excludeReasoning,
    seed: settings.seed,
  };
}

/** 读取并校验本地生成预设，丢弃无效条目并限制历史数量。 */
export function loadStoredPresets(): StoredPreset[] {
  try {
    const decoded = JSON.parse(localStorage.getItem(STORED_PRESETS_KEY) || "[]");
    if (!Array.isArray(decoded)) return [];
    const fallback = presetSnapshot(DEFAULT_SETTINGS);
    return decoded.flatMap((candidate): StoredPreset[] => {
      if (!candidate || typeof candidate !== "object") return [];
      const raw = candidate as Record<string, unknown>;
      const name = typeof raw.name === "string" ? raw.name.replace(/\s+/g, " ").trim() : "";
      const source =
        raw.settings && typeof raw.settings === "object"
          ? (raw.settings as Record<string, unknown>)
          : {};
      if (!name) return [];
      const number = (key: keyof StoredPresetSettings, defaultValue: number) => {
        const value = Number(source[key]);
        return Number.isFinite(value) ? value : defaultValue;
      };
      const contextSize = Number(raw.contextSize);
      return [{
        id: typeof raw.id === "string" ? raw.id : undefined,
        name: name.slice(0, 64),
        settings: {
          systemPrompt:
            typeof source.systemPrompt === "string" ? source.systemPrompt : fallback.systemPrompt,
          maxTokens: number("maxTokens", fallback.maxTokens),
          temperature: number("temperature", fallback.temperature),
          topP: number("topP", fallback.topP),
          topK: number("topK", fallback.topK),
          repetitionPenalty: number("repetitionPenalty", fallback.repetitionPenalty),
          presencePenalty: number("presencePenalty", fallback.presencePenalty),
          frequencyPenalty: number("frequencyPenalty", fallback.frequencyPenalty),
          enableThinking:
            typeof source.enableThinking === "boolean"
              ? source.enableThinking
              : fallback.enableThinking,
          enableVision:
            typeof source.enableVision === "boolean"
              ? source.enableVision
              : fallback.enableVision,
          enableMtp:
            typeof source.enableMtp === "boolean"
              ? source.enableMtp
              : fallback.enableMtp,
          reasoningEffort:
            typeof source.reasoningEffort === "string"
              ? source.reasoningEffort
              : fallback.reasoningEffort,
          excludeReasoning:
            typeof source.excludeReasoning === "boolean"
              ? source.excludeReasoning
              : fallback.excludeReasoning,
          seed:
            source.seed == null || source.seed === ""
              ? null
              : number("seed", fallback.seed ?? 0),
        },
        inheritGlobalSettings:
          typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true,
        contextSize:
          Number.isFinite(contextSize) && contextSize >= 512
            ? Math.floor(contextSize)
            : 32768,
        model: typeof raw.model === "string" ? raw.model : null,
        mode:
          raw.mode === "text" || raw.mode === "voice" || raw.mode === "full_duplex"
            ? raw.mode
            : null,
        icon: typeof raw.icon === "string" ? raw.icon.slice(0, 8) : undefined,
        updatedAt:
          typeof raw.updatedAt === "string" ? raw.updatedAt : new Date(0).toISOString(),
      }];
    }).slice(0, 50);
  } catch {
    return [];
  }
}

/** 将服务端生成预设转换为界面使用的设置模型。 */
export function storedPresetFromResource(preset: GenerationPresetResource): StoredPreset {
  const sampling = preset.settings.sampling;
  return {
    id: preset.id,
    name: preset.name,
    settings: {
      systemPrompt: preset.settings.system_prompt ?? "",
      maxTokens: sampling.max_tokens,
      temperature: sampling.temperature,
      topP: sampling.top_p,
      topK: sampling.top_k,
      repetitionPenalty: sampling.repetition_penalty,
      presencePenalty: sampling.presence_penalty,
      frequencyPenalty: sampling.frequency_penalty,
      enableThinking: sampling.enable_thinking,
      enableVision: sampling.enable_vision,
      enableMtp: sampling.enable_mtp,
      reasoningEffort: sampling.reasoning_effort ?? "",
      excludeReasoning: !preset.settings.include_reasoning_history,
      seed: sampling.seed ?? null,
    },
    inheritGlobalSettings:
      typeof preset.metadata?.inherit_global_settings === "boolean"
        ? preset.metadata.inherit_global_settings
        : true,
    contextSize: preset.context_size,
    model: preset.model,
    mode: preset.mode,
    icon: typeof preset.metadata?.icon === "string" ? preset.metadata.icon : undefined,
    updatedAt: preset.updated_at,
  };
}

/** 将本地预设转换为创建或更新服务端资源的请求体。 */
export function presetResourceBody(
  preset: StoredPreset,
  fallbackModel: string,
  fallbackMode: SessionMode,
): Omit<GenerationPresetResource, "id" | "created_at" | "updated_at"> {
  return {
    name: preset.name,
    model: preset.model ?? fallbackModel,
    mode: preset.mode ?? fallbackMode,
    settings: {
      sampling: {
        max_tokens: preset.settings.maxTokens,
        temperature: preset.settings.temperature,
        top_k: preset.settings.topK,
        top_p: preset.settings.topP,
        presence_penalty: preset.settings.presencePenalty,
        frequency_penalty: preset.settings.frequencyPenalty,
        repetition_penalty: preset.settings.repetitionPenalty,
        seed: preset.settings.seed,
        enable_thinking: preset.settings.enableThinking,
        enable_vision: preset.settings.enableVision,
        enable_mtp: preset.settings.enableMtp,
        reasoning_effort: preset.settings.reasoningEffort || null,
      },
      system_prompt: preset.settings.systemPrompt || null,
      include_reasoning_history: !preset.settings.excludeReasoning,
      input_role: "user",
      tools: [],
      tool_choice: "auto",
      response_format: { type: "text" },
    },
    context_size: preset.contextSize,
    metadata: {
      icon: preset.icon || preset.name.slice(0, 1).toLocaleUpperCase(),
      inherit_global_settings: preset.inheritGlobalSettings,
    },
  };
}
