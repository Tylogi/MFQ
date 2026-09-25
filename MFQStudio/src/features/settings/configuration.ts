/** 管理界面与推理设置的默认值、本地读取及模型模式继承规则。 */
import { RealtimeCapabilities, RuntimeStatus, SessionMode } from '../../shared/api/types';

export type UiLanguage = "system" | "zh-CN" | "en";

export type UiTheme = "system" | "light" | "dark";

export type PresetName = "precise" | "balanced" | "creative" | "custom";

export interface GenerationSettings {
  language: UiLanguage;
  theme: UiTheme;
  inheritModelDefaults: boolean;
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
  enableThinking: boolean;
  enableVision: boolean;
  enableMtp: boolean;
  reasoningEffort: string;
  excludeReasoning: boolean;
  playbackEnabled: boolean;
  fullDuplex: boolean;
  preset: PresetName;
  seed: number | null;
}

export const SETTINGS_KEY = "mfq.studio.generation.v1";

export const DEFAULT_SETTINGS: GenerationSettings = {
  language: "system",
  theme: "system",
  inheritModelDefaults: true,
  systemPrompt: "",
  maxTokens: 4096,
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  enableThinking: true,
  enableVision: true,
  enableMtp: true,
  reasoningEffort: "",
  excludeReasoning: false,
  playbackEnabled: true,
  fullDuplex: true,
  preset: "balanced",
  seed: null,
};

export const PRESETS: Record<Exclude<PresetName, "custom">, Partial<GenerationSettings>> = {
  precise: { temperature: 0.2, topP: 0.75, topK: 20, repetitionPenalty: 1.05 },
  balanced: { temperature: 0.7, topP: 0.8, topK: 20, repetitionPenalty: 1 },
  creative: { temperature: 1, topP: 0.95, topK: 50, repetitionPenalty: 1 },
};

/** 根据会话模式和服务默认值生成推理设置，不修改输入对象。 */
export function modeTemplateSettings(
  current: GenerationSettings,
  mode: SessionMode,
  runtime: RuntimeStatus | null,
  realtime: RealtimeCapabilities | null,
): GenerationSettings {
  const voice = mode !== "text";
  const defaults = (voice
    ? (runtime?.duplex_sampling_defaults ?? realtime?.defaults ?? {})
    : (runtime?.sampling_defaults ?? {})) as Record<string, unknown>;
  const value = (key: string, fallback: number): number => {
    const candidate = Number(defaults[key]);
    return Number.isFinite(candidate) ? candidate : fallback;
  };
  return {
    ...current,
    systemPrompt: current.systemPrompt,
    maxTokens: voice ? current.maxTokens : value("max_tokens", current.maxTokens),
    temperature: value("temperature", current.temperature),
    topP: value("top_p", current.topP),
    topK: value("top_k", current.topK),
    repetitionPenalty: value(
      voice ? "text_repetition_penalty" : "repetition_penalty",
      current.repetitionPenalty,
    ),
    presencePenalty: voice ? 0 : value("presence_penalty", current.presencePenalty),
    frequencyPenalty: voice ? 0 : value("frequency_penalty", current.frequencyPenalty),
    enableThinking: voice
      ? false
      : typeof defaults.enable_thinking === "boolean"
        ? defaults.enable_thinking
        : current.enableThinking,
    enableVision:
      typeof defaults.enable_vision === "boolean"
        ? defaults.enable_vision
        : current.enableVision,
    enableMtp:
      typeof defaults.enable_mtp === "boolean"
        ? defaults.enable_mtp
        : current.enableMtp,
    fullDuplex: mode === "full_duplex",
    preset: "custom",
    seed: null,
  };
}

/** 读取本地界面及推理设置，损坏或缺失时使用默认值。 */
export function loadSettings(): GenerationSettings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}
