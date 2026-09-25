/** 定义实时语音会话、输入输出轮次及业务回调契约。 */
/** 实时语音连接与录音处理阶段，供页面呈现当前状态。 */
export type VoiceState = "idle" | "connecting" | "listening" | "processing" | "error";

/** 建立实时会话时传给服务端的生成配置。 */
export interface RealtimeSessionConfig {
  sessionId: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
}

/** 已完成或由迟到数据更新的助手语音轮次。 */
export interface VoiceTurn {
  id: string;
  sessionId: string;
  text: string;
  audio: Blob | null;
}

/** 用户录音轮次；开始时尚无音频，结束时返回 WAV。 */
export interface VoiceInputTurn {
  id: string;
  sessionId: string;
  audio?: Blob | null;
}

/** 控制器内部维护的文本和 PCM 音频累积结果。 */
export interface BufferedVoiceTurn {
  id: string;
  sessionId: string;
  inputTurnId: string | null;
  text: string;
  audio: Float32Array[];
  audioRate: number;
}

/** 控制器向业务层发布会话、音频和错误事件的回调契约。 */
export interface RealtimeCallbacks {
  /** 连接、录音或处理状态改变时同步界面状态。 */
  onState(state: VoiceState): void;
  /** 采集音频时提供归一化音量，停止时归零。 */
  onLevel(level: number): void;
  /** 增量更新指定会话的助手文本，轮次发布后清空。 */
  onText(sessionId: string, text: string): void;
  /** 检测到用户新轮次时建立录音消息占位。 */
  onInputStart(turn: VoiceInputTurn): void;
  /** 结束用户轮次并提供可保存的 WAV 音频。 */
  onInputEnd(turn: VoiceInputTurn): void;
  /** 发布助手轮次或更新迟到片段，由业务层持久化。 */
  onTurn(turn: VoiceTurn): void;
  /** 连接或处理失败时通知业务层展示错误。 */
  onError(message: string): void;
}
