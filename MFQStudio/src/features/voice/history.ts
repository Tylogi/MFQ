/** 定义语音消息与实时输出结构，并读取本地语音历史。 */


export interface VoiceMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  text: string;
  audioId?: string;
  pending?: boolean;
  created_at: string;
}

export interface LiveVoiceOutput {
  sessionId: string;
  text: string;
}

export const VOICE_HISTORY_KEY = "mfq.studio.voice-history.v1";

/** 读取最多两百条本地语音记录，解析失败时返回空列表。 */
export function loadVoiceHistory(): VoiceMessage[] {
  try {
    const value = JSON.parse(localStorage.getItem(VOICE_HISTORY_KEY) || "[]");
    return Array.isArray(value) ? value.slice(-200) : [];
  } catch {
    return [];
  }
}
