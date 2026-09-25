/** 保留实时语音旧导入路径，实际实现按语音模块职责拆分。 */
export { RealtimeAudioController } from './features/voice/RealtimeAudioController';
export { loadVoiceClip, saveVoiceClip } from './features/voice/clipStorage';
export type { RealtimeCallbacks, RealtimeSessionConfig, VoiceInputTurn, VoiceState, VoiceTurn } from './features/voice/realtimeTypes';
