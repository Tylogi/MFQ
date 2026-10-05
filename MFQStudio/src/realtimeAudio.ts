/** Preserve the legacy real-time voice import path; implementation is split by voice-module responsibility. */
export { RealtimeAudioController } from './features/voice/RealtimeAudioController';
export { deleteVoiceClip, loadVoiceClip, pruneVoiceClips, saveVoiceClip } from './features/voice/clipStorage';
export type { RealtimeCallbacks, RealtimeSessionConfig, VoiceInputTurn, VoiceState, VoiceTurn } from './features/voice/realtimeTypes';
