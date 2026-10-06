/** Define real-time voice session, input/output turn, and business callback contracts. */
import type { RealtimeCapabilities } from '../../shared/api/types';
/** Real-time voice connection and recording phases used to display the current page state. */
export type VoiceState = "idle" | "connecting" | "listening" | "processing" | "error";

/** Generation configuration sent to the server when establishing a real-time session. */
export interface RealtimeSessionConfig {
  /** Advertised protocol capabilities, checked before opening audio resources. */
  capabilities?: RealtimeCapabilities | null;
  sessionId: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
}

/** An assistant voice turn that is complete or updated by late-arriving data. */
export interface VoiceTurn {
  id: string;
  sessionId: string;
  text: string;
  audio: Blob | null;
}

/** A user recording turn; it has no audio at start and returns WAV when complete. */
export interface VoiceInputTurn {
  id: string;
  sessionId: string;
  audio?: Blob | null;
}

/** Accumulated text and PCM audio maintained internally by the controller. */
export interface BufferedVoiceTurn {
  id: string;
  sessionId: string;
  inputTurnId: string | null;
  text: string;
  audio: Float32Array[];
  audioRate: number;
}

/** Callback contract for publishing session, audio, and error events from the controller to the business layer. */
export interface RealtimeCallbacks {
  /** Synchronize interface state when connection, recording, or processing status changes. */
  onState(state: VoiceState): void;
  /** Provide normalized volume during capture and reset it to zero when stopped. */
  onLevel(level: number): void;
  /** Incrementally update assistant text for the specified session and clear it after publishing the turn. */
  onText(sessionId: string, text: string): void;
  /** Create a recording-message placeholder when a new user turn is detected. */
  onInputStart(turn: VoiceInputTurn): void;
  /** End a user turn and provide WAV audio that can be saved. */
  onInputEnd(turn: VoiceInputTurn): void;
  /** Publish an assistant turn or update a late fragment for persistence by the business layer. */
  onTurn(turn: VoiceTurn): void;
  /** Notify the business layer to display an error when connection or processing fails. */
  onError(message: string): void;
}
