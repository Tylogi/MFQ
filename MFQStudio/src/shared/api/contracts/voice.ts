/** Define service contracts for the voice domain using types only, with no runtime dependencies. */
import type { ModelCapabilities } from './runtime';

export interface RealtimeCapabilities {
  available: boolean;
  input?: string[];
  output?: string[];
  input_sample_rate?: number;
  output_sample_rate?: number;
  defaults?: Record<string, unknown>;
  model_capabilities?: ModelCapabilities;
}

export interface VoiceOutputComponentStatus {
  id: string;
  state: 'missing' | 'installing' | 'ready';
  ready: boolean;
  installed_bytes: number;
  total_bytes: number;
  repository: string;
  revision: string;
  active: boolean;
  supported_model_loaded: boolean;
  model?: string | null;
  error?: string | null;
}
