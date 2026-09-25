/** 定义 voice 领域的服务契约，仅包含类型，不依赖运行时代码。 */
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
