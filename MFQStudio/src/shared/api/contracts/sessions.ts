/** 定义 sessions 领域的服务契约，仅包含类型，不依赖运行时代码。 */
import type { MediaRef } from './media';

export type SessionMode = 'text' | 'voice' | 'full_duplex';

export type SessionState =
  | 'idle'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'interrupted'
  | 'reconnecting'
  | 'error'
  | 'closed';

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'transcript'; text: string; language?: string | null }
  | { type: 'document'; media: MediaRef; name: string }
  | { type: 'image'; media: MediaRef; width?: number | null; height?: number | null }
  | {
      type: 'video';
      media: MediaRef;
      width?: number | null;
      height?: number | null;
      duration_ms?: number | null;
    }
  | {
      type: 'audio' | 'generated_audio';
      media: MediaRef;
      sample_rate_hz: number;
      channels: number;
      duration_ms?: number | null;
    }
  | { type: 'tool_call'; call_id: string; name: string; arguments: Record<string, unknown> }
  | { type: 'tool_result'; call_id: string; result: unknown; is_error: boolean };

export interface Session {
  id: string;
  model: string;
  mode: SessionMode;
  state: SessionState;
  revision: number;
  title: string | null;
  runtime_instance_id: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface SessionArchive {
  format: 'mfq-session-v1';
  session: Session;
  messages: Array<{ role: MessageRole; parts: ContentPart[]; created_at: string }>;
  media: Array<{
    sha256: string;
    mime_type: string;
    data_base64: string;
    document?: Record<string, unknown> | null;
  }>;
}

export interface Message {
  id: string;
  role: MessageRole;
  parts: ContentPart[];
  parent_id: string | null;
  created_at: string;
}

export interface SamplingParams {
  max_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  presence_penalty: number;
  frequency_penalty: number;
  repetition_penalty: number;
  seed?: number | null;
  enable_thinking: boolean;
  enable_vision: boolean;
  enable_mtp: boolean;
  mtp_max_draft_tokens?: number;
  reasoning_effort?: string | null;
}

export interface ResponsePerformance {
  prefill_tokens: number;
  ttft_ms: number;
  prefill_ms: number;
  prefill_tps: number;
  multimodal_ms: number;
  model_prefill_ms: number;
  processor_ms: number;
  complete_prefill_ms: number;
  complete_prefill_tps: number;
  decode_ms: number;
  decode_tps: number;
  generation_ms: number;
  complete_generation_ms: number;
  generation_tps: number;
  mtp_available?: boolean;
  mtp_used?: boolean;
  mtp_cycles?: number;
  mtp_drafted_tokens?: number;
  mtp_accepted_tokens?: number;
  mtp_acceptance_rate?: number;
  mtp_selected_depth?: number;
  mtp_depth_0_cycles?: number;
  mtp_depth_1_cycles?: number;
  mtp_depth_2_cycles?: number;
  mtp_depth_3_cycles?: number;
  mtp_depth_4_cycles?: number;
  mtp_depth_5_cycles?: number;
  mtp_position_1_acceptance_rate?: number;
  mtp_position_2_acceptance_rate?: number;
  mtp_position_3_acceptance_rate?: number;
  mtp_position_4_acceptance_rate?: number;
  mtp_position_5_acceptance_rate?: number;
  mtp_depth_0_cycle_ms?: number;
  mtp_depth_1_cycle_ms?: number;
  mtp_depth_2_cycle_ms?: number;
  mtp_depth_3_cycle_ms?: number;
  mtp_depth_4_cycle_ms?: number;
  mtp_depth_5_cycle_ms?: number;
  mtp_target_ms?: number;
  mtp_head_ms?: number;
  mtp_rollback_ms?: number;
  sampling: SamplingParams;
}

export interface ResponseResource {
  id: string;
  request_id: string;
  session_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  output_message_id?: string | null;
  output: ContentPart[];
  finish_reason?: string | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
  performance?: ResponsePerformance | null;
  settings?: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
  } | null;
  created_at: string;
  completed_at?: string | null;
}

export interface GenerationPresetResource {
  id: string;
  name: string;
  model?: string | null;
  mode?: SessionMode | null;
  settings: {
    sampling: SamplingParams;
    system_prompt?: string | null;
    include_reasoning_history: boolean;
    input_role: 'user' | 'tool';
    tools: Array<Record<string, unknown>>;
    tool_choice: 'auto' | 'none' | 'required' | Record<string, unknown>;
    response_format: { type: 'text' | 'json_object' | 'json_schema'; [key: string]: unknown };
  };
  context_size: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface StreamRequest {
  request_id: string;
  expected_revision: number;
  input: ContentPart[];
  input_role?: 'user' | 'tool';
  sampling: SamplingParams;
  system_prompt?: string | null;
  include_reasoning_history: boolean;
  tools?: Array<{
    type: 'function';
    function: {
      name: string;
      description?: string | null;
      parameters: Record<string, unknown>;
    };
  }>;
  tool_choice?: 'auto' | 'none' | 'required';
  stream: true;
}
