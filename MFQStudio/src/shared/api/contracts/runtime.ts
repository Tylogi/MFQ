/** 定义 runtime 领域的服务契约，仅包含类型，不依赖运行时代码。 */
import type { SamplingParams } from './sessions';
import type { ApiErrorBody } from './protocol';

export interface ModelFeatureSet {
  text: boolean;
  image_input: boolean;
  video_input: boolean;
  audio_input: boolean;
  audio_output: boolean;
  full_duplex: boolean;
  mtp: boolean;
}

export interface ModelCapabilities {
  architecture_family: string;
  source: string;
  features: ModelFeatureSet;
}

export interface RuntimeCapabilities {
  model: string;
  model_type: string;
  model_capabilities: ModelCapabilities;
  vision_available: boolean;
  mtp_available: boolean;
  duplex_available: boolean;
}

export interface RuntimeRequestMetrics {
  id?: string;
  endpoint?: string;
  stream?: boolean;
  prompt_tokens?: number;
  prefill_tokens?: number;
  completion_tokens?: number;
  prefill_tps?: number;
  prefill_ms?: number;
  multimodal_ms?: number;
  model_prefill_ms?: number;
  processor_ms?: number;
  complete_prefill_ms?: number;
  complete_prefill_tps?: number;
  decode_tps?: number;
  decode_ms?: number;
  ttft_ms?: number;
  generation_ms?: number;
  complete_generation_ms?: number;
  generation_tps?: number;
  finish_reason?: string;
  client_connected?: boolean;
  completed_at?: number;
}

export interface RuntimeStatus {
  instance_id?: string;
  runtime_state?: string;
  model?: string;
  model_type?: string;
  model_capabilities?: ModelCapabilities;
  duplex_available?: boolean;
  active_requests?: number;
  total_requests?: number;
  failed_requests?: number;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  uptime_seconds?: number;
  max_context?: number;
  context_capacity?: number;
  reloading?: boolean;
  process_resident_bytes?: number | null;
  runtime_memory_budget_bytes?: number | null;
  runtime_memory_effective_budget_bytes?: number | null;
  runtime_memory_budget_mode?: 'automatic' | 'explicit' | 'disabled';
  runtime_memory_committed_bytes?: number;
  runtime_memory_headroom_bytes?: number | null;
  runtime_memory_pressure_level?: 'disabled' | 'normal' | 'soft' | 'hard';
  runtime_memory_pressure_ratio?: number | null;
  runtime_memory_shared_cache_reclaims?: number;
  runtime_memory_shared_cache_released_bytes?: number;
  runtime_memory_shared_cache_reclaim_failures?: number;
  mlx_active_bytes?: number;
  mlx_cache_bytes?: number;
  mlx_peak_bytes?: number;
  cuda_allocated_bytes?: number;
  cuda_reserved_bytes?: number;
  device_free_bytes?: number;
  device_total_bytes?: number;
  prefix_cache_queries?: number;
  prefix_cache_hits?: number;
  prefix_cache_hit_tokens?: number;
  prefix_cache_sessions?: number;
  prefix_cache_snapshots?: number;
  prefix_cache_tokens?: number;
  prefix_cache_bytes?: number;
  prefix_cache_max_sessions?: number;
  prefix_cache_max_snapshots_per_session?: number;
  prefix_cache_max_bytes?: number;
  prefix_cache_disk_blocks?: number;
  prefix_cache_disk_bytes?: number;
  prefix_cache_disk_max_bytes?: number;
  prefix_cache_hot_blocks?: number;
  prefix_cache_hot_bytes?: number;
  prefix_cache_pending_writes?: number;
  prefix_cache_pending_bytes?: number;
  prefix_cache_pending_max_bytes?: number;
  prefix_cache_writes?: number;
  prefix_cache_deduplicated_writes?: number;
  prefix_cache_disk_hits?: number;
  prefix_cache_hot_hits?: number;
  prefix_cache_evictions?: number;
  prefix_cache_corrupt_blocks?: number;
  prefix_cache_mode?: string;
  sampling_defaults?: Partial<SamplingParams>;
  duplex_sampling_defaults?: {
    system_prompt?: string;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    text_repetition_penalty?: number;
    [key: string]: unknown;
  };
  tts_sampling_defaults?: {
    temperature?: number;
    repetition_penalty?: number;
    [key: string]: unknown;
  };
  chat_template_capabilities?: {
    thinking?: { supported?: boolean };
    reasoning_effort?: { supported?: boolean; values?: string[] };
  };
  last_request?: RuntimeRequestMetrics | null;
  [key: string]: unknown;
}

export interface RuntimeModel {
  id: string;
  object?: string;
  owned_by?: string;
}

export interface RuntimeInstance {
  id: string;
  model: string;
  state: 'loading' | 'ready' | 'busy' | 'unloading' | 'failed';
  devices: string[];
  active_sessions: number;
  queued_requests: number;
  resident_bytes?: number | null;
  kv_bytes?: number | null;
  context_size?: number | null;
  started_at?: string | null;
  last_used_at?: string | null;
  idle_ttl_seconds?: number | null;
  pinned?: boolean;
  mtp_supported?: boolean;
  mtp_available?: boolean;
  error?: ApiErrorBody['error'] | null;
}

export interface RuntimeProfile {
  id: string;
  name: string;
  load: {
    model: string;
    artifact_uri?: string | null;
    device_ids: string[];
    idle_ttl_seconds?: number | null;
    pin: boolean;
    context_size: number;
    prefill_chunk_size: number;
    moe_gpu_cache_gb?: number | null;
    prefix_cache_max_sessions?: number | null;
    prefix_cache_max_snapshots_per_session?: number | null;
    prefix_cache_max_bytes?: number | null;
    prefix_cache_enabled?: boolean;
    prefix_cache_disk_bytes?: number | null;
    prefix_cache_hot_bytes?: number | null;
    prefix_cache_block_tokens?: number | null;
    prefix_cache_pending_bytes?: number | null;
    sampling_defaults?: SamplingParams | null;
  };
  artifact_id: string;
  artifact_modified_at: string;
  drifted: boolean;
  drift_reason?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RuntimeMetricSnapshot {
  sequence: number;
  instance_id?: string | null;
  model?: string | null;
  values: RuntimeStatus;
  captured_at: string;
}

export interface RuntimeLogEntry {
  sequence: number;
  instance_id?: string | null;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  fields: Record<string, unknown>;
  created_at: string;
}
