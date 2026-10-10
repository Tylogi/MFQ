/** 定义 models 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface ModelArtifact {
  id: string;
  name: string;
  architecture: string;
  format: 'mfq' | 'hf';
  shard_count: number;
  missing_shards?: number;
  total_bytes: number;
  estimated_resident_weight_bytes?: number | null;
  ssd_ple_bytes?: number | null;
  context_capacity?: number | null;
  tensor_count: number;
  record_count: number;
  dtypes: string[];
  complete: boolean;
  loadable: boolean;
  modified_at: string;
  error?: string | null;
}

export interface AnalysisProjection {
  name: string;
  category: string;
  parameters: number;
  stored_bytes: number;
  average_bpw: number | null;
  tensor_count: number;
  formats: string[];
}

export interface AnalysisLayer {
  layer: number;
  attention_type: string;
  parameters: number;
  stored_bytes: number;
  average_bpw: number | null;
}

export interface AnalysisExpert {
  layer: number;
  expert: number;
  projection: string;
  parameters: number;
  stored_bytes: number;
  bpw: number;
  format: string;
  granularity: string;
}

export interface AnalysisTensor {
  name: string;
  shape: number[];
  category: string;
  projection: string;
  layer: number | null;
  parameters: number;
  stored_bytes: number;
  bpw: number | null;
  format: string;
}

export interface AnalysisNormalization {
  component: string;
  position: string;
  kind?: string | null;
  dimension?: number | null;
  epsilon?: number | null;
  groups?: number | null;
  layers: number[];
}

export interface AnalysisAdvanced {
  ffn_hidden_size?: number | null;
  expert_hidden_size?: number | null;
  shared_expert_count?: number | null;
  shared_expert_hidden_size?: number | null;
  hc_count?: number | null;
  hc_dim?: number | null;
  hc_lowrank?: number | null;
  ffn_activation?: string | null;
  router_activation?: string | null;
  router_normalization?: string | null;
  gdn_conv_activation?: string | null;
  gdn_conv_type?: string | null;
  gdn_conv_stride?: number | null;
  gdn_conv_dilation?: number | null;
  gdn_gate_activation?: string | null;
  qsa_gate_activation?: string | null;
  full_attention_gate_activation?: string | null;
  residual_gate_activation?: string | null;
  indexer_query_heads?: number | null;
  indexer_kv_heads?: number | null;
  indexer_head_dim?: number | null;
  compression_stride?: number | null;
  compression_strides?: number[];
  active_tokens?: number | null;
  active_blocks?: number | null;
  linear_key_head_dim?: number | null;
  linear_value_head_dim?: number | null;
  linear_conv_kernel?: number | null;
  ple_ngram?: number | null;
  ple_heads?: number | null;
  ple_conv_kernel?: number | null;
  ple_conv_type?: string | null;
  ple_conv_stride?: number | null;
  ple_conv_dilation?: number | null;
  ple_conv_activation?: string | null;
  ple_layer_ids?: number[];
  predictor_layers?: number | null;
  max_context?: number | null;
  vocab_size?: number | null;
  rope_theta?: number | null;
  rope_partial_factor?: number | null;
  rms_norm_eps?: number | null;
  normalizations?: AnalysisNormalization[];
}

export interface CheckpointAnalysis {
  model_id: string;
  name: string;
  architecture: string;
  format: string;
  complete: boolean;
  layer_count: number;
  hidden_size: number | null;
  attention_heads: number | null;
  kv_heads: number | null;
  head_dim: number | null;
  linear_key_heads: number | null;
  linear_value_heads: number | null;
  expert_count: number;
  experts_per_token: number | null;
  parameters: number;
  stored_bytes: number;
  average_bpw: number | null;
  attention_distribution: Record<string, number[]>;
  projections: AnalysisProjection[];
  layers: AnalysisLayer[];
  experts: AnalysisExpert[];
  tensors: AnalysisTensor[];
  graph: Record<string, unknown>;
  advanced?: AnalysisAdvanced;
  cache_profile?: ModelCacheProfile | null;
  warnings: string[];
}

export interface ModelDirectoryEntry {
  id: string;
  name: string;
  model_file_count: number;
}

export interface ModelDirectoryList {
  current_id: string | null;
  current_name: string | null;
  current_path: string | null;
  parent_id: string | null;
  model_file_count: number;
  files?: Array<{ name: string; byte_size: number }>;
  can_open_in_finder?: boolean;
  data: ModelDirectoryEntry[];
}

export interface HubModelSummary {
  provider: 'huggingface' | 'modelscope';
  repo_id: string;
  source_url?: string | null;
  author?: string | null;
  description?: string | null;
  downloads: number;
  likes: number;
  total_bytes: number;
  updated_at?: string | null;
}

export interface ModelConfigurationStatus {
  status: 'recommended' | 'warning' | 'unknown';
  recommendation: 'three_stars' | 'two_stars' | 'one_star' | 'caution' | 'not_recommended' | 'unknown';
  required_memory_bytes?: number | null;
  available_memory_bytes?: number | null;
  reasons: string[];
}

export interface HubModelVariant {
  id: string;
  label: string;
  format: 'mfq' | 'hf' | 'gguf' | 'unknown';
  precision?: string | null;
  files: string[];
  byte_size: number;
  resident_weight_bytes?: number | null;
  estimated_resident_weight_bytes?: number | null;
  ssd_ple_bytes?: number | null;
  configuration: ModelConfigurationStatus;
}

export interface HubMemoryPool {
  kind: 'uma' | 'vram' | 'ram';
  device?: string | null;
  capacity_bytes?: number | null;
  bandwidth_bytes_per_second?: number | null;
}

export interface HubSystemProfile {
  platform: string;
  machine: string;
  backend: 'metal' | 'cuda' | 'rocm' | 'cpu' | 'unknown';
  cpu_name?: string | null;
  cpu_cores?: number | null;
  gpu_names?: string[];
  gpu_cores?: number | null;
  physical_memory_bytes?: number | null;
  available_memory_bytes?: number | null;
  runtime_memory_budget_bytes?: number | null;
  memory_pools?: HubMemoryPool[];
}

export interface ModelParameterBreakdown {
  total: number;
  dense: number;
  routed_experts: number;
  ple: number;
  active?: number | null;
}

export interface ModelCacheProfile {
  max_context: number;
  fixed_bytes: number;
  fixed_components?: Array<{ group: string; name: string; layers?: number | null; bytes: number }>;
  components: Array<{
    group?: string | null;
    name?: string | null;
    subgroup?: string | null;
    layers?: number | null;
    bytes_per_row: number;
    tokens_per_row: number;
    allocation: 'exact' | 'power_of_two';
    minimum_rows: number;
    row_rounding?: 'ceil' | 'floor';
    active_after?: number;
    max_read_rows_per_token?: number | null;
    head_dimension?: number | null;
    kv_heads?: number | null;
  }>;
}

export interface HubModelInfo extends HubModelSummary {
  revision: string;
  files: Array<{ name: string; byte_size: number; sha256?: string | null; weight_bytes?: number | null; weight_bytes_by_dtype?: Record<string, number>; ssd_ple_bytes?: number | null }>;
  tags: string[];
  license?: string | null;
  library?: string | null;
  pipeline_tag?: string | null;
  architectures: string[];
  modalities: string[];
  parameter_count?: number | null;
  parameter_breakdown?: ModelParameterBreakdown | null;
  mtp_supported?: boolean | null;
  cache_profile?: ModelCacheProfile | null;
  ple_parameter_count?: number | null;
  published_at?: string | null;
  gated: boolean;
  runtime_compatible?: boolean | null;
  variants: HubModelVariant[];
}

export interface OfficialModelSource {
  provider: HubModelSummary['provider'];
  repo_id: string;
  revision?: string | null;
  url: string;
  available: boolean;
}

export interface OfficialModelInfo {
  id: string;
  name: string;
  family: string;
  architecture: string;
  description: string;
  description_zh: string;
  parameter_label?: string | null;
  active_parameter_label?: string | null;
  parameter_breakdown?: ModelParameterBreakdown | null;
  mtp_supported?: boolean | null;
  cache_profile?: ModelCacheProfile | null;
  modalities: string[];
  capabilities: string[];
  precision_options: string[];
  license?: string | null;
  supports_ssd_streaming: boolean;
  sources: OfficialModelSource[];
  selected_source: OfficialModelSource;
  revision: string;
  downloads: number;
  likes: number;
  updated_at?: string | null;
  variants: HubModelVariant[];
  published_at?: string | null;
  configuration: ModelConfigurationStatus;
}

export interface OfficialModelList {
  system: HubSystemProfile;
  data: OfficialModelInfo[];
  refreshing?: boolean;
}

export interface ArtifactLineage {
  id: string;
  artifact_uri: string;
  artifact_name: string;
  producer_job_id: string;
  producer_kind: string;
  source_uris: string[];
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  validation_job_ids: string[];
  created_at: string;
}
