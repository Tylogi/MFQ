/** 定义 models 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface ModelArtifact {
  id: string;
  name: string;
  architecture: string;
  format: 'mfq' | 'hf';
  shard_count: number;
  total_bytes: number;
  tensor_count: number;
  record_count: number;
  dtypes: string[];
  complete: boolean;
  loadable: boolean;
  modified_at: string;
  error?: string | null;
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

export interface HubModelInfo extends HubModelSummary {
  revision: string;
  files: Array<{ name: string; byte_size: number; sha256?: string | null }>;
  tags: string[];
  license?: string | null;
  library?: string | null;
  pipeline_tag?: string | null;
  architectures: string[];
  modalities: string[];
  parameter_count?: number | null;
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
