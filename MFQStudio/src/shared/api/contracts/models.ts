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
  downloads: number;
  likes: number;
  total_bytes: number;
  updated_at?: string | null;
}

export interface HubModelInfo extends HubModelSummary {
  revision: string;
  files: Array<{ name: string; byte_size: number }>;
  tags: string[];
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
