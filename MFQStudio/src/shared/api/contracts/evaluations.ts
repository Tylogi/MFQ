/** 定义 evaluations 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface DatasetResource {
  id: string;
  name: string;
  kind: 'wikitext2' | 'custom';
  artifact_uri: string;
  sha256: string;
  byte_size: number;
  source_uri?: string | null;
  revision?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface EvaluationResult {
  id: string;
  job_id: string;
  kind: 'perplexity' | 'kernel_benchmark';
  model_id: string;
  metrics: Record<string, unknown>;
  parameters: Record<string, unknown>;
  dataset_id?: string | null;
  dataset_manifest: Record<string, unknown>;
  hardware_identity: Record<string, unknown>;
  runtime_identity: Record<string, unknown>;
  comparison_key: string;
  created_at: string;
}

export interface EvaluationComparison {
  comparison_key: string;
  baseline_id: string;
  metrics: string[];
  rows: Array<{
    evaluation: EvaluationResult;
    deltas: Record<string, number | null>;
    ratios: Record<string, number | null>;
  }>;
}
