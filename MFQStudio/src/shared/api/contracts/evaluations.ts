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
  kind: 'perplexity' | 'kernel_benchmark' | 'inference_benchmark' | 'accuracy_benchmark';
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

export interface BenchmarkParameters {
  protocol: 'mcq' | 'math' | 'code' | 'likelihood';
  sample_count: number;
  max_tokens: number | null;
  seed: number | null;
  enable_thinking: boolean | null;
  enable_mtp: boolean;
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  num_generations: number;
}

export interface OfficialBenchmarkReadiness {
  available: boolean;
  reason?: string;
  repository: string;
  revision: string;
  scoring_files: Record<string, string>;
  protocol: string;
  defaults?: { available: boolean; parameters: Partial<BenchmarkParameters>; reason?: string; source: string };
}

export interface EvaluationTools {
  workspace_root: string | null;
  api_base: string;
  quality_available: boolean;
  benchmark_available: boolean;
  accuracy_available: boolean;
  task_benchmarks?: Record<string, OfficialBenchmarkReadiness>;
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

export interface OfficialDataset {
  id: string;
  name: string;
  kind: DatasetResource['kind'];
  repository: string;
  revision: string;
  filename: string;
  sha256: string;
  byte_size: number;
  rows: number;
  license: string;
  format?: string;
  split?: string;
  task?: string | null;
  origin?: 'huggingface' | 'github';
}
