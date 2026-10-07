import type { AnalysisExpert, AnalysisTensor, CheckpointAnalysis, ModelArtifact } from '../../src/shared/api/types';

export const analysisArtifacts: ModelArtifact[] = [
  { id: 'a'.repeat(32), name: 'Qwen3.8-Flash-Next-EWQ-MFQ-S4-L', architecture: 'qwen4_exp', format: 'mfq', shard_count: 6,
    total_bytes: 110e9, tensor_count: 1342, record_count: 1348, dtypes: ['MFE', 'NINT'], complete: true, loadable: true, modified_at: '2026-10-01T00:00:00Z' },
  { id: 'b'.repeat(32), name: 'Dense checkpoint', architecture: 'qwen3_5', format: 'mfq', shard_count: 1,
    total_bytes: 18e9, tensor_count: 900, record_count: 905, dtypes: ['NINT'], complete: true, loadable: true, modified_at: '2026-10-01T00:00:00Z' },
  { id: 'c'.repeat(32), name: 'Incomplete checkpoint', architecture: 'unknown', format: 'mfq', shard_count: 6, missing_shards: 1,
    total_bytes: 90e9, tensor_count: 0, record_count: 0, dtypes: [], complete: false, loadable: false, modified_at: '2026-10-01T00:00:00Z' },
];

export function analysisFixture(dense = false): CheckpointAnalysis {
  const experts: AnalysisExpert[] = dense ? [] : Array.from({ length: 12 }, (_, layer) => Array.from({ length: 32 }, (_, expert) =>
    ['gate', 'up', 'down'].map((projection, index) => {
      const bpw = 2 + ((layer + expert + index) % 6);
      return { layer, expert, projection, parameters: 1000, stored_bytes: bpw * 125, bpw,
        format: bpw < 3 ? 'NVQ3J-L' : 'NINTv2', granularity: 'row' };
    }))).flat(2);
  const tensors: AnalysisTensor[] = Array.from({ length: 12 }, (_, layer) => {
    const qsa = layer % 4 === 3;
    const bpw = qsa ? 6.4 : 5.2;
    return { name: `model.block.${layer}.${qsa ? 'attention.query' : 'linear_attention.qkv'}.weight`,
      shape: [128, 256], category: qsa ? 'attention' : 'gdn', projection: qsa ? 'query' : 'qkv', layer,
      parameters: 32768, stored_bytes: bpw * 4096, bpw, format: 'NINTv2' };
  });
  return { model_id: analysisArtifacts[dense ? 1 : 0].id, name: analysisArtifacts[dense ? 1 : 0].name,
    architecture: dense ? 'qwen3_5' : 'qwen4_exp', format: 'mfq', complete: true, layer_count: 12,
    hidden_size: 2560, attention_heads: 24, kv_heads: 2, head_dim: 256, linear_key_heads: 16, linear_value_heads: 48,
    expert_count: dense ? 0 : 32, experts_per_token: dense ? null : 4, parameters: 179e9,
    stored_bytes: 112e9, average_bpw: 5.006,
    cache_profile: { max_context: 262144, fixed_bytes: 0, fixed_components: [], components: [
      { group: dense ? 'GQA' : 'QSA', name: 'raw_kv', layers: 3, bytes_per_row: 3 * 2 * 256 * 4,
        tokens_per_row: 1, allocation: 'power_of_two', minimum_rows: 16 },
      ...(dense ? [] : [
        { group: 'QSA', name: 'indexer_key', subgroup: 'indexer', layers: 3, bytes_per_row: 3 * 128 * 2,
          tokens_per_row: 1, allocation: 'power_of_two' as const, minimum_rows: 16 },
        { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', layers: 3, bytes_per_row: 3 * 128 * 4,
          tokens_per_row: 4, allocation: 'power_of_two' as const, minimum_rows: 16, row_rounding: 'floor' as const, active_after: 2048 },
      ]),
    ] },
    attention_distribution: { GDN: [0, 1, 2, 4, 5, 6, 8, 9, 10], [dense ? 'GQA' : 'QSA']: [3, 7, 11] },
    projections: [
      { name: 'query', category: 'attention', parameters: 98304, stored_bytes: 78643, average_bpw: 6.4, tensor_count: 3, formats: ['NINTv2'] },
      { name: 'qkv', category: 'gdn', parameters: 294912, stored_bytes: 191693, average_bpw: 5.2, tensor_count: 9, formats: ['NINTv2'] },
      { name: 'gate', category: 'routed_experts', parameters: 384000, stored_bytes: 216000, average_bpw: 4.5, tensor_count: 12, formats: ['NINTv2', 'NVQ3J-L'] },
    ],
    layers: Array.from({ length: 12 }, (_, layer) => ({ layer, attention_type: layer % 4 === 3 ? dense ? 'GQA' : 'QSA' : 'GDN',
      parameters: 1e9, stored_bytes: (0.4 + layer / 20) * 2 ** 30, average_bpw: 4 + layer / 10 })),
    advanced: { ffn_hidden_size: dense ? 17408 : null, expert_hidden_size: dense ? null : 640,
      shared_expert_count: dense ? 0 : 1, shared_expert_hidden_size: dense ? null : 640,
      hc_count: dense ? null : 4, hc_dim: dense ? null : 2560, hc_lowrank: dense ? null : 320,
      ffn_activation: 'silu', router_activation: dense ? null : 'softmax', router_normalization: dense ? null : 'l1_topk',
      gdn_conv_activation: 'silu', gdn_gate_activation: dense ? 'silu' : 'sigmoid',
      gdn_conv_type: 'causal_depthwise_1d', gdn_conv_stride: 1, gdn_conv_dilation: 1,
      qsa_gate_activation: dense ? null : 'sigmoid', full_attention_gate_activation: dense ? 'sigmoid' : null,
      residual_gate_activation: dense ? null : 'sigmoid',
      indexer_query_heads: dense ? null : 4, indexer_kv_heads: dense ? null : 1, indexer_head_dim: dense ? null : 128,
      compression_stride: dense ? null : 4, active_tokens: dense ? null : 2048, active_blocks: dense ? null : 512,
      linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel: 4,
      ple_ngram: dense ? null : 3, ple_heads: dense ? null : 8, ple_conv_kernel: dense ? null : 4,
      ple_conv_type: dense ? null : 'causal_depthwise_1d', ple_conv_stride: dense ? null : 1,
      ple_conv_dilation: dense ? null : 3, ple_conv_activation: dense ? null : 'silu',
      ple_layer_ids: dense ? [] : [1], predictor_layers: dense ? 0 : 1,
      max_context: 262144, vocab_size: 248320, rope_theta: 10000000, rope_partial_factor: 0.25, rms_norm_eps: 1e-6,
      normalizations: [
        { component: 'gdn', position: 'query', kind: 'L2Norm', dimension: 128, epsilon: 1e-6, groups: 16, layers: [0, 1, 2] },
        { component: 'gdn', position: 'output', kind: 'GroupedRMSNorm', dimension: 128, epsilon: 1e-6, groups: 48, layers: [0, 1, 2] },
        ...(dense ? [
          { component: 'attention', position: 'query', kind: 'GroupedRMSNorm', dimension: 256, epsilon: 1e-6, groups: 24, layers: [3, 7, 11] },
          { component: 'attention', position: 'key', kind: 'GroupedRMSNorm', dimension: 256, epsilon: 1e-6, groups: 2, layers: [3, 7, 11] },
        ] : []),
        ...(dense ? [] : [
          { component: 'qsa', position: 'query', kind: 'GroupedRMSNorm', dimension: 256, epsilon: 1e-6, groups: 24, layers: [3, 7, 11] },
          { component: 'indexer', position: 'query', kind: 'GroupedRMSNorm', dimension: 128, epsilon: 1e-6, groups: 4, layers: [3, 7, 11] },
          { component: 'residual', position: 'attention_input', kind: 'GroupedRMSNorm', dimension: 2560, epsilon: 1e-6, groups: 4, layers: [0, 1, 2] },
          { component: 'ple', position: 'conv', kind: 'GroupedRMSNorm', dimension: 2560, epsilon: 1e-6, groups: 4, layers: [1] },
          { component: 'predictor', position: 'block.0.attention.indexer.query_norm', kind: 'GroupedRMSNorm', dimension: 128, epsilon: 1e-6, groups: 4, layers: [] },
        ]),
      ] },
    experts, tensors, graph: { hc_count: dense ? null : 4, has_ple: !dense, has_shared_expert: !dense,
      optional_components: { predictor: !dense } }, warnings: [],
  };
}
