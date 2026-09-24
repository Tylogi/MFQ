from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
BATCHING = (
    ROOT
    / "cpp_runtime"
    / "backends"
    / "cuda"
    / "runtime"
    / "qwen_continuous_batching.h"
).read_text(encoding="utf-8")
ROPE = (ROOT / "mfq" / "kernels" / "cuda" / "rope.cu").read_text(
    encoding="utf-8"
)
ATTENTION = (ROOT / "mfq" / "kernels" / "cuda" / "attention.cu").read_text(
    encoding="utf-8"
)
ATTENTION_MMA = (
    ROOT / "mfq" / "kernels" / "cuda" / "attention_mma.cu"
).read_text(encoding="utf-8")
KV_CACHE = (ROOT / "mfq" / "kernels" / "cuda" / "kv_cache.cu").read_text(
    encoding="utf-8"
)
QWEN_LOADER = (
    CUDA_ROOT / "models" / "qwen35" / "qwen35_causal_lm.cpp"
).read_text(encoding="utf-8")
QWEN_CONFIG = (
    ROOT / "cpp_runtime" / "core" / "models" / "qwen35.cpp"
).read_text(encoding="utf-8")
CAUSAL_LM = (CUDA_ROOT / "runtime" / "causal_lm.h").read_text(
    encoding="utf-8"
)


def test_continuous_batching_is_an_explicit_server_mode():
    assert '"--continuous-batching"' in DECODE
    assert '"--check-continuous-batching"' in DECODE
    assert "CudaContinuousBatcher" in DECODE
    assert "decode=target_only mtp=disabled" in DECODE


def test_cuda_server_prefill_is_bounded_for_serial_mtp_and_batched_paths():
    assert '"--prefill-chunk-size"' in DECODE
    assert "prefill_tail(" in DECODE
    assert "hidden_forward_chunked(" in DECODE
    assert "prefill_chunk_size_" in BATCHING
    assert "std::optional<QwenBatchState> prefill_state" in BATCHING
    assert "advance_prefills(incoming, contended)" in BATCHING
    assert "request->prefill_offset += count" in BATCHING
    assert "active_.size() + prefilling_.size()" in BATCHING
    assert "continuous_batching_prefill_chunks" in BATCHING
    assert "continuous_batching_prefill_yields" in BATCHING


def test_scheduler_supports_dynamic_join_retire_and_per_request_sampling():
    assert "take_qwen_batch_state" in BATCHING
    assert "restore_qwen_batch_states" in BATCHING
    assert "compact_qwen_batch_state" in BATCHING
    assert "mfq::cuda::sample_logits" in BATCHING
    assert "request->sampler" in BATCHING
    assert "request->token_constraint" in BATCHING
    assert "pending_" in BATCHING
    assert "active_" in BATCHING
    assert "output_tokens" in BATCHING
    assert "publish_token" in BATCHING
    assert "cancel_requested" in BATCHING
    assert "retire_cancelled_requests" in BATCHING
    assert "MFQ_CONTINUOUS_BATCH_GREEDY" in BATCHING
    assert "environment == nullptr || std::atoi(environment) != 0" in BATCHING
    assert "continuous_batching_batched_greedy_batches" in BATCHING
    assert "sample_greedy_cuda(" in BATCHING
    assert "MFQ_CONTINUOUS_BATCH_PACKED_METADATA" in BATCHING
    assert BATCHING.count(
        "environment == nullptr || std::atoi(environment) != 0"
    ) >= 3
    assert "continuous_batching_packed_metadata_batches" in BATCHING
    assert "ensure_decode_metadata_buffers" in BATCHING
    assert "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH" in BATCHING
    assert "QwenContinuousDecodeGraph" in BATCHING
    assert "continuous_batching_cuda_graph_captures" in BATCHING
    assert "continuous_batching_cuda_graph_replays" in BATCHING
    assert "DecodeGraphTpProjectionScope tp_projection_scope" in BATCHING


def test_scheduler_supports_resident_and_cached_qwen_moe():
    assert "qwen_continuous_batch_has_moe" in BATCHING
    assert "continuous_batching_moe" in BATCHING
    assert "continuous_batching_moe_cached_row_serial" in BATCHING
    assert "continuous batching requires dense Qwen blocks" not in BATCHING
    assert "continuous batching cannot use the expert cache" not in BATCHING
    assert "qwen_continuous_batch_has_cached_moe" in BATCHING
    assert "uses_moe_expert_cache()" in DECODE
    assert "MoeContinuousBatchCacheScope moe_cache_scope" in BATCHING
    assert "g_moe_continuous_batch_cache_serial" in DECODE
    assert "each routed FFN row independently" in DECODE
    assert (
        "cpu_moe_down ||\n"
        "                        g_moe_continuous_batch_cache_serial"
        in DECODE
    )


def test_generic_qwen_loader_constructs_moe_ffns():
    assert '"experts.gate_up.weight"' in QWEN_LOADER
    assert '"experts.gate.weight"' in QWEN_LOADER
    assert '"experts.up.weight"' in QWEN_LOADER
    assert '"experts.down.weight"' in QWEN_LOADER
    assert '"shared_expert.router.weight"' in QWEN_LOADER
    assert "load_mfe_gpu(" in QWEN_LOADER
    assert "ffn.is_moe = true" in QWEN_LOADER
    assert "load_qwen_ffn(" in QWEN_LOADER
    assert "config.num_experts" in QWEN_LOADER
    assert "return this->config.num_experts" in CAUSAL_LM
    assert "dense Qwen model config intermediate_size must be positive" in QWEN_CONFIG
    assert '"Qwen model config intermediate_size must be positive"' not in QWEN_CONFIG


def test_scheduler_services_decode_before_contended_prefill_admission():
    decode_first = "if (decode_was_active) decode_active();"
    limited_join = "? size_t{1}"
    admission = "advance_prefills(incoming, contended);"
    assert decode_first in BATCHING
    assert limited_join in BATCHING
    assert BATCHING.index(decode_first) < BATCHING.index(admission)
    assert "continuous_batching_interleaved_admissions" in BATCHING


def test_qwen_decode_accepts_independent_batch_positions():
    assert "cache_positions.size(0) == B" in DECODE
    assert "write_positions.size(0) == B" in DECODE
    assert "cache_positions_override.has_value()" in DECODE
    assert "pos_batches" in ROPE
    assert "rows_per_batch" in ROPE
    assert ATTENTION.count("seq_len[b]") >= 5
    assert "seq_len.numel() == B" in ATTENTION
    assert "seq_len[batch]" in ATTENTION_MMA
    assert "seq_len.numel() == B" in ATTENTION_MMA
    assert "positions[(size_t)b * T + t]" in KV_CACHE
    assert "kv_cache_write_cuda(k, v, kh, vh, pos)" in DECODE


def test_real_weight_gate_exercises_join_and_compaction():
    assert "run_qwen_continuous_batching_check" in BATCHING
    assert "a blocked response callback stalled the scheduler" in BATCHING
    assert "cancellation_produced == 1" in BATCHING
    assert 'metric("continuous_batching_max_batch") >= 2.0' in BATCHING
    assert 'metric("continuous_batching_compactions") >= 1.0' in BATCHING
    assert "std::vector<int64_t> first_prompt(193)" in BATCHING
    assert '" prompt_lengths=193,17 split_k=1"' in BATCHING


def test_linear_attention_uses_the_canonical_nint_operator():
    assert "nint_matmul_ws_cuda" in DECODE
    assert "MFQ_LINEAR_ATTN_SMALL_M_QX_REUSE" not in DECODE
    assert "nint_small_m_qx_compatible" not in DECODE
    assert "shared_qx_weight" not in DECODE
    assert "forward_qx(" not in DECODE
