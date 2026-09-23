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
BENCH = (CUDA_ROOT / "benchmarks" / "qwen_continuous_batching_benchmark.h").read_text(
    encoding="utf-8"
)
BENCH_APP = (CUDA_ROOT / "apps" / "mfq_bench.cpp").read_text(encoding="utf-8")
DECODE_RUNTIME = (CUDA_ROOT / "runtime" / "cuda_decode_runtime.cpp").read_text(
    encoding="utf-8"
)
CUDA_CMAKE = (ROOT / "cpp_runtime" / "cmake" / "CudaRuntime.cmake").read_text(
    encoding="utf-8"
)
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


def test_continuous_batching_is_an_explicit_server_mode():
    assert '"--continuous-batching"' in DECODE
    assert '"--check-continuous-batching"' in DECODE
    assert "CudaContinuousBatcher" in DECODE
    assert "decode=target_only mtp=disabled" in DECODE


def test_cuda_server_prefill_is_bounded_for_serial_mtp_and_batched_paths():
    assert '"--prefill-chunk-size"' in DECODE
    assert "server_prefill_tail(" in DECODE
    assert "server_hidden_forward_chunked(" in DECODE
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


def test_contention_benchmark_uses_the_real_batcher_and_reports_latency():
    assert "add_executable(mfq-bench" in CUDA_CMAKE
    assert "target_link_libraries(mfq-bench PRIVATE mfq-cuda-runtime mfq-server)" in CUDA_CMAKE
    assert '"continuous-batching"' in BENCH_APP
    assert "run_qwen_continuous_workload(" in BENCH_APP
    assert "run_qwen_continuous_batching_benchmark(" in BENCH_APP
    assert "bench-continuous-batching" not in DECODE_RUNTIME
    assert "qwen_continuous_batching_benchmark.h" not in DECODE_RUNTIME
    assert "CudaContinuousBatcher" in DECODE_RUNTIME
    assert "generate_server_tokens(" in DECODE_RUNTIME
    assert "serial_generate(" in BENCH
    assert "start_batcher()" in BENCH
    assert "batcher.submit(" in BENCH
    assert "a_timestamps.push_back(now)" in BENCH
    assert "last > b_submit_started && first < *b_prefill_callback" in BENCH
    assert BENCH.index("metrics_before =") < BENCH.index("(void)batcher.submit(", BENCH.index("std::thread b_thread"))
    for window in ("baseline", "contended"):
        for percentile in ("p50", "p95", "p99", "max"):
            assert f"{window}_{percentile}_itl_ms=" in BENCH
    assert "b_ttft_ms=" in BENCH
    assert "continuous_batching_interleaved_admissions" in BENCH
    assert "continuous_batching_prefill_yields" in BENCH
    assert "output differs from serial greedy oracle" in BENCH


def test_linear_attention_uses_the_canonical_nint_operator():
    assert "nint_matmul_ws_cuda" in DECODE
    assert "MFQ_LINEAR_ATTN_SMALL_M_QX_REUSE" not in DECODE
    assert "nint_small_m_qx_compatible" not in DECODE
    assert "shared_qx_weight" not in DECODE
    assert "forward_qx(" not in DECODE
