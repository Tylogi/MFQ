from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
BATCHING = "\n".join(
    (CUDA_ROOT / "engine" / name).read_text(encoding="utf-8")
    for name in ("cuda_batching.h", "cuda_batching.cpp")
)
BATCH_STATE = "\n".join(
    (CUDA_ROOT / "models" / "qwen35" / name).read_text(encoding="utf-8")
    for name in ("batch_state.h", "batch_state.cpp")
)
BATCHING_CHECK = (
    CUDA_ROOT / "diagnostics" / "runtime_checks.cpp"
).read_text(encoding="utf-8")
SCHEDULER = (ROOT / "csrc" / "scheduler" / "src" / "scheduler.cpp").read_text(encoding="utf-8")
ROPE = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "rope.cu").read_text(
    encoding="utf-8"
)
ATTENTION = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "attention.cu").read_text(
    encoding="utf-8"
)
ATTENTION_MMA = (
    ROOT / "csrc" / "backends" / "cuda" / "kernels" / "attention_mma.cu"
).read_text(encoding="utf-8")
KV_CACHE = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "kv_cache.cu").read_text(
    encoding="utf-8"
)
QWEN_LOADER = (
    ROOT / "csrc" / "models" / "qwen35" / "causal_lm.h"
).read_text(encoding="utf-8")
QWEN_CONFIG = (
    ROOT / "csrc" / "models" / "qwen35" / "config.cpp"
).read_text(encoding="utf-8")
CAUSAL_LM = "\n".join(
    (CUDA_ROOT / name).read_text(encoding="utf-8")
    for name in ("models/common/causal_model_ops.h", "models/common/causal_model_ops.cpp", "storage/session_codec.h")
)
RUNTIME_OPTIONS = (
    CUDA_ROOT / "storage" / "load_options.cpp"
).read_text(encoding="utf-8")


SHARED_ENGINE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "engine" / "include").glob("*.h")
)
SHARED_MODELS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "csrc" / "models").rglob("*.h")
)

DECODE += SHARED_ENGINE + SHARED_MODELS
BATCHING += SHARED_ENGINE
CAUSAL_LM += SHARED_MODELS

def test_continuous_batching_is_an_explicit_server_mode():
    assert '"--continuous-batching"' in DECODE
    assert '"--check-continuous-batching"' in DECODE
    assert "ContinuousBatch<QwenBatchOperations>" in BATCHING_CHECK
    assert "QwenBatchExecutor" not in BATCHING
    assert "ContinuousBatchingController" not in BATCHING
    assert "ContinuousBatchQueue" not in BATCHING
    assert "std::thread" not in (CUDA_ROOT / "engine/cuda_batching.cpp").read_text()
    assert "engine_.step(eligible)" in SCHEDULER
    assert "worker_" not in BATCHING
    assert "queue_" not in BATCHING
    assert "queue_mutex_" not in BATCHING
    assert "batch_compatible(constEngineRequest&request," in "".join(DECODE.split())
    assert "run_exclusive_generation" not in BATCHING


def test_cuda_server_prefill_is_bounded_for_serial_mtp_and_batched_paths():
    assert '"--prefill-chunk-size"' in DECODE
    assert "co_yield PrefillProgress" in DECODE
    assert "InferenceOutput& output" in DECODE
    assert "prefill_chunk_size" in SHARED_ENGINE
    assert "std::min(chunk_size, token_budget)" in SHARED_ENGINE
    assert "std::optional<QwenBatchState> prefill_" in BATCH_STATE
    assert "operations.prefill(request, chunk)" in BATCHING
    assert "schedule_prefill(PrefillChunk chunk)" in BATCHING
    assert "request->prefill_offset = chunk.offset + chunk.count" in SHARED_ENGINE
    assert "request->prefill_offset =" not in (CUDA_ROOT / "engine/cuda_batching.cpp").read_text()
    assert "next_prefill_chunk(" in BATCHING
    assert "work.prefill" in BATCHING
    assert "continuous_batching_prefill_chunks" in BATCHING
    assert "continuous_batching_prefill_yields" in BATCHING


def test_scheduler_supports_dynamic_join_retire_and_per_request_sampling():
    for internal in ("take", "restore", "make_slot_state", "copy_to_slot", "bind_paged_slots"):
        assert f"state_adapter_.{internal}(" not in BATCHING
    assert "qwen35::QwenBatchRequestState cache" in BATCHING
    assert "state_adapter_.prepare_prefill(" in BATCHING
    assert "state_adapter_.activate(" in BATCHING
    assert "state_adapter_.resume_decode(" in BATCHING
    assert "model_.cache_pos =" not in BATCHING
    assert "slots_" not in (CUDA_ROOT / "engine/cuda_batching.cpp").read_text()
    for operation in (
        "Qwen35BatchStateAdapter::take(",
        "Qwen35BatchStateAdapter::restore(",
        "Qwen35BatchStateAdapter::copy_to_slot(",
        "Qwen35BatchStateAdapter::release(",
        "Qwen35BatchStateAdapter::bind_paged_slots(",
    ):
        assert operation in BATCH_STATE
    assert "QwenPagedKvArena" not in BATCHING
    assert "mfq::cuda::sample_logits" in BATCHING
    assert "request->sampler" in BATCHING
    assert "request->token_constraint" in BATCHING
    assert "state.active" in BATCHING
    assert "generate_sequence(ops, output" in SHARED_ENGINE
    assert "execution.append(token)" not in BATCHING
    assert "output.finish(" not in BATCHING
    assert "request->eligible" in BATCHING
    assert "state_adapter_.prepare_decode(slots, batch" in BATCHING
    assert "const int64_t batch = max_sequences_" not in BATCHING
    assert "retire_cancelled" not in BATCHING
    assert "ops.generate(it->first, current)" in SHARED_ENGINE
    assert "step_batch(" not in SHARED_ENGINE
    assert "admit_batch(" not in SHARED_ENGINE
    assert "MFQ_CONTINUOUS_BATCH_GREEDY" in RUNTIME_OPTIONS
    assert "std::getenv(" not in BATCHING
    assert "config_.greedy" in BATCHING
    assert "continuous_batching_batched_greedy_batches" in BATCHING
    assert "sample_greedy_cuda(" in BATCHING
    assert "MFQ_CONTINUOUS_BATCH_PACKED_METADATA" not in RUNTIME_OPTIONS
    assert "continuous_batching_packed_metadata_batches" in BATCHING
    assert "continuous_batching_stable_slot_releases" in BATCHING
    assert "ensure_decode_metadata_buffers" in BATCHING
    assert "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH" in RUNTIME_OPTIONS
    assert "QwenContinuousDecodeGraph" in BATCHING
    assert "continuous_batching_cuda_graph_captures" in BATCHING
    assert "continuous_batching_cuda_graph_replays" in BATCHING
    assert "DecodeGraphTpProjectionScope tp_projection_scope" in BATCHING


def test_scheduler_supports_resident_and_cached_qwen_moe():
    assert "state_adapter_.has_moe()" in BATCHING
    assert "continuous_batching_moe" in BATCHING
    assert "continuous_batching_moe_cached_row_serial" in BATCHING
    assert "continuous batching requires dense Qwen blocks" not in BATCHING
    assert "continuous batching cannot use the expert cache" not in BATCHING
    assert "state_adapter_.has_cached_moe()" in BATCHING
    assert "uses_moe_expert_cache()" in DECODE
    assert "MoeContinuousBatchCacheScope moe_cache_scope" in BATCHING
    assert "execution.continuous_batch_cache_serial" in DECODE
    assert "each routed FFN row independently" in DECODE
    assert (
        "cpu_moe_down||execution.continuous_batch_cache_serial"
        in "".join(DECODE.split())
    )


def test_generic_qwen_loader_constructs_moe_ffns():
    assert '"experts.gate_up.weight"' in QWEN_LOADER
    assert '"experts.gate.weight"' in QWEN_LOADER
    assert '"experts.up.weight"' in QWEN_LOADER
    assert '"experts.down.weight"' in QWEN_LOADER
    assert '"shared_expert.router.weight"' in QWEN_LOADER
    assert "models::load_moe_weights<Ffn>(" in QWEN_LOADER
    assert "load_mfe_gpu(" in DECODE
    assert "result.is_moe = true" in DECODE
    assert "load_ffn<typename Loader::Ffn>(" in QWEN_LOADER
    assert "metadata.num_experts = config.num_experts" in (ROOT / "csrc/models/qwen35/causal_lm.h").read_text(encoding="utf-8")
    assert "return this->metadata.num_experts" in CAUSAL_LM
    assert "dense Qwen model config intermediate_size must be positive" in QWEN_CONFIG
    assert '"Qwen model config intermediate_size must be positive"' not in QWEN_CONFIG


def test_step_alternates_bounded_prefill_and_decode():
    assert "has_decode && (prefill == requests.end() || decode_next)" in BATCHING
    assert "decode_next = !decode_next" in BATCHING
    assert "operations.prefill(request, chunk)" in BATCHING
    assert "release(*work);" in BATCHING


def test_qwen_decode_accepts_independent_batch_positions():
    assert "positions.size(0) == B" in DECODE
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
    assert "run_qwen_continuous_batching_check" in BATCHING_CHECK
    assert "paused row advanced" in BATCHING_CHECK
    assert 'tokens["first"] == first_reference' in BATCHING_CHECK
    assert 'tokens["second"] == second_reference' in BATCHING_CHECK
    assert 'cancellations["cancel"] == 1 && terminals["cancel"] == 1' in BATCHING_CHECK
    assert 'tokens["cancel"].size() == before_cancel' in BATCHING_CHECK
    assert '"paged_kv_live_pages"' in BATCHING_CHECK
    assert "std::vector<int64_t> first_prompt(193)" in BATCHING_CHECK
    assert "session restore differs from serial oracle" in BATCHING_CHECK


def test_linear_attention_uses_the_canonical_nint_operator():
    assert "nint_matmul_ws_cuda" in DECODE
    assert "MFQ_LINEAR_ATTN_SMALL_M_QX_REUSE" not in DECODE
    assert "nint_small_m_qx_compatible" not in DECODE
    assert "shared_qx_weight" not in DECODE
    assert "forward_qx(" not in DECODE
