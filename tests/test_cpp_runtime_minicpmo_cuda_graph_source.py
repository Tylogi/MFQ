from pathlib import Path


ROOT = Path(__file__).parents[1]
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
CUDA_RUNTIME = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (
        CUDA_ROOT / "engine" / "generation.cpp",
        CUDA_ROOT / "engine" / "generation.h",
        CUDA_ROOT / "diagnostics" / "token_generation.h",
    )
)
BACKEND_CHECKS = (
    CUDA_ROOT / "diagnostics" / "backend_checks.cpp"
).read_text(encoding="utf-8")
MODEL_METADATA_SOURCE = "\n".join(
    (ROOT / "cpp_runtime/models" / model / "causal_lm.h").read_text(
        encoding="utf-8"
    )
    for model in ("glm_dsa", "minicpmo45")
)
SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (
        CUDA_ROOT / "models" / "causal_ops.h",
        CUDA_ROOT / "models" / "causal_ops.cpp",
        CUDA_ROOT / "models" / "session_codec_impl.h",
        CUDA_ROOT / "models" / "transformer.h",
        CUDA_ROOT / "models" / "transformer.cpp",
        CUDA_ROOT / "models" / "full_block.h",
        CUDA_ROOT / "models" / "full_block.cpp",
        CUDA_ROOT / "models" / "kv_cache.h",
        CUDA_ROOT / "models" / "kv_cache.cpp",
        CUDA_ROOT / "include" / "mfq_cuda_ops.h",
        CUDA_ROOT / "ops" / "include" / "cuda_execution.h",
        CUDA_ROOT / "ops" / "cuda_execution.cpp",
        CUDA_ROOT / "engine" / "decode_graph.h",
    )
) + "\n" + BACKEND_CHECKS + "\n" + CUDA_RUNTIME + "\n" + (
    CUDA_ROOT / "commands" / "diagnostics.cpp"
).read_text(encoding="utf-8")
ATTENTION_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "attention.cu"
).read_text(encoding="utf-8")
ATTENTION_MMA_SOURCE = (
    Path(__file__).parents[1]
    / "mfq"
    / "kernels"
    / "cuda"
    / "attention_mma.cu"
).read_text(encoding="utf-8")
BACKEND_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "backends" / "cuda" / "include" / "mfq_tensor_backend.h"
).read_text(encoding="utf-8")
CONTEXT_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "backends" / "cuda" / "src" / "mfq_cuda_context.cu"
).read_text(encoding="utf-8")
NATIVE_OPS_SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((CUDA_ROOT / "src").glob("mfq_native_tensor*"))
    if path.suffix in {".cpp", ".cu", ".cuh"}
)
NATIVE_TENSOR_SOURCE = (
    Path(__file__).parents[1] / "cpp_runtime" / "backends" / "cuda" / "src" / "mfq_native_tensor.cu"
).read_text(encoding="utf-8")
KV_CACHE_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "kv_cache.cu"
).read_text(encoding="utf-8")
ACC_SOURCE = (
    Path(__file__).parents[1] / "mfq" / "kernels" / "cuda" / "acc.cu"
).read_text(encoding="utf-8")


SHARED_ENGINE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "cpp_runtime" / "engine" / "include").glob("*.h")
)
SHARED_MODELS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "cpp_runtime" / "models").rglob("*.h")
)

SOURCE += SHARED_ENGINE + SHARED_MODELS

def test_minicpmo_native_runtime_keeps_cuda_graph_enabled() -> None:
    assert "graph_architecture_supported" not in SOURCE
    graph_gate = CUDA_RUNTIME.split(
        "bool graph_eligible() const {", 1
    )[1].split(
        "void prepare_graph()", 1
    )[0]
    assert "is_minicpmo45" not in graph_gate


def test_static_decode_uses_dynamic_position_for_kv_writes() -> None:
    causal_lm = (ROOT / "cpp_runtime/models/common/causal_model.h").read_text(
        encoding="utf-8"
    )
    static_forward = causal_lm.split(
        "Tensor hidden_forward_static", 1
    )[1].split("Tensor last_logits_static", 1)[0]
    assert "nullptr,positions,nullptr,0" in "".join(static_forward.split())
    assert "ops.device_ids(*ops.cache_positions_override)" in SOURCE
    assert (
        '"cache_positions must have shape [tokens] or [batch,tokens]"'
        in SOURCE
    )


def test_minicpmo_persistent_decode_workspaces_are_warmed_before_capture() -> None:
    assert MODEL_METADATA_SOURCE.count(
        "metadata.decode_graph_double_warmup = true;"
    ) == 2
    assert "model.metadata.decode_graph_double_warmup" in SOURCE
    assert "graph.ensure_captured(" in CUDA_RUNTIME
    assert CUDA_RUNTIME.count("prepare_decode_graph_memory(model,") == 1


def test_graph_stage_events_start_after_decode_workspace_warmup() -> None:
    graph_path = CUDA_RUNTIME.rsplit(
        "MfqCudaGraph graph;", 1
    )[1].split(
        "graph.capture_end();", 1
    )[0]
    warmup = graph_path.index("model.next_token_static(")
    profiler_reset = graph_path.index("profiler.reset();")
    external_events = graph_path.index(
        "profiler.graph_events = profile_cuda_graph;"
    )
    capture = graph_path.index("graph.capture_begin();")
    assert warmup < profiler_reset < external_events < capture


def test_graph_profile_covers_model_and_commit_boundaries() -> None:
    graph_path = CUDA_RUNTIME.rsplit(
        "MfqCudaGraph graph;", 1
    )[1].split(
        "graph.capture_end();", 1
    )[0]
    assert 'profiler.measure("decode.model_total"' in graph_path
    assert 'profiler.measure("decode.commit"' in graph_path
    assert graph_path.index('profiler.measure("decode.model_total"') < graph_path.index(
        'profiler.measure("decode.commit"'
    )


def test_torch_reference_graph_can_emit_a_debug_dump() -> None:
    assert 'std::getenv("MFQ_TORCH_CUDA_GRAPH_DUMP")' in BACKEND_SOURCE
    assert "graph.enable_debug_mode();" in BACKEND_SOURCE
    assert "graph.debug_dump(debug_path);" in BACKEND_SOURCE
    assert "mfq_debug_dump_cuda_graph(graph);" in SOURCE


def test_backend_bf16_add_check_covers_eager_and_graph_paths() -> None:
    assert "run_backend_bf16_add_check" in SOURCE
    assert 'option == "--check-backend-bf16-add"' in SOURCE
    check = BACKEND_CHECKS.split("int run_backend_bf16_add_check", 1)[1].split(
        "int run_backend_argmax_check", 1
    )[0]
    assert "eager = (left + right).contiguous();" in check
    assert "graph.capture_begin();" in check
    assert "graph.replay();" in check
    assert '<< " max_abs=" << maximum' in check
    assert "constexpr int chain_nodes = 512;" in check
    assert "chain_graph.capture_begin();" in check
    assert '<< " chain_node_us="' in check
    assert "shared_result = silu_mul_cuda(left, right);" in check
    assert "shared_graph.capture_begin();" in check
    assert '<< " shared_chain_node_us="' in check


def test_native_bf16_argmax_uses_bounded_contiguous_last_dimension_path() -> None:
    assert "argmax_last_contiguous_bf16_kernel" in NATIVE_OPS_SOURCE
    selection = NATIVE_OPS_SOURCE.split("if (operation == 3 &&", 1)[1].split(
        "auto launch =", 1
    )[0]
    assert "source.scalar_type() == kBFloat16" in selection
    assert "source.is_contiguous()" in selection
    assert "outer <= std::numeric_limits<unsigned int>::max()" in selection
    assert "selected + 1 == static_cast<std::size_t>(source.dim())" in selection


def test_native_prefill_batched_matmul_preserves_per_matrix_gemm() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL")', 1
    )[1].split("\n    for (std::int64_t batch = 0; batch < batches; ++batch)", 1)[0]
    assert "rows >= 32" in NATIVE_OPS_SOURCE
    assert "cudaStreamCaptureStatusNone" in NATIVE_OPS_SOURCE
    assert "parallel.ready.record(stream);" in selection
    assert "worker.wait(parallel.ready);" in selection
    assert "cublasGemmEx(" in selection
    assert "cublasGemmStridedBatchedEx(" not in selection
    assert "cudaStreamWaitEvent(" in selection


def test_native_bf16_causal_scale_fusion_is_exactly_bounded() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE")', 1
    )[1].split("if (fused_causal_scale)", 1)[0]
    assert "causal && !mask.has_value()" in selection
    assert "scores.scalar_type() == kBFloat16" in selection
    assert "scores.is_contiguous()" in selection
    assert "scores.dim() >= 2" in selection
    assert "scale_causal_bf16_kernel" in NATIVE_OPS_SOURCE
    assert "load_number(source, linear) * factor" in NATIVE_OPS_SOURCE


def test_native_bf16_softmax_preserves_serial_reduction_order() -> None:
    selection = NATIVE_OPS_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_EXACT_BF16_SOFTMAX")', 1
    )[1].split("auto working =", 1)[0]
    assert "input.scalar_type() == kBFloat16" in selection
    assert "input.is_contiguous()" in selection
    assert "selected + 1 == static_cast<std::size_t>(input.dim())" in selection
    assert "exact_bf16_softmax_max_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_numerator_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_sum_kernel" in NATIVE_OPS_SOURCE
    assert "exact_bf16_softmax_normalize_element_kernel" in NATIVE_OPS_SOURCE


def test_minicpmo_bf16_prefill_flash128_is_strictly_bounded() -> None:
    selection = SOURCE.split(
        "const bool bf16_flash128 =", 1
    )[1].split("if (bf16_flash128)", 1)[0]
    assert "!sliding && hd == 128" in selection
    assert "nh == 4 * nkh" in selection
    assert "!seq_len.has_value()" in selection
    assert "!attention_mask.has_value()" in selection
    flash_path = SOURCE.split("if (bf16_flash128)", 1)[1].split("} else {", 1)[0]
    assert "mfq_attention_mma128_cuda" in flash_path
    assert ".to(mfq_tensor_backend::kBFloat16)" in flash_path
    assert "mfq_attention_mma128_cuda" in ATTENTION_MMA_SOURCE
    implementation = ATTENTION_MMA_SOURCE.split(
        "mfq_attention_mma128_cuda", 1
    )[1].split("mfq_attention_mma512_cuda", 1)[0]
    assert "D == 128" in implementation
    assert "(gqa_ratio == 4 || gqa_ratio == 8)" in implementation
    assert "mfq_attention_mma_launch<128, 128, 16, 4>" in implementation
    assert "mfq_attention_mma_launch<128, 128, 8, 8>" in implementation
    casts = SOURCE.split(
        "const bool specialized_casts =", 1
    )[1].split("attention_token_major = true", 1)[0]
    assert "minicpm_flash128_q_cast_cuda" in casts
    assert "minicpm_flash128_kv_cast_cuda" in casts
    assert "minicpm_flash128_output_cast_cuda" in casts
    assert "minicpm_flash128_q_cast_kernel" in ATTENTION_MMA_SOURCE
    assert "minicpm_flash128_kv_cast_kernel" in ATTENTION_MMA_SOURCE
    assert "minicpm_flash128_output_cast_kernel" in ATTENTION_MMA_SOURCE
    assert "__bfloat162float" in ATTENTION_MMA_SOURCE
    assert "__float2half_rn" in ATTENTION_MMA_SOURCE
    assert "__float2bfloat16_rn" in ATTENTION_MMA_SOURCE
    assert ATTENTION_MMA_SOURCE.count("numel() > 0") >= 3


def test_full_flash_attention_accepts_gqa8_with_matching_tiles() -> None:
    implementation = ATTENTION_MMA_SOURCE.split(
        "mfq_attention_mma256_cuda", 1
    )[1].split("mfq_attention_mma128_cuda", 1)[0]
    assert "(gqa_ratio == 4 || gqa_ratio == 8)" in implementation
    assert "mfq_attention_mma_launch<256, 256, 16, 4>" in implementation
    assert "mfq_attention_mma_launch<256, 256, 8, 8>" in implementation
    selection = SOURCE.split(
        "const bool mma_attention_enabled =", 1
    )[1].split("else if (sliding)", 1)[0]
    assert "nh == 4 * nkh || nh == 8 * nkh" in selection


def test_minicpmo_bf16_residual_uses_contiguous_specialized_add() -> None:
    selection = SOURCE.split(
        "if (execution.config.minicpm_bf16_residual_acc)", 1
    )[1].split("return (rr + ff2)", 1)[0]
    assert "rr.scalar_type() == mfq_tensor_backend::kBFloat16" in SOURCE
    assert "return acc_cuda(rr, ff2)" in selection
    assert "acc_bf16_kernel" in ACC_SOURCE
    assert "a.is_cuda() && a.is_contiguous()" in ACC_SOURCE
    assert "a.sizes() == b.sizes()" in ACC_SOURCE
    assert "__float2bfloat16_rn" in ACC_SOURCE


def test_cuda_profiler_filter_supports_low_perturbation_eager_attribution() -> None:
    assert '"MFQ_PROFILE_CUDA_FILTER"' in SOURCE
    assert "profiler.filter = config.profile_filter" in SOURCE
    assert "if (!enabled || !selected(name)) return fn();" in SOURCE
    eager_path = CUDA_RUNTIME.rsplit("} else {", 1)[1].split(
        "mfq_cuda_synchronize();", 1
    )[0]
    assert 'profiler.measure("decode.eager_model"' in eager_path
    assert 'profiler.measure("decode.eager_commit"' in eager_path


def test_bf16_head_to_token_candidate_is_exactly_stride_bounded() -> None:
    assert "materialize_bf16_head_to_token_d128_kernel" in NATIVE_TENSOR_SOURCE
    assert "MFQ_NATIVE_BF16_HEAD_TO_TOKEN_CONTIGUOUS" not in NATIVE_TENSOR_SOURCE
    selection = NATIVE_TENSOR_SOURCE.split(
        "const bool head_to_token_layout =", 1
    )[1].split("if (head_to_token_layout)", 1)[0]
    assert "source.dim() == 4" in selection
    assert "destination.dim() == 4" in selection
    assert "destination.size(1) == tokens" in selection
    assert "destination.is_contiguous()" in selection
    assert "batch > 0 && tokens > 0" in selection
    assert "depth == 128" in selection
    assert "source.stride(1) == depth" in selection
    assert "source.stride(2) == tokens * depth" in selection
    assert "source.stride(0) == heads * tokens * depth" in selection
    assert "alignof(uint4)" in selection


def test_native_contiguous_bf16_to_f16_avoids_rank_decoding() -> None:
    assert "convert_contiguous_bf16_to_f16_kernel" in NATIVE_TENSOR_SOURCE
    selection = NATIVE_TENSOR_SOURCE.split(
        'std::getenv("MFQ_DISABLE_NATIVE_CONTIGUOUS_BF16_TO_F16")', 1
    )[1].split("if constexpr (", 1)[0]
    assert "source.is_contiguous()" in selection
    assert "destination.is_contiguous()" in selection
    assert "__float2half_rn(__bfloat162float(input[linear]))" in NATIVE_TENSOR_SOURCE


def test_native_bf16_kv_cache_uses_the_fused_writer() -> None:
    append = SOURCE.split("struct KVCache", 1)[1].split("struct Dsv4RopeTable", 1)[0]
    native_gate = append.split("#ifdef MFQ_NATIVE_CUDA_RUNTIME", 1)[1].split(
        "#else", 1
    )[0]
    assert "k.scalar_type()" not in native_gate
    assert "mfq_tensor_backend::kBFloat16" in KV_CACHE_SOURCE
    assert KV_CACHE_SOURCE.count("MFQ_DISPATCH_FLOATING_TYPES_AND2") >= 3


def test_graph_attention_tracks_eager_split_count_from_device_length() -> None:
    assert "attention_cache_decode_dynamic_cuda" in SOURCE
    assert "decode_attention_parts > 1" in SOURCE
    assert "context.decode_attention_parts" in SOURCE
    assert "g_decode_graph_attention_parts" not in SOURCE
    active_parts = ATTENTION_SOURCE.split(
        "__device__ __forceinline__ int attention_decode_active_parts", 1
    )[1].split("template <int BD, typename scalar_t>", 1)[0]
    assert "cache_position < 192" in active_parts
    assert "return 1;" in active_parts
    assert "attention_cache_decode_select_kernel" not in ATTENTION_SOURCE


def test_minicpmo_bf16_residual_norm_is_a_fused_cuda_operation() -> None:
    assert "acc_rms_norm_bf16_cuda" in SOURCE
    official_branch = SOURCE.split(
        "if (rr.scalar_type() == mfq_tensor_backend::kBFloat16", 1
    )[1].split("return acc_rms_norm_cuda", 1)[0]
    assert "return acc_rms_norm_bf16_cuda" in official_branch
