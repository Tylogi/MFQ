"""检查 MiniCPM C++ 原生执行与 Python 实时网关契约。"""
from pathlib import Path

ROOT = Path(__file__).parents[1]
CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CUDA_COMPONENTS = "\n".join(
    (CUDA_ROOT / "storage" / name).read_text(encoding="utf-8")
    for name in ("model_loader.h", "model_loader.cpp")
)
CUDA_ENGINE = (CUDA_ROOT / "engine" / "cuda_engine.cpp").read_text(
    encoding="utf-8"
)
CUDA_OPTIONS = (CUDA_ROOT / "include/mfq/cuda/engine.h").read_text(
    encoding="utf-8"
)
MINICPM_ENGINE = (
    CUDA_ROOT / "models" / "minicpmo45" / "components.cpp"
).read_text(encoding="utf-8")
ROPE = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "rope.cu").read_text(
    encoding="utf-8"
)
ATTENTION = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "attention.cu").read_text(
    encoding="utf-8"
)
NORM = (ROOT / "csrc" / "backends" / "cuda" / "kernels" / "norm.cu").read_text(
    encoding="utf-8"
)
GRAPH = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (CUDA_ROOT / "models" / "minicpmo45").glob("*")
    if path.suffix in {".h", ".cpp"}
)
COMMAND = (CUDA_ROOT / "commands/minicpmo45.cpp").read_text(encoding="utf-8")
SHARED = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "csrc/models/minicpmo45").glob("*.h"))
METAL_GRAPH = (ROOT / "csrc" / "backends" / "metal" / "models/minicpmo45" / "mlx_minicpmo45.cpp").read_text(
    encoding="utf-8"
)
METAL_HEADER = (ROOT / "csrc" / "backends" / "metal" / "models/minicpmo45" / "mlx_minicpmo45.h").read_text(
    encoding="utf-8"
)
METAL_DECODE = (ROOT / "csrc" / "backends" / "metal" / "apps" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)
METAL_COMPONENTS = (
    ROOT / "csrc" / "backends" / "metal" / "runtime" / "mlx_server_components.cpp"
).read_text(encoding="utf-8")
METAL_PLATFORM = (
    ROOT / "csrc" / "backends" / "metal" / "runtime" / "mlx_platform.h"
).read_text(encoding="utf-8")
SERVER_HEADER = (ROOT / "csrc" / "core" / "include" / "mfq" / "runtime.h").read_text(
    encoding="utf-8"
)
TRANSPORT_SRC = ROOT / "csrc" / "transport"
SERVER_SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
TEXT_PROCESSOR = (
    ROOT / "csrc" / "engine" / "src" / "text_processor.cpp"
).read_text(encoding="utf-8")
REALTIME_GATEWAY = (
    ROOT / "mfq" / "runtime" / "minicpmo45_realtime.py"
).read_text(encoding="utf-8")


def test_minicpmo45_uses_native_composite_graph_and_canonical_names():
    common = (ROOT / "csrc/models/common/causal_model.h").read_text()
    assert "models/minicpmo45/causal_lm.h" in DECODE
    assert 'constexpr auto embedding = "model.token_embedding.weight"' in common
    assert 'source, "model.output_norm.weight"' in DECODE
    assert 'constexpr auto output = "model.output.weight"' in common
    assert "models::minicpmo45::CausalLm<CudaCausalOps<MiniCPMO45Model>>" in DECODE
    assert "llm.model." not in DECODE
    assert "llm.lm_head.weight" not in DECODE


def test_minicpmo45_graph_binds_all_checkpoint_components():
    required_names = (
        "vision.patch_embedding.weight",
        "vision.block.",
        "vision.resampler.attention.qkv.weight",
        "audio.patch_embedding.conv1.weight",
        "audio.block.",
        "audio.projector.input",
        'result.config, index,\n                "full_attention", mfq::models::minicpmo45::LanguageComponent::tts, "tts"',
        "tts.text_embedding.weight",
        "tts.code_embedding.0.weight",
        "tts.semantic_projector.input",
        "tts.code_output.0.weight_norm.magnitude",
        "tts.code_output.0.weight_norm.direction",
    )
    for name in required_names:
        assert name in GRAPH


def test_minicpmo45_audio_and_tts_follow_official_attention_contracts():
    assert "query_positions / mfq::models::minicpmo45::audio_chunk_frames + 1" in GRAPH
    assert "audio_chunk_frames = 50" in SHARED
    assert "raw_lengths.to(" in GRAPH
    assert "mfq_scaled_dot_product_attention(" in GRAPH
    assert "mfq_tensor_backend::baddbmm(" in GRAPH
    assert "mfq_linear(" in GRAPH
    assert 'result.model_type = "minicpmtts"' in SHARED
    assert "block->official_bf16 = component == mfq::models::minicpmo45::LanguageComponent::text" in DECODE
    assert "CudaBackbone::minicpmo_tts" in DECODE
    assert "component == LanguageComponent::standalone_tts ? 1.0 : 0.0" in SHARED
    assert "cache_position += tokens" in SHARED
    assert "mfq::models::minicpmo45::tts_forward(" in GRAPH
    assert "generate_official(" in GRAPH
    assert "mfq_tensor_backend::multinomial(" in GRAPH
    assert "repetition_penalty = 1.05" in GRAPH
    assert "!history.empty() && config.repetition_penalty != 1.0" in SHARED
    assert "result.tokens - (result.hit_eos ? 1 : 0)" in GRAPH
    assert "logits_trace->push_back(raw_step_logits.clone())" in GRAPH


def test_minicpmo45_resampler_requires_exact_numpy_position_asset():
    assert "minicpmo45-resampler-pos-embed-v1.bf16" in GRAPH
    assert "MFQRSPB1" in GRAPH
    assert ".narrow(0, 0, height)" in GRAPH
    assert "sincos_position" not in GRAPH


def test_minicpmo45_supports_native_tensor_files_and_bfloat16_tts():
    assert "mfq_tensor_backend::pickle_load(bytes)" in COMMAND
    assert "mfq_tensor_backend::pickle_save(" in COMMAND
    assert "MFQTNSR1" in (ROOT / "csrc" / "backends" / "cuda" / "native" / "tensor.cpp").read_text(
        encoding="utf-8"
    )
    assert "rr.scalar_type() == mfq_tensor_backend::kBFloat16" in DECODE
    assert "ff2.scalar_type() == mfq_tensor_backend::kBFloat16" in DECODE
    assert "down.is_dense() || down.is_mxfp8()" in DECODE
    assert "official_bf16 ? mfq_tensor_backend::kBFloat16" in DECODE


def test_minicpmo45_qwen_runtime_follows_official_bfloat16_boundaries():
    assert "qwen_rms_norm_bf16" in DECODE
    assert '"MFQ_MINICPM_FUSED_BF16_RMSNORM"' in DECODE
    assert "config.minicpm_fused_bf16_rmsnorm" in DECODE
    assert "qwen_rms_norm_bf16_cuda(" in DECODE
    assert "qwen_rms_norm_pair_bf16_cuda(" in DECODE
    assert "qwen_rms_norm_bf16_kernel" in NORM
    assert "qwen_rms_norm_pair_bf16_finalize_kernel" in NORM
    assert "MFQ_DISABLE_NATIVE_PARALLEL_F32_MEAN" in (
        ROOT / "csrc" / "backends" / "cuda" / "native" / "tensor_reduction.cu"
    ).read_text(encoding="utf-8")
    assert 'rec.dtype != "NINT"' in DECODE
    assert "dequant_nint_dense_f32(to_gpu_nint(unpack_nint(blob)))" in DECODE
    assert "attention_cache_decode_split_gqa4_d128_part_kernel" in ATTENTION
    assert "mfq_dispatch_bfloat16" in ATTENTION
    assert "minicpm_bf16_rope_cache_write_cuda" in DECODE
    assert '"MFQ_MINICPM_FUSED_ROPE_KV"' in DECODE
    assert "config.minicpm_fused_rope_kv" in DECODE
    assert "minicpm_bf16_rope_cache_write_kernel" in ROPE
    assert "minicpm_qk_norm_rope_cache_write_bf16_kernel" in NORM
    assert '"MFQ_MINICPM_FUSED_QK_NORM_ROPE_KV"' in DECODE
    assert "config.minicpm_fused_qk_norm_rope_kv" in DECODE
    assert "active_rope.apply_bf16" in DECODE
    assert "official_bf16 ? mfq_tensor_backend::kBFloat16" in DECODE
    assert "k.scalar_type() != mfq_tensor_backend::kFloat16" in DECODE
    assert '"MFQ_MINICPM_BF16_GQA_DECODE"' in DECODE
    assert "execution.config.minicpm_bf16_gqa_decode" in DECODE
    assert "constboolbf16_gqa_decode=official_bf16&&T==1" in "".join(DECODE.split())
    assert "official_bf16 && !bf16_gqa_decode" in DECODE
    assert "MiniCPMO45Model::adapter_logits(" in GRAPH
    assert (
        "returnlm_head.forward(*execution,hidden).to(mfq_tensor_backend::kBFloat16).contiguous();"
        in "".join(GRAPH.split())
    )
    assert "repeated_k = kh.repeat_interleave(repeat, 1)" in DECODE
    assert '"full.minicpmo45_ffn_swiglu"' in DECODE
    assert "mfq_tensor_backend::silu(gate) * up" in DECODE
    assert "return logits_from_hidden(" in (ROOT / "csrc/models/common/causal_model.h").read_text(encoding="utf-8")
    assert "hidden.to(mfq_tensor_backend::kBFloat16)" in GRAPH
    assert "cache_pos > 0 && T > 1" in DECODE
    assert "minicpmo45_attention_mask" in DECODE
    assert "std::numeric_limits<mfq_bfloat16>::lowest()" in DECODE
    shared = (ROOT / "csrc/models/minicpmo45/causal_lm.h").read_text(encoding="utf-8")
    assert "bool adapter_uses_decode_sequence_length() const noexcept { return false; }" in shared
    assert "model().adapter_uses_decode_sequence_length() &&" in (ROOT / "csrc/models/common/causal_model.h").read_text(encoding="utf-8")
    assert "mask.eq(1).all().item<bool>()" in GRAPH
    assert "(tokens == 1 || cache_position == 0) && this->mask_all_ones(*mask)" in shared


def test_minicpmo45_preserves_qkv_projection_boundaries():
    assert "bool preserve_projection_boundaries = false" in DECODE
    assert "std::move(layers), preserve_projection_boundaries" in DECODE
    assert "loader.preserve_projection_boundaries = block->official_bf16" in DECODE
    loader = (CUDA_ROOT / "storage/transformer_loader.cpp").read_text()
    assert "names, 2, nullptr," in loader
    assert "down, shared_gate_up_compatible_prefix," in loader
    assert loader.count("preserve_projection_boundaries);") == 2
    assert "MFQ_DIAGNOSTIC_DISABLE_NINT_GROUP" in DECODE


def test_minicpmo45_matches_official_rope_frequency_construction():
    assert "bool official_reciprocal_frequencies = false" in DECODE
    assert "mfq_tensor_backend::reciprocal(" in DECODE
    assert "freq.copy_(official_freq)" in DECODE
    assert "metadata.rope_interleaved = true" in SHARED
    assert "model.metadata.rope_interleaved" in DECODE
    assert "rope_table_bf16_cuda" in DECODE
    assert "rope_table_bf16_kernel" in ROPE


def test_minicpmo45_serializes_independent_decode_projection_branches():
    assert "bool decode_branch_parallel = true" in DECODE
    assert "result.decode_branch_parallel = !preserve_projection_boundaries" in DECODE
    assert "decode_branch_parallel &&" in DECODE


def test_minicpmo45_cli_exposes_tensor_fixture_contract():
    assert 'option == "--minicpmo-input-prefix"' in DECODE
    assert 'option == "--minicpmo-output-prefix"' in DECODE
    assert 'option == "--minicpmo-tts-steps"' in DECODE
    assert 'input_prefix + ".input_ids.pt"' in COMMAND
    assert 'input_prefix + ".position_ids.pt"' in COMMAND
    assert 'input_prefix + ".attention_mask.pt"' in COMMAND
    assert 'output_prefix + ".image_embeddings.pt"' in COMMAND
    assert 'output_prefix + ".audio_embeddings.pt"' in COMMAND
    assert 'output_prefix + ".tts_codes.pt"' in COMMAND


def test_minicpmo45_native_duplex_preserves_streaming_caches():
    assert "forward_streaming(" in GRAPH
    assert "cache_length() + conv_tokens_before_crop" in GRAPH
    assert "prefix_extra_frames + 1" in GRAPH
    assert "suffix_extra_frames + 1" in GRAPH
    assert "MiniCPMO45DuplexSession" in GRAPH
    assert "runtime.language.reset(1)" in GRAPH
    assert "runtime.audio.reset()" in GRAPH
    assert "runtime.tts.reset(1)" in GRAPH
    assert "runtime.language.cache_pos" in GRAPH
    assert "runtime.audio.cache_length()" in GRAPH
    assert "runtime.tts.cache_position" in GRAPH
    assert "session.audio_chunk_index" in COMMAND


def test_minicpmo45_native_duplex_follows_official_unit_state_machine():
    assert "feed_id(ids.unit_start)" in GRAPH
    assert "feed_id(ids.unit_end)" in GRAPH
    assert "ids.is_chunk_terminator(token)" in GRAPH
    assert "!forced_decision && token == ids.listen" in GRAPH
    assert "!current_turn_ended" in GRAPH
    assert "token = ids.tts_bos" in GRAPH
    assert "index == max_new_speak_tokens - 1" in GRAPH
    assert "feed_id(ids.chunk_eos)" in GRAPH
    assert "if (index != 0)" in GRAPH
    assert "result.end_of_turn = token == ids.turn_eos" in GRAPH
    assert "generation_logits = pending.first" in GRAPH


def test_minicpmo45_native_duplex_uses_official_sampling_contracts():
    assert "first_id == ids.chunk_eos" in GRAPH
    assert "selected / repetition_penalty" in GRAPH
    assert "logits.index_fill_" in GRAPH
    assert "mfq_tensor_backend::topk(" in GRAPH
    assert "logits, top_k, -1, true, true" in GRAPH
    assert "cumulative > top_p" in GRAPH
    assert "generate_duplex_chunk(" in GRAPH
    assert "sampled.size() > 16" in GRAPH
    assert "condition.size(1) + result.tts_codes.size(1)" in GRAPH
    assert "int64_t top_k = 100" in GRAPH
    assert "double length_penalty = 1.0" in GRAPH
    assert "selected * length_penalty" in GRAPH


def test_minicpmo45_cli_exposes_native_duplex_tensor_contract():
    assert 'option == "--minicpmo-duplex-input-prefix"' in DECODE
    assert 'option == "--minicpmo-duplex-output-prefix"' in DECODE
    assert 'option == "--minicpmo-duplex-steps"' in DECODE
    assert 'option == "--minicpmo-duplex-max-speak-tokens"' in DECODE
    assert 'option == "--minicpmo-duplex-seed"' in DECODE
    assert 'option == "--minicpmo-duplex-greedy"' in DECODE
    assert 'input_prefix + ".special_ids.pt"' in COMMAND
    assert 'input + ".audio_features.pt"' in COMMAND
    assert 'input + ".force_listen.pt"' in COMMAND
    assert 'input + ".reset_session.pt"' in COMMAND
    assert 'output + ".generated_ids.pt"' in COMMAND
    assert 'output + ".tts_codes.pt"' in COMMAND
    assert 'output + ".state.pt"' in COMMAND


def test_minicpmo45_cuda_server_binds_the_realtime_backend():
    assert 'option == "--minicpmo-duplex"' not in DECODE
    assert "Components::start(" in MINICPM_ENGINE
    assert "Components::step(" in MINICPM_ENGINE
    assert "Components::stop(" in MINICPM_ENGINE
    assert "Components::prepare(" in MINICPM_ENGINE
    assert "mfq::engine::Engine" not in MINICPM_ENGINE
    assert "bind_runtime" not in CUDA_COMPONENTS
    assert "load_runtime_components(" in CUDA_COMPONENTS
    assert "state_->runtime.encode(" in MINICPM_ENGINE
    assert "MiniCPMO45Runtime::load_with_language(" in MINICPM_ENGINE
    assert "MiniCPMO45Runtime" not in CUDA_ENGINE
    assert "minicpmo" not in CUDA_OPTIONS.lower()
    assert "MiniCPMO45Runtime" not in CUDA_COMPONENTS
    assert "components.minicpmo" not in CUDA_COMPONENTS
    assert "state_->duplex_session->prepare_steps(" in MINICPM_ENGINE
    assert "parameters.reference_audio_features" in DECODE
    assert "input.force_speak" in DECODE
    assert "result.tts_force_flush" in DECODE


def test_minicpmo45_native_servers_share_mfqd_vision_tensors():
    assert "struct MfqMultimodalInput" in SERVER_HEADER
    assert "using MfqVisionInput = MfqMultimodalInput" in SERVER_HEADER
    assert "enum class MfqMultimodalProcessor" in SERVER_HEADER
    assert "MfqMultimodalGenerateFn" not in SERVER_HEADER
    assert "parse_mfq_vision(" in SERVER_SOURCE
    assert "class TensorFileReader final" in SERVER_SOURCE
    assert 'value.contains("binary_file")' in SERVER_SOURCE
    assert "file_reader->read(" in SERVER_SOURCE
    assert 'special_token(tokenizer, "<image>")' in TEXT_PROCESSOR
    assert "MiniCPM-o image placeholder must contain 64 query tokens" in TEXT_PROCESSOR
    assert "special_token(" not in SERVER_SOURCE
    assert "MfqCancellationCheck" not in MINICPM_ENGINE
    assert "InferenceCancelled" not in MINICPM_ENGINE
    assert "check_cancelled" not in GRAPH
    assert "mfq::StepSequence<CudaPreparedPrompt> Components::prepare(" in MINICPM_ENGINE
    assert "runtime.forward(" in DECODE
    assert "mfq::cuda::sample_logits(" in DECODE
    assert "generate_multimodal(" in METAL_HEADER
    assert "MlxMiniCPMO45Runtime::generate_multimodal(" in METAL_GRAPH
    assert "make_mlx_engine_components(" in METAL_DECODE
    assert "runtime_components.multimodal_generate" in METAL_DECODE
    assert "arguments.minicpmo_duplex" not in METAL_DECODE
    assert "std::optional<MlxMiniCPMO45Runtime>" in METAL_COMPONENTS


def test_minicpmo45_metal_dispatches_m3_family_tuning_by_device():
    assert '#include "mlx_platform.h"' in METAL_GRAPH
    assert "mlx_apple_chip_name()" in METAL_GRAPH
    assert 'sysctlbyname(\n                    "machdep.cpu.brand_string"' in METAL_PLATFORM
    assert '"MFQ_MINICPM_METAL_PROFILE"' in METAL_GRAPH
    assert 'std::strcmp(requested, "m3") == 0' in METAL_GRAPH
    assert 'std::strcmp(requested, "baseline") == 0' in METAL_GRAPH
    assert 'chip_name.rfind("Apple M3", 0) == 0' in METAL_GRAPH
    assert '"Apple M3 Pro"' in METAL_GRAPH
    assert '"Apple M3 Ultra"' in METAL_GRAPH
    assert '"Apple M4 Max"' in METAL_GRAPH
    assert '"Apple M5 Max"' in METAL_GRAPH
    assert "std::clamp((sequence + 255) / 256, 8, 16)" in METAL_GRAPH
    assert "std::min(6, (sequence + 1'023) / 1'024)" in METAL_GRAPH


def test_minicpmo45_metal_tracks_both_in_place_kv_cache_outputs():
    assert (
        "encoder.register_output_array(inputs[5]);\n"
        "        encoder.register_output_array(inputs[6]);"
    ) in METAL_GRAPH
    assert (
        "encoder.register_output_array(inputs[3]);\n"
        "        encoder.register_output_array(inputs[4]);"
    ) in METAL_GRAPH


def test_minicpmo45_cuda_duplex_uses_runtime_profile_tts_sampling():
    assert "double tts_temperature = 0.8" in GRAPH
    assert "double tts_repetition_penalty = 1.05" in GRAPH
    assert "tts_temperature, tts_repetition_penalty" in GRAPH
    assert 'number_field(\n                            generation, "tts_temperature"' in SERVER_SOURCE
    assert 'number_field(\n                            generation, "tts_repetition_penalty"' in SERVER_SOURCE


def test_minicpmo45_eval_batch_matches_pr_tts_sampler_and_optional_media():
    assert "std::mt19937*evaluator_rng=nullptr" in "".join(GRAPH.split())
    assert "std::uniform_real_distribution<float> distribution" in (ROOT / "csrc/engine/include/sampling.h").read_text(encoding="utf-8")
    assert "mfq::engine::sample_top_k_top_p(" in GRAPH
    assert 'request.value("tts_temperature", 0.8)' in COMMAND
    assert 'request.value("tts_top_p", 0.85)' in COMMAND
    assert 'request.value("tts_top_k", int64_t{25})' in COMMAND
    assert 'request.value("tts_min_tokens_to_keep", int64_t{3})' in COMMAND
    assert 'input_prefix + ".pixel_values.pt", false' in COMMAND
    assert '(reuse_prefix_cache&&prefix_length>=input_ids.size(1))' in ''.join(COMMAND.split())


def test_minicpmo45_eval_batch_maps_each_audio_bound_to_its_source():
    assert "valid_lengths[bound.source]" in COMMAND
    assert "bound.source, Slice(0, available), Slice()" in COMMAND
    assert "used_audio[static_cast<size_t>(bound.source)] = true" in COMMAND
    assert "MiniCPM-o eval batch has unused Whisper segments" in COMMAND


def test_minicpmo45_eval_batch_preserves_pr_teacher_forcing_segments():
    assert 'current_prefix + ".prefill_splits.pt",' in COMMAND
    assert "tts_teacher_forcing || require_segmented_prefill" in COMMAND
    assert "segment_begin == text_begin && segment_end == text_end" in COMMAND
    assert "teacher_text_hidden = segment_hidden" in COMMAND
    assert "teacher-forcing splits omit the text span" in COMMAND


def test_minicpmo45_eval_batch_can_match_pr_prefill_call_boundaries():
    assert 'request.value("require_segmented_prefill", false)' in COMMAND
    assert 'request.value("segmented_prefill_chunk_tokens", int64_t{0})' in COMMAND
    assert 'first_prefix + ".prefill_splits.pt"' in COMMAND
    assert "minicpmo45_prefill_segments(" in COMMAND
    assert "minicpmo45_teacher_prefill_segments(" in COMMAND
    assert "boundary - segment_begin >= chunk_tokens" in COMMAND
    assert "segments.emplace_back(text)" in COMMAND
    assert "teacher-forcing splits omit the text span" in COMMAND
    assert "MiniCPM-o segmented prefill omits a prompt span" in COMMAND
    assert "image_embedding_parts.push_back(runtime.resampler.forward(" in COMMAND
    assert "MiniCPM-o per-segment Whisper length mismatch" in COMMAND
    assert "Slice(0, raw_lengths[index])" in COMMAND


def test_minicpmo45_realtime_renderer_prefers_cuda_when_available():
    cuda_probe = "if torch.cuda.is_available():"
    mps_probe = "elif torch.backends.mps.is_available():"
    assert cuda_probe in REALTIME_GATEWAY
    assert mps_probe in REALTIME_GATEWAY
    assert REALTIME_GATEWAY.index(cuda_probe) < REALTIME_GATEWAY.index(mps_probe)


def test_minicpmo45_metal_uses_an_independent_qwen3_backbone():
    assert "class MiniQwen3Block" in METAL_GRAPH
    assert "class MiniQwen3Language" in METAL_GRAPH
    assert '"model.block."' in METAL_GRAPH
    assert "result.query_heads != 32" in METAL_GRAPH
    assert "result.kv_heads != 8" in METAL_GRAPH
    assert "mlx_qwen35" not in METAL_GRAPH
    assert "mlx_qwen35" not in METAL_HEADER
    assert 'backbone == "minicpmo45"' in METAL_DECODE


def test_minicpmo45_metal_binds_the_complete_composite_graph():
    for symbol in (
        "class VisionEncoder",
        "class Resampler",
        "class AudioEncoder",
        "class TtsDecoder",
        "vision.patch_embedding.weight",
        "vision.resampler.attention.qkv.weight",
        "audio.patch_embedding.conv1.weight",
        "audio.projector.input",
        "tts.text_embedding.weight",
        "tts.code_output.0.weight_norm.magnitude",
    ):
        assert symbol in METAL_GRAPH
    assert "minicpmo45-resampler-pos-embed-v1.bf16" in METAL_GRAPH
    assert "MFQRSPB1" in METAL_GRAPH


def test_minicpmo45_metal_fuses_streaming_audio_projections_safely():
    assert '"MFQ_METAL_AUDIO_FUSED_QKV"' in METAL_GRAPH
    assert "mlx::core::concatenate(" in METAL_GRAPH
    assert "auto pieces = mlx::core::split(" in METAL_GRAPH
    assert '"MFQ_METAL_AUDIO_STREAMING_NO_MASK"' in METAL_GRAPH
    assert "? std::nullopt" in METAL_GRAPH
    assert '"MFQ_METAL_AUDIO_BITWISE_HASH"' in METAL_GRAPH


def test_minicpmo45_metal_batches_only_official_vision_geometry():
    assert "minicpmo_vision_batchable_length" in METAL_GRAPH
    assert "return tokens >= 900 && tokens <= 1100" in METAL_GRAPH
    assert METAL_GRAPH.count("!minicpmo_vision_batchable_length(") == 2


def test_minicpmo45_metal_duplex_tracks_all_cache_lifetimes():
    assert "prepare_duplex(" in METAL_GRAPH
    assert "duplex_step(" in METAL_GRAPH
    assert "tts_text_start_position" in METAL_GRAPH
    assert "TTS cache did not advance exactly" in METAL_GRAPH
    assert "audio_chunk_index" in METAL_GRAPH
    assert "implementation_->duplex.reset()" in METAL_GRAPH
    assert "language_cache_position" in METAL_HEADER
    assert "audio_cache_position" in METAL_HEADER
    assert "tts_cache_position" in METAL_HEADER
    assert "kMinicpmoDuplexCacheLimitBytes" in METAL_DECODE
    assert "set_cache_limit(kMinicpmoDuplexCacheLimitBytes)" in METAL_DECODE
    assert "runtime_holder->reset();" in METAL_DECODE
    assert "release_model_load_staging_memory(runtime_stream);" in METAL_DECODE


def test_minicpmo45_realtime_uses_official_demo_defaults():
    assert "int32_t top_k = 100" in SERVER_HEADER
    assert "double length_penalty = 1.0" in SERVER_HEADER
    assert "duplex_defaults.top_k.value_or(100)" in SERVER_SOURCE
    assert "duplex_defaults.length_penalty.value_or(1.0)" in SERVER_SOURCE
    assert "std::random_device random" in SERVER_SOURCE
    assert "std::int32_t top_k = 100" in METAL_HEADER
    assert "double length_penalty = 1.0" in METAL_HEADER
    assert "config.length_penalty" in METAL_GRAPH
    assert '"force_listen_count": 0' in REALTIME_GATEWAY
    assert '"length_penalty": 1.0' in REALTIME_GATEWAY
    assert "await self.backend_runtime_defaults()" in REALTIME_GATEWAY
    assert "token2wav_steps: int = 10" in REALTIME_GATEWAY
    assert "DEFAULT_DUPLEX_SYSTEM_PROMPT" in REALTIME_GATEWAY


def test_minicpmo45_realtime_preserves_official_first_tts_flush():
    assert "tts_force_flush" in METAL_HEADER
    assert "first_tts_chunk" in METAL_GRAPH
    assert "end_of_turn || first_tts_chunk ? 0 : 26" in METAL_GRAPH
    assert '{"force_flush", result.tts_force_flush}' in SERVER_SOURCE
    assert "result.end_of_turn && !result.is_listen" in SERVER_SOURCE
    assert "force_flush=bool(event.get(\"force_flush\", False))" in REALTIME_GATEWAY
