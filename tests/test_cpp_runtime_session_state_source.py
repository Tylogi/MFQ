from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
METAL_DECODE = (ROOT / "cpp_runtime" / "backends" / "metal" / "apps" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)
METAL_QWEN = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "models/qwen35" / "mlx_qwen35_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_DSV4 = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "models/deepseek_v4" / "mlx_deepseek_v4_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_DSV41 = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "models/deepseek_v41"
    / "mlx_deepseek_v41_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_MINICPM = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "models/minicpmo45" / "mlx_minicpmo45.cpp"
).read_text(encoding="utf-8")
TRANSPORT_SRC = ROOT / "cpp_runtime" / "transport"
SERVER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
HEADER = (ROOT / "cpp_runtime" / "core" / "include" / "mfq" / "runtime.h").read_text(encoding="utf-8")
PAGED_HEADER = (ROOT / "cpp_runtime" / "core" / "mfq_paged_prefix_cache.h").read_text(
    encoding="utf-8"
)
PAGED_SOURCE = (ROOT / "cpp_runtime" / "core" / "mfq_paged_prefix_cache.cpp").read_text(
    encoding="utf-8"
)
METAL_PAGED_CODEC = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "runtime" / "mlx_paged_session_codec.cpp"
).read_text(encoding="utf-8")
METAL_PAGED_CODEC_HEADER = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "runtime" / "mlx_paged_session_codec.h"
).read_text(encoding="utf-8")
METAL_COMPONENTS = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "runtime" / "mlx_server_components.cpp"
).read_text(encoding="utf-8")
METAL_STREAM_SYNC = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "runtime" / "mlx_stream_sync.h"
).read_text(encoding="utf-8")


def test_native_session_identifier_reaches_the_cuda_runtime() -> None:
    assert "std::string session_id;" in HEADER
    assert 'body.contains("mfq_session_id")' in SERVER
    assert "valid_mfq_session_id" in SERVER
    assert "const MfqPromptCachePlan & cache_plan" in DECODE
    assert "cache_plan.session_id" in DECODE


def test_stateless_text_requests_use_the_content_addressed_prefix_cache() -> None:
    assert "work.cache_plan.stable_prefix_tokens = work.prompt.size();" in SERVER
    assert "persistent_prefix_enabled()" in DECODE
    assert "persistent_prefix_enabled()" in METAL_DECODE
    assert "!cache_plan.session_id.empty() ||" in DECODE
    assert "!cache_plan.session_id.empty() ||" in METAL_DECODE
    assert "if (!requested_session.empty())" in DECODE
    assert "if (!requested_session.empty())" in METAL_DECODE


def test_full_attention_session_state_copies_only_visible_linear_kv() -> None:
    assert "struct TextSessionState" in DECODE
    assert "supports_text_session_state" in DECODE
    assert "state.ring" in DECODE
    assert "std::min<int64_t>(cache_pos, state.capacity)" in DECODE
    assert "restore_text_session_state" in DECODE


def test_deepseek_v4_session_state_preserves_local_and_compressed_caches() -> None:
    assert "TextSessionStateKind::DeepseekV4" in DECODE
    assert "struct Dsv4PoolSessionState" in DECODE
    assert "saved.local_cache = dsv4->local_cache.clone()" in DECODE
    assert "capture_dsv4_pool_session_state(" in DECODE
    assert "restore_dsv4_pool_session_state(" in DECODE
    assert "cache_pos / source.ratio" in DECODE


def test_glm_dsa_session_state_preserves_mla_and_index_caches() -> None:
    assert "TextSessionStateKind::GlmDsa" in DECODE
    assert "struct GlmDsaBlockSessionState" in DECODE
    assert "saved.kv_cache = glm->kv_cache.narrow(" in DECODE
    assert "saved.index_cache = glm->index_cache.narrow(" in DECODE
    assert "glm->shared_state->reset()" in DECODE
    assert 'a == "--check-text-session-state"' in DECODE
    assert '"text_session_state_check dsv4=1 glm_dsa=1\\n"' in DECODE


def test_partial_stable_prefix_is_saved_before_generation_suffix() -> None:
    assert "stable_prefix_tokens < prompt.size()" in DECODE
    assert "store_session_snapshot(std::vector<int64_t>(" in DECODE
    assert "tokens.size() > maximum_prefix_tokens" in DECODE


def test_qwen_hybrid_and_mtp_session_state_are_restored_together() -> None:
    assert "TextSessionStateKind::HybridAttention" in DECODE
    assert "saved.convolution_state = linear->conv_state.clone()" in DECODE
    assert "saved.recurrent_state = linear->gdn_state.clone()" in DECODE
    assert "std::optional<MtpSessionState> mtp" in DECODE
    assert "mtp->restore_session_state(*selected.mtp)" in DECODE
    assert "prompt.size() - reused_tokens" in DECODE


def test_multimodal_cache_keys_include_media_and_reuse_vision_output() -> None:
    assert "state.input_key != input_key" in DECODE
    assert "media.pixel_values.data()" in DECODE
    assert "cached_vision_key_ == cache_key" in DECODE
    assert "stable_prefix_tokens = transformed_prompt ? 0" not in DECODE
    assert "work.cache_plan.stable_prefix_tokens = 0;" not in SERVER


def test_session_cache_uses_exact_prefixes_and_reports_suffix_prefill() -> None:
    assert "!std::equal(" in DECODE
    assert "tokens.begin(), tokens.end(), prompt.begin()))" in DECODE
    assert "tokens.size() >= prompt.size()" in DECODE
    assert "prompt.size() - reused_tokens" in DECODE
    assert "MFQ_RUNTIME_MAX_KV_SESSIONS" in DECODE
    assert "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION" in DECODE
    assert "MFQ_RUNTIME_KV_SESSION_BYTES" in DECODE
    assert "MFQ_SERVER_" not in DECODE


def test_session_cache_retains_history_and_exposes_lifecycle_controls() -> None:
    assert "std::vector<TextSessionState>> states_" in DECODE
    assert "fork_session(" in DECODE
    assert "close_session(" in DECODE
    assert 'server.Post("/runtime/sessions/fork"' in SERVER
    assert 'R"(/runtime/sessions/' in SERVER
    assert "MfqSessionControl" in HEADER


def test_metal_runtime_matches_native_session_lifecycle_and_limits() -> None:
    assert "class MlxServerTextSessionCache" in METAL_DECODE
    assert "MFQ_SERVER_MAX_KV_SESSIONS" in METAL_DECODE
    assert "MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION" in METAL_DECODE
    assert "MFQ_SERVER_KV_SESSION_BYTES" in METAL_DECODE
    assert "fork_session(" in METAL_DECODE
    assert "close_session(" in METAL_DECODE
    assert "MfqSessionControl session_control" in METAL_DECODE
    assert "backend=metal" in METAL_DECODE


def test_metal_server_bounds_and_explicitly_reclaims_allocator_cache() -> None:
    assert "server_cache_limit_bytes()" in METAL_DECODE
    assert "physical_memory_bytes() / 16" in METAL_DECODE
    assert "mlx::core::set_cache_limit(allocator_cache_limit)" in METAL_DECODE
    assert '"mlx_cache_limit_bytes"' in METAL_DECODE
    assert "loaded_runtime.reset_cache(1)" in METAL_DECODE
    assert "drain_metal_work(runtime_stream)" in METAL_DECODE
    assert "release_model_load_staging_memory(runtime_stream);" in METAL_DECODE


def test_metal_server_drains_async_work_before_releasing_cache_storage() -> None:
    assert METAL_STREAM_SYNC.index("synchronize(runtime_stream)") < METAL_STREAM_SYNC.index(
        "synchronize();"
    )
    clear_control = METAL_DECODE.split("session_control.clear =", 1)[1].split(
        "session_control.trim_hot =", 1
    )[0]
    assert clear_control.index("drain_metal_work(runtime_stream)") < clear_control.index(
        "session_cache->clear()"
    )
    duplex_start = METAL_COMPONENTS.split("result.duplex.start =", 1)[1].split(
        "result.duplex.step =", 1
    )[0]
    assert duplex_start.index("drain_metal_work(runtime_stream)") < duplex_start.index(
        "runtime.reset()"
    )
    duplex_stop = METAL_COMPONENTS.split("result.duplex.stop =", 1)[1].split(
        "return result;", 1
    )[0]
    assert duplex_stop.index("drain_metal_work(runtime_stream)") < duplex_stop.index(
        "runtime_holder->value().reset()"
    )


def test_supported_metal_text_graphs_capture_and_restore_prefix_state() -> None:
    for source in (METAL_QWEN, METAL_DSV4, METAL_DSV41, METAL_MINICPM):
        assert "capture_text_session_state" in source
        assert "restore_text_session_state" in source
        assert "prompt.size() - reused_tokens" in source


def test_qwen_hybrid_prefix_cache_uses_exact_recurrent_boundaries() -> None:
    qwen_codec = METAL_PAGED_CODEC_HEADER.split(
        "MlxPagedSessionCodec<MlxQwen35TextSessionState>", 1
    )[1]
    assert 'name = "qwen35-hybrid-kv-v2"' in qwen_codec
    assert "kRecurrentUnavailableLayer" in METAL_PAGED_CODEC
    assert "decodable_blocks" in METAL_PAGED_CODEC
    assert "normalize_stable_prefix_tokens" in METAL_DECODE
    assert "paged_cache_->replace(" in METAL_DECODE
    assert "existing_blocks == full_blocks" in METAL_DECODE


def test_persistent_prefix_cache_is_content_addressed_and_restart_safe() -> None:
    assert "class PagedPrefixCache" in PAGED_HEADER
    assert "BlockHash block_hash(" in PAGED_HEADER
    assert "const BlockHash& parent" in PAGED_HEADER
    assert "compatibility_key" in PAGED_HEADER
    assert "payload_hash" in PAGED_SOURCE
    assert "std::filesystem::rename(temporary, final_path" in PAGED_SOURCE
    assert "worker_ = std::thread" in PAGED_SOURCE
    assert "max_pending_bytes" in PAGED_HEADER
    assert "pending_max_bytes" in PAGED_HEADER
    assert "pending_write_bytes_" in PAGED_SOURCE
    assert "enforce_disk_budget_locked" in PAGED_SOURCE
    assert "void pin(" in PAGED_HEADER
    assert "void unpin(" in PAGED_HEADER
    assert "load_prefix(" in PAGED_HEADER
    assert "max_parallel_reads" in PAGED_HEADER
    assert "LoadSource::Disk" in PAGED_SOURCE
    assert "std::vector<std::thread> readers" in PAGED_SOURCE
    assert "read_header(input, header)" in PAGED_SOURCE
    assert "release_disk_read_pins_locked(requests)" in PAGED_SOURCE
    assert "iterator->second.pins != 0" in PAGED_SOURCE
    assert "return !clearing_" in PAGED_SOURCE
    assert "bool clearing_ = false" in PAGED_SOURCE


def test_tiered_prefix_cache_can_release_only_its_hot_payloads() -> None:
    assert "std::uint64_t trim_hot(" in PAGED_HEADER
    assert "pins_.count(iterator->first) != 0" in PAGED_SOURCE
    assert 'server.Post("/runtime/cache/trim"' in SERVER
    assert "session_control.trim_hot" in SERVER
    assert "session_control.trim_hot" in METAL_DECODE
    assert "text_session_cache.trim_hot(target_bytes)" in DECODE
    assert "release_host_allocator_cache()" in METAL_DECODE
    assert "mfq_release_host_allocator_cache()" in DECODE


def test_metal_paged_codec_preserves_raw_kv_tensor_storage() -> None:
    assert "encode_state(" in METAL_PAGED_CODEC
    assert "encode_block(" in METAL_PAGED_CODEC
    assert "decode_blocks(" in METAL_PAGED_CODEC
    assert "value.data<std::uint8_t>()" in METAL_PAGED_CODEC
    assert "MlxMiniCPMO45TextSessionState" in METAL_PAGED_CODEC
    assert "make_metal_paged_cache<Runtime>" in METAL_DECODE
    assert "replace_paged_cache(" in METAL_DECODE
    assert "prefix_cache_disk_blocks" in METAL_DECODE
    assert "MFQ_SERVER_PREFIX_CACHE_PENDING_BYTES" in METAL_DECODE
    assert "prefix_cache_pending_max_bytes" in METAL_DECODE
    assert "paged_cache_->load_prefix(match.blocks)" in METAL_DECODE


def test_cuda_paged_cache_only_accepts_linear_full_attention_kv() -> None:
    assert "supports_paged_text_session_state" in DECODE
    assert "return full != nullptr && !full->sliding" in DECODE
    assert "encode_cuda_paged_session(" in DECODE
    assert "encode_cuda_paged_block(" in DECODE
    assert "decode_cuda_paged_session(" in DECODE
    assert "if (layer.ring" in DECODE
    assert "make_cuda_paged_prefix_cache(" in DECODE
    assert "backend=cuda action=paged_hit" in DECODE
    assert "prefix_cache_disk_blocks" in DECODE
    assert "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES" in DECODE
    assert "prefix_cache_pending_max_bytes" in DECODE
    assert "paged_cache_->load_prefix(match.blocks)" in DECODE
