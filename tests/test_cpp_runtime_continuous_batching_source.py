"""Dependency checks only; batch behavior is covered by native execution tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
SHARED_MODELS = "\n".join(
    path.read_text(encoding="utf-8") for path in (ROOT / "csrc/models").rglob("*.h")
)
DECODE += SHARED_MODELS
CAUSAL_LM = (CUDA_ROOT / "models/common/causal_model_ops.h").read_text() + SHARED_MODELS
QWEN_LOADER = (ROOT / "csrc/models/qwen35/causal_lm.h").read_text()
QWEN_CONFIG = (ROOT / "csrc/models/qwen35/config.cpp").read_text()
ROPE = (CUDA_ROOT / "kernels/rope.cu").read_text()
ATTENTION = (CUDA_ROOT / "kernels/attention.cu").read_text()
ATTENTION_MMA = (CUDA_ROOT / "kernels/attention_mma.cu").read_text()
KV_CACHE = (CUDA_ROOT / "kernels/kv_cache.cu").read_text()


def test_shared_batch_policy_does_not_depend_on_native_state_protocols():
    source = (ROOT / "csrc/engine/internal/continuous_batch.h").read_text()
    for concrete in ("Qwen", "Cuda", "state_adapter", "suspend_decode", "resume_decode", "bind_paged_slots", "cache_position"):
        assert concrete not in source
    assert "backends/" not in source


def test_generation_interface_has_no_concrete_batch_dependency():
    for name in ("generation.h", "generation.cpp"):
        source = (CUDA_ROOT / "engine" / name).read_text()
        for concrete in ("qwen_batching.h", "QwenBatch", "Qwen35BatchStateAdapter", "ContinuousBatch<"):
            assert concrete not in source


def test_reusable_cuda_batch_execution_has_no_model_family_dependency():
    for name in ("batch_execution.h", "batch_execution.cpp", "batch_execution.tpp"):
        source = (CUDA_ROOT / "engine" / name).read_text()
        for concrete in ("Qwen", "qwen35/", "models/qwen", "StateAdapter", "suspend_decode(",
                         "resume_decode(", "prepare_decode(", "finish_decode(", ".slot(", "cache_position"):
            assert concrete not in source
    assert "engine/batch_execution.cpp" in (CUDA_ROOT / "CMakeLists.txt").read_text()


def test_paged_batch_storage_has_no_model_or_execution_dependency():
    source = (CUDA_ROOT / "storage/batch_paged_kv.h").read_text()
    for concrete in ("Qwen", "FullBlock", "models/", '"engine/', "InferenceRequest", "InferenceOutput"):
        assert concrete not in source
    assert not (CUDA_ROOT / "models/qwen35/paged_kv.h").exists()


def test_generic_qwen_loader_constructs_moe_ffns():
    for role in ("experts.gate_up.weight", "experts.gate.weight", "experts.up.weight", "experts.down.weight", "shared_expert.router.weight"):
        assert f'"{role}"' in QWEN_LOADER
    assert "models::load_moe_weights<Ffn>(" in QWEN_LOADER
    assert "load_mfe_gpu(" in DECODE
    assert "result.is_moe = true" in DECODE
    assert "load_ffn<typename Loader::Ffn>(" in QWEN_LOADER
    assert "metadata.num_experts = config.num_experts" in QWEN_LOADER
    assert "return this->metadata.num_experts" in CAUSAL_LM
    assert "dense Qwen model config intermediate_size must be positive" in QWEN_CONFIG
    assert '"Qwen model config intermediate_size must be positive"' not in QWEN_CONFIG


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


def test_linear_attention_uses_the_canonical_nint_operator():
    assert "nint_matmul_ws_cuda" in DECODE
    assert "MFQ_LINEAR_ATTN_SMALL_M_QX_REUSE" not in DECODE
    assert "nint_small_m_qx_compatible" not in DECODE
    assert "shared_qx_weight" not in DECODE
    assert "forward_qx(" not in DECODE
