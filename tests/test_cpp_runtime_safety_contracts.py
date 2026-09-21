"""Source contracts for runtime safety and boundary checks."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
MODEL_LOADER = (CUDA_ROOT / "runtime" / "causal_lm_loader.cpp").read_text(
    encoding="utf-8"
)
CUDA_RUNTIME = (CUDA_ROOT / "runtime" / "cuda_runtime.cpp").read_text(
    encoding="utf-8"
)
CUDA_NINT = (CUDA_ROOT / "ops" / "nint.cpp").read_text(encoding="utf-8")
TRANSPORT_SRC = ROOT / "cpp_runtime" / "transport"
TRANSPORT = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
SERVER = TRANSPORT
TRANSPORT_BASE = (
    TRANSPORT_SRC / "include" / "transport.h"
).read_text(encoding="utf-8")
HTTP_TRANSPORT = (TRANSPORT_SRC / "src" / "http.cpp").read_text(encoding="utf-8")
STDIO_TRANSPORT = (TRANSPORT_SRC / "src" / "stdio.cpp").read_text(encoding="utf-8")
METAL_RUNTIME = (
    ROOT / "cpp_runtime" / "backends" / "metal" / "apps" / "mfq_decode_mlx.cpp"
).read_text(encoding="utf-8")
METAL_VQ = (ROOT / "cpp_runtime" / "backends" / "metal" / "ops" / "mlx_vq.cpp").read_text(
    encoding="utf-8"
)
NVQ2J_CUDA = (ROOT / "mfq" / "quantize" / "cuda" / "nvq2j_assign.cu").read_text(
    encoding="utf-8"
)
NVQ3J_CUDA = (ROOT / "mfq" / "quantize" / "cuda" / "nvq3j_assign.cu").read_text(
    encoding="utf-8"
)
UNIFIED_CUDA_EXT = (ROOT / "mfq" / "kernels" / "cuda" / "_ext.py").read_text(
    encoding="utf-8"
)
CUDA_ATTENTION = (ROOT / "mfq" / "kernels" / "cuda" / "attention_mma.cu").read_text(
    encoding="utf-8"
)
CUDA_FATTN = (
    ROOT / "mfq" / "kernels" / "cuda" / "mfq_fattn_mma_f16.cuh"
).read_text(encoding="utf-8")
CUDA_MOE = (ROOT / "mfq" / "kernels" / "cuda" / "moe.cu").read_text(
    encoding="utf-8"
)
CUDA_MOE_PYTHON = (ROOT / "mfq" / "kernels" / "cuda" / "moe.py").read_text(
    encoding="utf-8"
)
CUDA_FATTN_SWIZZLE = (
    ROOT
    / "cpp_runtime"
    / "components"
    / "ggml"
    / "src"
    / "ggml-cuda"
    / "fattn-swizzle.cuh"
).read_text(encoding="utf-8")


def test_single_source_moe_cache_holds_full_demand_set() -> None:
    start = DECODE.index("bool MoeExpertCache::prepare(")
    stop = DECODE.index("bool MoeExpertCache::prepare_bundle(", start)
    prepare = DECODE[start:stop]
    assert "arena_demands" in prepare
    assert "item.first->book->capacity()" in prepare
    assert "book->mark_inflight(slot)" in prepare
    assert "&held_slots" in prepare


def test_moe_cache_capacity_failure_uses_full_projection_path() -> None:
    assert "if (!cache_->prepare(" in DECODE
    assert "count_full_projection_fallback" in DECODE
    assert "stage_cpu_mixed_moe(cpu_)" in DECODE


def test_optional_predictor_experts_join_the_shared_moe_cache() -> None:
    assert "bool defer_moe_cache_finalize = false" in DECODE
    assert "!defer_moe_cache_finalize" in MODEL_LOADER
    assert "const bool load_optional_components" in CUDA_RUNTIME
    assert CUDA_RUNTIME.index("load_runtime_components(") < CUDA_RUNTIME.index(
        "finalize_moe_expert_cache();"
    )


def test_reload_and_request_registration_share_one_gate() -> None:
    assert "std::mutex reload_gate;" in SERVER
    assert SERVER.count("std::lock_guard<std::mutex> gate(reload_gate);") >= 2
    assert "std::make_shared<ActiveRequest>(request_metrics_store)" in SERVER
    assert "active_request->complete(" in SERVER


def test_http_and_stdio_are_separate_transport_implementations() -> None:
    assert "class MfqTransport" in TRANSPORT_BASE
    assert "class MfqHttpTransport final : public MfqTransport" in HTTP_TRANSPORT
    assert "class MfqStdioTransport final : public MfqTransport" in STDIO_TRANSPORT
    assert "std::cin" not in HTTP_TRANSPORT
    assert "httplib::" not in STDIO_TRANSPORT


def test_stdio_transport_owns_stdin_and_isolates_stdout() -> None:
    assert "::dup2(STDERR_FILENO, STDOUT_FILENO)" in TRANSPORT
    assert "FD_CLOEXEC" in TRANSPORT
    assert "DuplicateHandle(" in TRANSPORT
    assert "FALSE, DUPLICATE_SAME_ACCESS" in TRANSPORT
    assert CUDA_RUNTIME.index("prepare_mfq_stdio_transport();") < CUDA_RUNTIME.index(
        "Model model ="
    )
    assert METAL_RUNTIME.index("prepare_mfq_stdio_transport();") < METAL_RUNTIME.index(
        "const mfq::metal::MfqContainer model(arguments.mfq);"
    )

    stdin_users: set[str] = set()
    for path in (ROOT / "cpp_runtime").rglob("*"):
        if path.suffix not in {".cpp", ".cc", ".cxx", ".cu", ".h", ".hpp"}:
            continue
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if any(
            marker in source
            for marker in ("std::cin", "STDIN_FILENO", "_fileno(stdin)")
        ):
            stdin_users.add(path.relative_to(ROOT / "cpp_runtime").as_posix())
    assert stdin_users == {
        "backends/cuda/models/minicpmo45/minicpmo45_runtime.cpp",
        "transport/src/stdio.cpp",
    }
    assert "if (!minicpmo_input_prefix.empty() || transport_mode ||" in CUDA_RUNTIME
    assert "MiniCPM-o eval mode cannot be combined with" in CUDA_RUNTIME


def test_metal_jsc_rejects_partial_code_vectors() -> None:
    assert "jsc && header.input_size % profile.vector_size != 0" in METAL_VQ


def test_cuda_jsc_direct_entry_validates_balanced_bank_mapping() -> None:
    assert "bank_counts[bank] == kStatesPerBank" in NVQ2J_CUDA
    assert "bank_counts[bank] == kStatesPerBank" in NVQ3J_CUDA


def test_cuda_kl_rejects_context_larger_than_model_capacity() -> None:
    assert DECODE.count("KL reference exceeds model context capacity") >= 2


def test_cuda_nint_loader_expands_v2_metadata_before_kernel_dispatch() -> None:
    start = CUDA_NINT.index("NintCpu unpack_nint(")
    stop = CUDA_NINT.index("Nint8ZeroCpu unpack_nint8_zero(", start)
    loader = CUDA_NINT[start:stop]
    assert "const bool is_nint_v2 = (raw_bits & 0x80) != 0;" in loader
    assert "t.sub_bits = blob[off++];" in loader
    assert "if (is_nint_v2)" in loader
    assert "const auto selectors = unpack_bits(" in loader
    assert "t.sub_scale.resize(sub_count);" in loader
    assert "t.sub_min.resize(sub_count);" in loader
    assert 'throw std::runtime_error("invalid NINT trailing bytes")' in loader


def test_unified_cuda_extension_can_include_runtime_headers() -> None:
    for include in (
        "_REPOSITORY_ROOT",
        "_CUDA_RUNTIME_INCLUDE",
        "_GGML_INCLUDE",
        "_GGML_SOURCE_INCLUDE",
        "_GGML_CUDA_INCLUDE",
    ):
        assert include in UNIFIED_CUDA_EXT
    assert "extra_include_paths=[" in UNIFIED_CUDA_EXT
    assert '"--extended-lambda"' in UNIFIED_CUDA_EXT
    assert '"-U__CUDA_NO_HALF_CONVERSIONS__"' in UNIFIED_CUDA_EXT


def test_cuda_swa_prefill_uses_implicit_kv_ranges() -> None:
    assert "mfq_causal_kv_range_kernel" in CUDA_ATTENTION
    assert "cache.kv_range" in CUDA_ATTENTION
    assert "mfq_swa_causal_mask_kernel" not in CUDA_ATTENTION


def test_cuda_flash_attention_swizzles_shared_kv_tiles() -> None:
    assert '#include "fattn-swizzle.cuh"' in CUDA_FATTN
    assert "ggml_cuda_fattn_smem_swizzle::bytes_rc" in CUDA_FATTN
    assert "namespace ggml_cuda_fattn_smem_swizzle" in CUDA_FATTN_SWIZZLE


def test_cuda_nint8_zero_prefill_can_consume_coarse_route_tiles() -> None:
    assert "template <int BM, bool COARSE_TILES = false>" in CUDA_MOE
    assert "COARSE_TILES ? BM : kRouteTile" in CUDA_MOE
    assert "coarse NINT8-0 route tile must match the MMA row tile" in CUDA_MOE
    assert "use_coarse_q8_tiles" in DECODE
    assert "use_coarse_q8_tiles ? route.mma_tile_m : 8" in DECODE


def test_python_nint8_zero_pool_passes_the_route_tile_size() -> None:
    tree = ast.parse(CUDA_MOE_PYTHON)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "nint8_zero_moe_grouped_matmul_pool_ws_cuda"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 21
    assert isinstance(calls[0].args[-1], ast.Constant)
    assert calls[0].args[-1].value == 8
