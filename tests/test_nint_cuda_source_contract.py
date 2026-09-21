from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NINT = (ROOT / "mfq/kernels/cuda/nint_matmul.cu").read_text()
BINDINGS = (ROOT / "mfq/kernels/cuda/mfq_cuda.cpp").read_text()
EXTENSION = (ROOT / "mfq/kernels/cuda/_ext.py").read_text()
CMAKE = (ROOT / "cpp_runtime/backends/cuda/CMakeLists.txt").read_text()
MOE = (ROOT / "mfq/kernels/cuda/moe.cu").read_text()
MOE_PYTHON = (ROOT / "mfq/kernels/cuda/moe.py").read_text()
CUDA_ROOT = ROOT / "cpp_runtime/backends/cuda"
RUNTIME = "\n".join(
    path.read_text()
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
METAL_NINT = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_nint.cpp"
).read_text()
METAL_NINT_PYTHON = (
    ROOT / "mfq/kernels/metal/nint.py"
).read_text()
METAL_MOE_PYTHON = (
    ROOT / "mfq/kernels/metal/moe.py"
).read_text()
METAL_GROUPED_PYTHON = (
    ROOT / "mfq/kernels/metal/grouped_linear.py"
).read_text()
MLX_LINEAR_PYTHON = (
    ROOT / "mfq/runtime/mlx_linear.py"
).read_text()
METAL_GROUPED = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_grouped_linear.cpp"
).read_text()
METAL_MOE = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_moe.cpp"
).read_text()
METAL_MFE_KERNELS = (
    ROOT / "cpp_runtime/backends/metal/kernels/mfq_mfe_prefill.metal"
).read_text()


def test_nint_has_one_metadata_driven_small_m_compute_kernel():
    kernels = re.findall(
        r"__global__\s+void(?:\s+__launch_bounds__\([^)]*\))?\s+"
        r"([A-Za-z0-9_]+)",
        NINT,
    )
    assert kernels.count("nint_matmul_kernel") == 1
    assert kernels.count("nint_quantize_activation_kernel") == 1
    assert kernels.count("nint_decode_rows_kernel") == 1
    assert kernels.count("nint_backward_input_kernel") == 1
    assert not any(
        token in name
        for name in kernels
        for token in ("nint2", "nint3", "nint4", "nint5", "nint6", "nint7")
    )


def test_mfe_nint_reuses_the_one_runtime_metadata_compute_kernel():
    kernels = re.findall(
        r"__global__\s+void(?:\s+__launch_bounds__\([^)]*\))?\s+"
        r"([A-Za-z0-9_]+)",
        MOE,
    )
    assert "mfe_nint_matmul_kernel" not in kernels
    assert "launch_nint_matmul_routed_cuda" in MOE
    assert "launch_nint_matmul_routed_cuda" in NINT
    assert "const int32_t * __restrict__ route_ids" in NINT
    assert "gs >= 4 && gs <= 64" in MOE
    assert "gs == 16 ||" not in MOE
    assert "nint_moe_mmvq_mixed_q_kernel" not in MOE
    assert "MFQ_MIXED_Q_MOE_TOKEN_SWITCH" not in MOE
    assert "nint_moe_grouped_matmul_hetero_qx_cuda" not in MOE_PYTHON
    assert "nint_moe_grouped_matmul_pool_ws_cuda" not in MOE_PYTHON
    for name in (
        "nint_moe_mmvq_kernel",
        "nint_moe_mmvq_ksplit_kernel",
        "nint_moe_grouped_tile_kernel",
        "nint_moe_hetero_mmvq_kernel",
        "nint_moe_hetero_grouped_tile_kernel",
        "nint_moe_group32_mmq_kernel",
        "nint_moe_hetero_mma_kernel",
        "nint_moe_grouped_matmul_hetero_qx_cuda",
        "nint_moe_grouped_matmul_hetero_f16_cuda",
        "nint_moe_grouped_matmul_pool_ws_cuda",
        "nint_moe_quantize_24_28_ws_cuda",
        "nint_moe_quantize_multi_ws_cuda",
    ):
        assert name not in MOE
        assert f'm.def("{name}"' not in BINDINGS


def test_mfe_nint_prefill_reuses_compact_expert_route_map():
    assert "const int total_tiles = tile_bounds[experts];" in NINT
    assert "const int pair = ids_dst[compact];" in NINT
    assert "nint_matmul_routed_pair(" in NINT
    assert "tokens > 8 && route_map_ready" in NINT
    assert "route.map_ready, input_quantized" in RUNTIME
    assert "route.ids_dst, route.expert_bounds" in RUNTIME


def test_mfe_nint_large_m_uses_one_heterogeneous_tiled_decoder():
    assert NINT.count("nint_matmul_tiled_prefill_kernel(") == 1
    assert "row_q_bits[weight_row]" in NINT
    assert "row_q_bit_offsets[weight_row]" in NINT
    assert "nint_prefill_load_weight_tile(" in NINT
    assert "unpack_nint_codes8_packed(" in NINT
    assert "constexpr int values_per_vector = 8;" in NINT
    assert "__floats2half2_rn(decoded[0], decoded[1])" in NINT
    assert "nint_matmul_tiled_prefill_kernel<TILE_M, true, FUSED_GLU>" in NINT
    assert "MFQ_LAUNCH_NINT_TILED_PREFILL(16" in NINT
    assert "MFQ_LAUNCH_NINT_TILED_PREFILL(32" in NINT
    assert "MFQ_LAUNCH_NINT_TILED_PREFILL(64" in NINT
    assert "MFQ_LAUNCH_NINT_TILED_PREFILL(128" in NINT
    assert "rows_per_expert <= 16" in RUNTIME
    assert "rows_per_expert <= 32" in RUNTIME
    assert "uniform_q" not in NINT
    assert "uniform_k" not in NINT


def test_mfe_nint_large_m_overlaps_copy_and_scales_persistent_work():
    assert "copy_16_async(destination, source)" in NINT
    assert "copy_async_commit()" in NINT
    assert "copy_async_wait()" in NINT
    assert "column_base + k_local < input_width" in NINT
    assert "(maximum_tasks + 3) / 4" in NINT
    assert "masked_experts && maximum_tasks > block_cap" in NINT


def test_mfe_nint_route_density_covers_medium_prefill_with_the_same_kernel():
    assert "rows_per_expert <= 16" in RUNTIME
    assert "rows_per_expert <= 32" in RUNTIME
    assert "route.mma_tile_m == 16 || route.mma_tile_m == 32" in RUNTIME
    assert "route.mma_tile_m in (16, 32, 64)" in MOE_PYTHON


def test_mfe_nint_pool_phase_and_quantized_workspace_state_are_explicit():
    assert "static_cast<int64_t>(pool_phase) * 17" in NINT
    assert RUNTIME.count("nint_pool_phase++") >= 2
    assert MOE_PYTHON.count("nint_pool_phase += 1") == 2
    assert "if (!input_quantized && !use_tiled_prefill)" in MOE
    assert "input_quantized || route_tile_m == 8" in RUNTIME
    assert "input_quantized or nint_tile_m == 8" in MOE_PYTHON


def test_mfe_nint_projection_fuses_glu_in_the_unified_kernel():
    assert "int epilogue_mode" in NINT
    assert "mfq_glu_runtime(" in NINT
    assert "FragmentC accumulators[projections]" in NINT
    assert "constexpr int projections = FusedGlu ? 2 : 1;" in NINT
    assert "row * result_rows" in NINT
    assert "return forward_impl(x, route, false, gelu ? 2 : 1);" in RUNTIME
    assert "supports_projection_glu_epilogue()" in RUNTIME
    assert '"moe.gate_up_swiglu"' in RUNTIME


def test_retired_nint_kernel_family_is_not_compiled_or_bound():
    retired = (
        "nint_small_m",
        "nint_gemv_packed_bits",
        "nint_gemv_packed_batch",
        "nint_gemv_packed_int6",
        "nint_mmq_packed",
        "nint5_gs28_q5",
        "nint_dequant_full_packed_compact",
    )
    for token in retired:
        assert token not in NINT
        assert token not in BINDINGS
    assert "nint_small_m.cu" not in EXTENSION
    assert "nint_small_m.cu" not in CMAKE


def test_public_cuda_nint_surface_is_canonical():
    bound = set(re.findall(r'm\.def\("(nint[^\"]+)', BINDINGS))
    ordinary = {
        name
        for name in bound
        if not name.startswith("nint8_") and not name.startswith("nint_moe_")
    }
    assert ordinary == {
        "nint_cublas_gemm_nt_f16acc_cuda",
        "nint_backward_input_cuda",
        "nint_decode_cuda",
        "nint_embedding_cuda",
        "nint_matmul_input_mul_ws_cuda",
        "nint_matmul_ws_cuda",
    }


def test_cuda_input_gate_reuses_activation_quantizer_and_main_matmul():
    assert NINT.count("nint_matmul_input_mul_ws_cuda(") == 1
    assert "const __half * __restrict__ gate" in NINT
    assert "input, &gate, activation_mode, group_size" in NINT
    assert RUNTIME.count("nint_matmul_input_mul_ws_cuda(") == 2


def test_cpp_runtime_keeps_only_canonical_nint_row_state():
    nint_cpu = RUNTIME[RUNTIME.index("struct NintCpu {") : RUNTIME.index("struct Nint8ZeroCpu {")]
    nint_weight = RUNTIME[
        RUNTIME.index("struct NintWeight {") : RUNTIME.index("static NintWeight to_device_nint")
    ]
    assert "qbytes" not in nint_cpu
    assert "mixed_q" not in nint_cpu
    assert "mixed_q" not in nint_weight
    assert "row_sub_bits" in nint_cpu
    assert "aggregate_bpw" in nint_cpu
    assert "distribution_entropy" in nint_cpu
    assert "refresh_nint_descriptor" in RUNTIME
    assert "repack_nint_cpu_rows" in RUNTIME
    assert "copy_nint_packed_bits" in RUNTIME
    assert "mfe_nint_matmul_ws_cuda(" in RUNTIME
    for retired in (
        "MoeHeteroWorkspace",
        "Nint6MmqMode",
        "--nint6-mmq",
        "pure_nint_candidate_",
        "hetero_host_map_",
        "nint_moe_grouped_matmul_hetero",
        "initialize_mfe_dispatch",
    ):
        assert retired not in RUNTIME
    assert '.rfind("NINT", 0)' not in RUNTIME


def test_cpp_runtime_accepts_canonical_and_legacy_mfe_delta_magics():
    assert 'std::memcmp(blob.data(), "MFD1", 4) == 0' in RUNTIME
    assert 'std::memcmp(blob.data(), "NID2", 4) == 0' in RUNTIME


def test_metal_nint_uses_one_metadata_driven_compute_kernel():
    assert METAL_NINT.count('"mfq_cpp_nint_runtime_"') == 1
    assert "row_metadata[row_metadata_base]" in METAL_NINT
    assert "row_metadata[row_metadata_base + 1u]" in METAL_NINT
    assert "auto output = routed_matmul(" in METAL_NINT
    assert METAL_GROUPED.count("mfq_cpp_nint_metadata_grouped_p") == 1
    assert "mfq_grouped_nint_read_row_value4" in METAL_GROUPED
    for retired in (
        "mfq_cpp_single_row_grouped_nint",
        "partitioned_nint4_qkv_kernel",
        "mfq_cpp_interleaved_grouped_nint4",
        "has_single_row_nint_fast_path",
        "mfq_grouped_nint_read_bits",
        "mfq_grouped_nint_read_value",
        "constexpr int kFamilyNint =",
        "direct_small_m_qkv_kernel",
    ):
        assert retired not in METAL_GROUPED


def test_python_metal_mfe_reuses_the_ordinary_nint_kernel():
    assert METAL_NINT_PYTHON.count('"mfq_nint_matmul"') == 1
    assert "def nint_routed_matmul(" in METAL_NINT_PYTHON
    assert "nint_routed_matmul(" in METAL_MOE_PYTHON
    assert "routed_to_nint" in METAL_MOE_PYTHON
    assert "mfq_nint4_grouped_qmv" not in METAL_MOE_PYTHON
    assert "_GROUPED_NINT4_QMV" not in METAL_MOE_PYTHON


def test_python_ordinary_grouped_linear_keeps_nint_out_of_heterogeneous_kernel():
    assert "MetalNintLinearGroupWeight.from_weights" in MLX_LINEAR_PYTHON
    assert "elif not contains_nint:" in MLX_LINEAR_PYTHON
    assert "mfq_nint_metadata_grouped_p" in METAL_GROUPED_PYTHON
    assert "mfq_grouped_linear_decode_weight" in METAL_GROUPED_PYTHON
    assert "mfq_grouped_decode_weight" not in METAL_GROUPED_PYTHON
    assert '"nint_q"' not in METAL_GROUPED_PYTHON
    assert "_FAMILY_NINT," not in METAL_GROUPED_PYTHON


def test_metal_mfe_has_no_second_dense_nint_compute_kernel():
    assert "mfq_dense_nint" not in METAL_MFE_KERNELS
    assert "descriptors[base + kNintV2] = 1;" in METAL_MOE
    assert "auto weight = add_nint_pool(" in METAL_MOE
    assert "family == kFamilyNint" in METAL_MOE
    assert "decode_nint_row_quad_at(" in METAL_MFE_KERNELS
    assert "nint_cohorts" not in METAL_MOE
    assert "mfq_moe_nint_" not in METAL_MOE
    assert "if (family == 0u)" not in METAL_MOE
    assert "grouped_nint4_group24" not in METAL_MOE
    assert "nint_profile_mask" not in METAL_MOE
    assert METAL_NINT.count("constexpr const char* kNintMatmul =") == 1
    assert METAL_NINT.count("kNintMatmul,") == 1
    assert METAL_NINT.count("compiled_nint_matmul_kernel({") == 2
    assert "? kNintStoreOutputAdd" in METAL_NINT
    assert ": kNintStoreOutput" in METAL_NINT
    assert "make_nint_matmul_kernel" not in METAL_NINT
    assert "make_nint_matmul_add_kernel" not in METAL_NINT
