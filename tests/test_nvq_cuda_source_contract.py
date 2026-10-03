from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NVQ = (ROOT / "mfq/kernels/cuda/nvq_matmul.cu").read_text()
ASYNC_COPY = (ROOT / "mfq/kernels/cuda/async_copy.cuh").read_text()
CUDA_OPS = ROOT / "cpp_runtime/backends/cuda/ops"
RUNTIME = "\n".join(
    path.read_text()
    for path in sorted(CUDA_OPS.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)


def test_nvq_moe_overlaps_activation_loads_with_weight_decode() -> None:
    assert 'include "async_copy.cuh"' in NVQ
    assert "cp.async.ca.shared.global" in ASYNC_COPY
    assert "copy_16_async(destination, source)" in NVQ
    assert "mfq::cuda_detail::copy_async_commit();" in NVQ
    assert "mfq::cuda_detail::copy_async_wait();" in NVQ
    assert "FORMAT_GROUP_VALUE, true><<<" in NVQ


def test_nvq_moe_skips_padding_and_adapts_tiles_to_route_density() -> None:
    assert "if (BM < 64 || k_base + k_local < K)" in NVQ
    assert "const bool use_narrow_tile = tile_m == 128" in NVQ
    assert "const int64_t useful_tasks = useful_tiles * ntiles_n;" in NVQ
    assert "128, 64, 128, 4" in NVQ
    assert "128, 128, 128, 2" in NVQ


def test_nvq_moe_rotates_masked_and_multi_pool_task_grids() -> None:
    assert "if (masked_experts && tile_m != 8" in NVQ
    assert "block_cap += task_period;" in NVQ
    assert "else if (pools > 1 && task_period >= 256)" in NVQ
    assert "++block_cap;" in NVQ
    assert "dispatch->masked_experts = owned_experts < runtime.n_experts;" in RUNTIME


def test_nvq_moe_specializes_resident_prefill_by_format_group() -> None:
    for group in ("Standard", "Extended", "Legacy"):
        assert f"MixedNvqF16FormatGroup::{group}" in RUNTIME
        assert f"kNvqMoeF16{group}Formats" in NVQ
    assert "mixed_nvq_f16_format_group(int format)" in RUNTIME
    assert "static_cast<int>(nvq_dispatch->f16_format_group)" in RUNTIME
    assert "int64_t format_group" in NVQ
    assert "nvq23_only" not in NVQ
    assert "nvq23_only" not in RUNTIME
