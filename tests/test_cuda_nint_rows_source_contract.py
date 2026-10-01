"""Wiring gates only; real CUDA execution is in the native row and graph tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_row_kernel_consumes_packed_q_and_k():
    source = (ROOT / "mfq/kernels/cuda/embedding.cu").read_text()
    kernel = source.split("__global__ void nint_selected_rows_kernel(", 1)[1].split(
        "__global__ void nint8_zero_embedding_kernel(", 1)[0]
    assert "qbits = layout & 15u" in kernel
    assert "kbits = (layout >> 4u) & 15u" in kernel
    assert kernel.count("unpack_nint_code(") == 3
    assert "__ushort_as_half" in kernel
    assert "subgroup_scale" not in kernel
    assert "__fsub_rn(__fmul_rn(" in kernel


def test_cuda_ple_nint_uses_retained_selected_row_source():
    root = ROOT / "cpp_runtime/backends/cuda"
    layers = (root / "models/flash_next/flash_next_layers.h").read_text()
    branch = layers.split('if (require_tensor(file,name).dtype=="NINT")', 1)[1].split(
        "continue;", 1)[0]
    assert "load_nint_row_table(file,name)" in branch
    assert "nint_row_embedding_lookup(*table,ids)" in branch
    assert "load_quant_linear" not in branch
    ops = (root / "ops/cuda_quantized_ops.cpp").read_text()
    lookup = ops.split("mfq_tensor_backend::Tensor nint_row_embedding_lookup(", 1)[1].split(
        "mfq_tensor_backend::Tensor quant_embedding_lookup(", 1)[0]
    assert "table.append_row(" in lookup and "selected.packed()" in lookup
    assert "nint_selected_rows_cuda(" in lookup
    assert "cudaStreamIsCapturing" in lookup
    assert "sub_scale" not in lookup


def test_shared_row_parser_and_cuda_test_are_build_targets():
    core = (ROOT / "cpp_runtime/core/CMakeLists.txt").read_text()
    cmake = (ROOT / "cpp_runtime/cmake/CudaRuntime.cmake").read_text()
    assert "nint_rows.cpp" in core
    assert "mfq-nint-rows-cuda-test" in cmake
    assert "mfq_nint_rows_test.cu" in cmake
