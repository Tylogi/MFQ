from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_formats_keep_distinct_scale_kernels() -> None:
    source = (ROOT / "mfq/kernels/cuda/fp8_sq.cu").read_text()
    header = (ROOT / "mfq/kernels/cuda/fp8_sq.h").read_text()
    assert source.count("__global__ void mxfp8_sq_mmq_kernel(") == 1
    assert source.count("__global__ void fp8_128_sq_mmq_kernel(") == 1
    assert "decode_e8m0" in source
    assert "decode_fp8_128_scale" in source
    assert "mxfp8_sq_matmul_cuda" in header
    assert "fp8_128_sq_matmul_cuda" in header
    assert "mxfp8_sq_backward_input_cuda" in header
    assert "fp8_128_sq_backward_input_cuda" in header

    metal = (
        ROOT / "cpp_runtime/backends/metal/ops/mlx_fp8_sq.cpp"
    ).read_text()
    assert metal.count('"mfq_cpp_mxfp8_sq_backward_matrix"') == 1
    assert metal.count('"mfq_cpp_fp8_128_sq_backward_matrix"') == 1
    assert metal.count("constexpr const char* kFp8SqBackwardMatrix") == 1
    assert "row_q[output]" in metal
    assert "row_symbol_byte_offsets[output]" in metal


def test_cuda_dense_dispatch_keeps_packed_decode_and_transient_gemm() -> None:
    source = (ROOT / "mfq/kernels/cuda/fp8_sq.cu").read_text()
    assert "constexpr int kDirectPackedMaxRows = 48;" in source
    assert "if (rows > kDirectPackedMaxRows)" in source
    assert "auto weight = dequant<MXFP8>(" in source
    assert "launch_dense_gemm_nt(input, weight, output, stream);" in source
    assert "CUBLAS_OP_T, CUBLAS_OP_N" in source
    assert "never cached or attached to the packed weight" in source


def test_cuda_runtime_routes_dense_and_mfe_without_model_branches() -> None:
    ops = ROOT / "cpp_runtime/backends/cuda/ops"
    source = "\n".join(
        path.read_text()
        for path in sorted(ops.rglob("*"))
        if path.suffix in {".h", ".cpp"}
    )
    assert 'weight.dtype == "MXFP8-SQ"' in source
    assert 'weight.dtype == "FP8-128SQ"' in source
    assert "mxfp8_sq_moe_matmul_cuda(" in source
    assert "fp8_128_sq_moe_matmul_cuda(" in source
    fp8_section = (ops / "fp8_sq.cpp").read_text()
    for architecture in ("qwen", "deepseek", "glm", "gemma"):
        assert architecture not in fp8_section.lower()


def test_python_cuda_mfe_registers_both_fp8_sq_families() -> None:
    bindings = (ROOT / "mfq/kernels/cuda/mfq_cuda.cpp").read_text()
    runtime = (ROOT / "mfq/kernels/cuda/moe.py").read_text()
    for name in ("mxfp8_sq_moe_matmul_cuda", "fp8_128_sq_moe_matmul_cuda"):
        assert f'm.def("{name}"' in bindings
        assert f"ext().{name}(" in runtime
    assert 'family = "mxfp8_sq"' in runtime
    assert 'family = "fp8_128_sq"' in runtime


def test_cuda_builds_include_fp8_sq_once() -> None:
    cmake = (ROOT / "cpp_runtime/backends/cuda/CMakeLists.txt").read_text()
    extension = (ROOT / "mfq/kernels/cuda/_ext.py").read_text()
    assert cmake.count("${MFQ_CUDA_KERNEL_ROOT}/fp8_sq.cu") == 1
    assert extension.count('os.path.join(_DIR, "fp8_sq.cu")') == 1


def test_wire_parser_and_runtime_share_the_three_bit_descriptor() -> None:
    parser = (
        ROOT / "cpp_runtime/core/include/mfq/fp8_sq_blob.h"
    ).read_text()
    cuda = (ROOT / "mfq/kernels/cuda/fp8_sq.cu").read_text()
    assert "row * 3" in parser
    assert "read_q(data + kHeaderBytes" in parser
    assert "const int bits = static_cast<int>(row_q[output]);" in cuda
    assert "if (bits == 8)" in cuda
    assert "row_symbol_byte_offsets[output]" in cuda
