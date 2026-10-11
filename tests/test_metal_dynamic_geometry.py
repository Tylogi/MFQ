from pathlib import Path

import pytest


METAL = Path(__file__).resolve().parents[1] / "cpp_runtime/backends/metal"


@pytest.mark.parametrize(
    ("filename", "forbidden"),
    [
        ("runtime/mlx_sampling.cpp", ['{"TOKENS",', '{"SIZE",']),
        ("ops/mlx_nint8_zero.cpp", ['{"M",', '{"COUNT",', '{"TOKENS",']),
        ("ops/mlx_dsa.cpp", ['{"M",', '{"K",', '{"KEY_TILES",', '{"TOTAL",',
                             '{"SELECTED",', '{"POOL_LEN",']),
        ("ops/mlx_sparse_attention.cpp", ['{"M",', '{"MAX_SEQ",', '{"SELECTED",',
                                          '{"POOL_CAPACITY",']),
        ("models/deepseek_v4/mlx_deepseek_v4_hc.cpp", ['{"ROWS",', '{"SIZE",',
                                                    '{"REDUCED_BYTES",']),
        ("models/deepseek_v41/mlx_deepseek_v41_mhc.cpp", ['{"ROWS",', '{"SIZE",']),
        ("models/deepseek_v4/mlx_deepseek_v4_attention.cpp", ['{"ROWS",', '{"TOKENS",',
                                                           '{"M",']),
    ],
)
def test_request_lengths_do_not_enter_kernel_template_keys(filename, forbidden):
    source = (METAL / filename).read_text()
    for parameter in forbidden:
        assert parameter not in source, (filename, parameter)


@pytest.mark.parametrize(
    ("filename", "kernel", "shape"),
    [
        ("ops/mlx_vq.cpp", "kMatmulSource", "x_shape[0]"),
        ("ops/mlx_vq.cpp", "kMmqSource", "x_shape[0]"),
        ("ops/mlx_vq.cpp", "kEmbeddingSource", "token_ids_shape[0]"),
        ("ops/mlx_vq.cpp", "kHadamardSource", "x_shape[0]"),
        ("ops/mlx_vq.cpp", "kResidualMatmulSource", "x_shape[0]"),
        ("ops/mlx_mx.cpp", "kMxMatmul", "x_shape[0]"),
        ("ops/mlx_mx.cpp", "kMxEmbedding", "x_shape[0]"),
        ("ops/mlx_mx.cpp", "kWeightedRmsRopeMxfp8SimSource", "cos_values_shape[0]"),
        ("ops/mlx_fp8_sq.cpp", "kFp8SqMatmul", "expert_ids_shape[0]"),
        ("ops/mlx_mxfp4_sq.cpp", "kSqMatmul", "expert_ids_shape[0]"),
    ],
)
def test_quantized_kernels_read_runtime_shapes(filename, kernel, shape):
    source = (METAL / filename).read_text()
    body = source.split(f"constexpr const char* {kernel} =", 1)[1].split(')METAL"', 1)[0]
    assert shape in body


def test_both_fp8_sq_formats_use_the_shared_shape_driven_kernel():
    source = (METAL / "ops/mlx_fp8_sq.cpp").read_text()
    assert '"mfq_cpp_mxfp8_sq_matmul", kFp8SqMatmul, true' in source
    assert '"mfq_cpp_fp8_128_sq_matmul", kFp8SqMatmul, true' in source


def test_mfe_shared_body_has_no_custom_kernel_shape_dependencies():
    source = (METAL / "ops/mlx_moe.cpp").read_text()
    body = source.split('constexpr const char* kMoeSource = R"METAL(', 1)[1].split(')METAL"', 1)[0]
    assert "expert_ids_shape" not in body
    assert '"const int TOKENS = expert_ids_shape[0];' in source
    assert '{"TOKENS", tokens}' not in source
    assert '{"VARIANT_STRIDE", variant_stride}' not in source
