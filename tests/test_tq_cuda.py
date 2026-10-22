from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.tq import TQTensor, fixed_codebook, pack_states, unpack_states  # noqa: E402
from mfq.kernels.cuda.tq import Projection, SwiGLU, SwiGLUFFN  # noqa: E402
from mfq.quantize.cuda.tq_encoder import encode  # noqa: E402
from mfq.quantize.tq import quantize  # noqa: E402


def fixture(rows, width, columns=False, seed=7):
    rng = np.random.default_rng(seed)
    stored = (width + 255) // 256 * 256
    bits = (rng.permutation(np.arange(rows)) % 8 + 1).astype(np.uint64)
    desc = np.r_[np.uint64(0), np.cumsum(stored * bits[:-1] // 32)]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(.015).view(np.uint16)) << np.uint64(40)
    n = (rows * ((stored + 511) // 512) + 1) // 2
    scales = np.ones(stored, np.float16) if columns else rng.integers(0, 256, n, dtype=np.uint8)
    data = rng.integers(0, 256, int((stored * bits // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, stored, 512, columns, desc, scales, data)


def rms(a, b):
    return ((a.float() - b.float()).square().mean() / b.float().square().mean().clamp_min(1e-20)).sqrt().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,width,columns", [(19, 257, False), (7, 513, False), (1, 37, True), (2048, 4096, False), (4096, 2048, True)])
def test_gemv(dtype, rows, width, columns):
    torch.backends.cuda.matmul.allow_tf32 = False
    tensor = fixture(rows, width, columns)
    w = torch.tensor(tensor.dequantize(width), device="cuda").float()
    bias = torch.linspace(-.02, .02, rows, device="cuda")
    op = Projection(tensor, input_columns=width, bias=bias)
    x = torch.randn(width + 1, device="cuda", dtype=dtype)[1:]
    output = torch.full((rows + 2,), 42, device="cuda", dtype=dtype)
    expected = w @ x.float() + bias
    for warps, slots in ((0, 0), (1, 4), (2, 2), (4, 1), (8, 1), (16, 1), (32, 1)):
        op.gemv(x, output[1:-1], warps=warps, rows_per_block=slots)
        assert rms(output[1:-1], expected) < (.001 if dtype == torch.float16 else .01)
        assert output[0] == 42 and output[-1] == 42


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,hidden,output", [(257, 19, 37), (4096, 2048, 4096)])
def test_ffn_graph(dtype, width, hidden, output):
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = [fixture(hidden, width, False, 11), fixture(hidden, width, True, 22), fixture(output, hidden, True, 33)]
    g, u, d = [Projection(t, input_columns=k) for t, k in zip(tensors, (width, width, hidden), strict=True)]
    gw, uw, dw = [torch.tensor(t.dequantize(k), device="cuda").float() for t, k in zip(tensors, (width, width, hidden), strict=True)]
    x = torch.randn(width, device="cuda", dtype=dtype) * .2
    y = torch.empty(output, device="cuda", dtype=dtype)
    workspace = torch.empty(hidden, device="cuda", dtype=dtype)
    gr, ur = (gw @ x.float()).to(dtype).float(), (uw @ x.float()).to(dtype).float()
    hr = (ur * torch.nn.functional.silu(gr)).to(dtype)
    expected = dw @ hr.float()
    swiglu, ffn = SwiGLU(g, u), SwiGLUFFN(g, u, d)
    swiglu(x, workspace)
    assert rms(workspace, hr) < .01
    ffn(x, y, workspace=workspace)
    assert rms(y, expected) < .015
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ffn(x, y, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ffn(x, y, workspace=workspace)
    graph.replay()
    assert rms(y, expected) < .015


@pytest.mark.parametrize("bits", range(1, 9))
def test_encoder_ring(bits):
    torch.manual_seed(4817)
    x = torch.randn(3, 256, device="cuda")
    metric = torch.exp(torch.randn_like(x) * .2)
    lut = fixed_codebook()
    value, states, _ = encode(x, metric, torch.tensor(lut, device="cuda"), bits)
    raw = states.cpu().numpy().view(np.uint16)
    np.testing.assert_array_equal(unpack_states(pack_states(raw, bits), bits), raw)
    np.testing.assert_array_equal(value.cpu().numpy(), lut[raw])


@pytest.mark.parametrize("bits", [1, 3, 8])
def test_encoder_matches_cpu_constrained_objective(bits):
    """Independent NumPy recurrence checks the exact constrained path objective."""
    rng = np.random.default_rng(7201 + bits)
    target = rng.normal(size=256).astype(np.float32)
    weight = np.exp(rng.normal(size=256) * .2).astype(np.float32)
    lut = fixed_codebook()
    values, _, boundary = encode(torch.tensor(target[None], device="cuda"),
                                 torch.tensor(weight[None], device="cuda"),
                                 torch.tensor(lut, device="cuda"), bits)
    edges = 1 << (16 - bits)
    states = np.arange(65536)
    costs = np.full(edges, np.inf, dtype=np.float32)
    start = boundary.item()
    costs[start] = 0
    for x, m in zip(target, weight, strict=True):
        delta = (lut.astype(np.float32) - x) * m
        # CUDA uses an FP32 fused multiply-add for delta^2 + predecessor cost.
        trial = (delta.astype(np.float64)**2 + costs[states >> bits]).astype(np.float32)
        costs = trial.reshape(1 << bits, edges).min(axis=0)
    decoded = values.cpu().numpy()[0]
    actual = (((decoded.astype(np.float64) - target) * weight)**2).sum()
    assert actual == pytest.approx(float(costs[start]), rel=1e-5, abs=1e-6)


def test_mixed_row_quantization_roundtrip():
    torch.manual_seed(123)
    w = torch.randn(8, 257, device="cuda") * .02
    h = torch.linspace(.1, 2, 257, device="cuda")
    tensor = quantize(w, np.arange(1, 9), imatrix=h, groupsize=512, scale_factors=(1.,))
    np.testing.assert_array_equal(tensor.bits, np.arange(1, 9))
    assert tensor.to_bytes() == TQTensor.from_bytes(tensor.to_bytes()).to_bytes()
    op = Projection(tensor, input_columns=257)
    x = torch.randn(257, dtype=torch.float16, device="cuda")
    expected = torch.tensor(tensor.dequantize(257), device="cuda").float() @ x.float()
    assert rms(op.gemv(x), expected) < .001
