"""ROCm TQ projection, input-gradient and fitting checks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("ROCm unavailable", allow_module_level=True)

from mfq.formats.tq import TQTensor, fixed_codebook, pack_states, unpack_states
from mfq.kernels.hip.tq import Projection, SwiGLUFFN
from mfq.quantize.cuda.tq_encoder import encode
from mfq.quantize.cuda.tq_fitter import pack
from mfq.quantize.tq import quantize


def fixture(rows, width, columns=False, groupsize=512):
    rng = np.random.default_rng(rows + width)
    stored = (width + 255) // 256 * 256
    bits = (rng.permutation(rows) % 8 + 1).astype(np.uint64)
    desc = np.r_[np.uint64(0), np.cumsum(stored * bits[:-1] // 32)]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(.015).view(np.uint16)) << np.uint64(40)
    n = (rows * ((stored + groupsize - 1) // groupsize) + 1) // 2
    scales = (rng.uniform(.2, 2, stored).astype(np.float16) if columns else
              rng.integers(0, 256, n, dtype=np.uint8))
    data = rng.integers(0, 256, int((stored * bits // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, stored, groupsize, columns, desc, scales, data)


def close(actual, expected):
    tol = {torch.float16: .002, torch.bfloat16: .015, torch.float32: 2e-5}[actual.dtype]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol * .1)


@pytest.mark.parametrize("columns,groupsize", [(False, 16), (False, 512), (True, 512)])
def test_exact_weight_reconstruction(columns, groupsize):
    tensor = fixture(19, 273, columns, groupsize)
    op = Projection(tensor, input_columns=273)
    expected = torch.tensor(tensor.dequantize(273), device="cuda").float()
    torch.testing.assert_close(op(torch.eye(273, device="cuda")).T, expected, rtol=0, atol=0)
    torch.testing.assert_close(op.backward(torch.eye(19, device="cuda")), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 2, 17, 33])
@pytest.mark.parametrize("columns,groupsize", [(False, 16), (False, 512), (True, 512)])
def test_forward_backward(dtype, tokens, columns, groupsize):
    tensor = fixture(19, 273, columns, groupsize)
    w = torch.tensor(tensor.dequantize(273), device="cuda").float()
    bias = torch.linspace(-.02, .02, 19, device="cuda")
    op = Projection(tensor, input_columns=273, bias=bias)
    shape = (273,) if tokens == 1 else (tokens, 273)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn((*shape[:-1], 19), device="cuda", dtype=dtype)
    y = op(x)
    close(y, x.detach().float() @ w.T + bias)
    y.backward(dy)
    close(x.grad, dy.float() @ w)
    close(op.backward(dy), dy.float() @ w)
    if tokens == 1:
        close(op.gemv(x.detach()), y.detach())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 17])
def test_graph_output_guards_and_changed_inputs(dtype, tokens):
    tensor = fixture(19, 257)
    op = Projection(tensor, input_columns=257)
    w = torch.tensor(tensor.dequantize(257), device="cuda").float()
    x = torch.randn(tokens, 257, device="cuda", dtype=dtype)
    dy = torch.randn(tokens, 19, device="cuda", dtype=dtype)
    storage = torch.full((tokens * 19 + 2,), 42, device="cuda", dtype=dtype)
    out = storage[1:-1].reshape(tokens, 19)
    gradient = torch.full((tokens * 257 + 2,), 42, device="cuda", dtype=dtype)
    dx = gradient[1:-1].reshape_as(x)
    op.forward(x, out)
    op.backward(dy, dx)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.forward(x, out)
        op.backward(dy, dx)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward(x, out)
        op.backward(dy, dx)
    x.add_(.3)
    dy.mul_(.4)
    graph.replay()
    close(out, x.float() @ w.T)
    close(dx, dy.float() @ w)
    assert storage[0] == 42 and storage[-1] == 42
    assert gradient[0] == 42 and gradient[-1] == 42


@pytest.mark.parametrize("tokens", [1, 17])
def test_ffn_training(tokens):
    tensors = [fixture(19, 257), fixture(19, 257, True), fixture(37, 19)]
    ops = [Projection(t, input_columns=k) for t, k in zip(tensors, (257, 257, 19), strict=True)]
    gw, uw, dw = [torch.tensor(t.dequantize(k), device="cuda").float()
                  for t, k in zip(tensors, (257, 257, 19), strict=True)]
    x = torch.randn(tokens, 257, device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    expected = (torch.nn.functional.silu(reference @ gw.T) * (reference @ uw.T)) @ dw.T
    actual = SwiGLUFFN(*ops)(x)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    close(actual, expected)
    close(x.grad, reference.grad)


def test_validation_empty_batch_and_aliases():
    op = Projection(fixture(19, 19), input_columns=19)
    x = torch.ones(19, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        op.forward(x, x)
    with pytest.raises(ValueError, match="width"):
        op(torch.ones(20, device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        op(torch.ones(19, 2, device="cuda")[:, 0])
    with pytest.raises(ValueError, match="gradients"):
        op.forward(x.requires_grad_(), torch.empty_like(x))
    assert op(torch.empty(0, 19, device="cuda")).shape == (0, 19)
    assert op(torch.empty(2, 3, 19, device="cuda")).shape == (2, 3, 19)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 32])
def test_large_projection(dtype, tokens):
    tensor = fixture(2048, 4096)
    op = Projection(tensor, input_columns=4096)
    w = torch.tensor(tensor.dequantize(), device="cuda").float()
    x = torch.randn(tokens, 4096, device="cuda", dtype=dtype)
    dy = torch.randn(tokens, 2048, device="cuda", dtype=dtype)
    actual, gradient = op(x), op.backward(dy)
    for value, expected in ((actual, x.double() @ w.double().T),
                            (gradient, dy.double() @ w.double())):
        error = (value.double() - expected).square().mean().sqrt()
        scale = expected.square().mean().sqrt()
        limit = {torch.float16: .001, torch.bfloat16: .004, torch.float32: 3e-6}[dtype]
        assert float(error / scale) < limit


@pytest.mark.parametrize("bits", range(1, 9))
def test_encoder_and_packing(bits):
    torch.manual_seed(bits)
    x = torch.randn(3, 256, device="cuda")
    metric = torch.rand_like(x) + .1
    lut = fixed_codebook()
    y, states, _ = encode(x, metric, torch.tensor(lut, device="cuda"), bits)
    raw = states.cpu().numpy().view(np.uint16)
    packed = pack(states, bits).cpu().numpy().reshape(3, 32 * bits)
    np.testing.assert_array_equal(packed, pack_states(raw, bits))
    np.testing.assert_array_equal(unpack_states(packed, bits), raw)
    np.testing.assert_array_equal(y.cpu().numpy(), lut[raw])


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("groupsize", [16, 512])
def test_adaptive_fitting(bits, groupsize):
    torch.manual_seed(123)
    w = torch.randn(4, 273, device="cuda") * .02
    w[0] = 0
    h = torch.linspace(.2, 2, 273, device="cuda")
    h[::7] = 0
    exact = quantize(w, bits, imatrix=h, groupsize=groupsize,
                     column_scaling=False, search="full")
    adaptive = quantize(w, bits, imatrix=h, groupsize=groupsize,
                        column_scaling=False, search="adaptive")
    scores = []
    for tensor in (exact, adaptive):
        assert tensor.to_bytes() == TQTensor.from_bytes(tensor.to_bytes()).to_bytes()
        decoded = torch.tensor(tensor.dequantize(273), device="cuda").float()
        scores.append(((decoded - w).square() * h).sum().item())
    assert scores[1] <= scores[0] * 1.01
    raw = unpack_states(adaptive.packed.reshape(-1, 32 * bits), bits)
    np.testing.assert_array_equal(raw >> bits,
                                  np.roll(raw, 1, axis=1) & ((1 << (16 - bits)) - 1))


def test_mixed_row_fitting_and_projection():
    w = torch.randn(8, 257, device="cuda") * .02
    tensor = quantize(w, np.arange(1, 9), groupsize=512)
    np.testing.assert_array_equal(tensor.bits, np.arange(1, 9))
    x = torch.randn(3, 257, device="cuda")
    op = Projection(tensor, input_columns=257)
    close(op(x), x @ torch.tensor(tensor.dequantize(257), device="cuda").float().T)
