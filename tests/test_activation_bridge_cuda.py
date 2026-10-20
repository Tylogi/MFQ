"""Small cross-platform numerical gate for the production Torch activation bridge."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.kernels.cuda._ext import ext


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("kind", ["silu", "gelu"])
@pytest.mark.parametrize("shape", [(0,), (33, 71)])
def test_activation_bridge(dtype, kind, shape):
    torch.manual_seed(20260907)
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    module = ext()
    operation = getattr(module, f"{kind}_mul_cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = operation(gate, up)
    torch.cuda.current_stream().wait_stream(stream)
    g = gate.float()
    activation = (torch.nn.functional.gelu(g, approximate="tanh")
                  if kind == "gelu" else torch.nn.functional.silu(g))
    expected = (activation * up.float()).to(dtype)
    tolerance = 1e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_gelu_half_saturation():
    gate = torch.tensor([200., 200., -200.], device="cuda", dtype=torch.float16)
    up = torch.tensor([400., -400., 400.], device="cuda", dtype=torch.float16)
    expected = torch.tensor([65504., -65504., 0.], device="cuda", dtype=torch.float16)
    actual = ext().gelu_mul_cuda(gate, up)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)


def test_silu_bf16_staged_rounding():
    # silu_mul_bf16_kernel rounds SiLU(gate) to BF16 before the product, so a
    # single-rounding reference is a different value on this fixture.  Every gate
    # is BF16-exact and keeps >=40 float32 ulps of headroom before its BF16
    # rounding changes, so the staged reference can be compared exactly.
    gate = torch.tensor([-7.9375, -5.21875, -2.65625, -0.1943359375, 2.40625, 7.875],
                        device="cuda", dtype=torch.bfloat16)
    up = torch.tensor([3.0, 3.0, 2.5, 7.0, 2.5, 2.5],
                      device="cuda", dtype=torch.bfloat16)
    staged = (torch.nn.functional.silu(gate.float()).to(torch.bfloat16).float()
              * up.float()).to(torch.bfloat16)
    single = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    assert not torch.equal(staged, single), "fixture must separate the two roundings"
    actual = ext().silu_mul_cuda(gate, up)
    torch.testing.assert_close(actual, staged, atol=0., rtol=0.)
    assert not torch.equal(actual, single), "kernel did not stage the BF16 rounding"


def test_silu_large_half_vector_and_offset_paths():
    # 1<<18 elements is the half2 threshold and the odd count also exercises the
    # scalar tail; base[1:] is contiguous but only 2-byte aligned, which forces
    # the scalar launch for the same element count.
    elements = (1 << 18) + 1
    torch.manual_seed(20260909)
    base = torch.randn(elements + 1, device="cuda", dtype=torch.float16)
    up = torch.randn(elements, device="cuda", dtype=torch.float16)
    module = ext()
    for gate in (base[:elements], base[1:]):
        assert gate.is_contiguous()
        expected = (torch.nn.functional.silu(gate.float())
                    * up.float()).to(torch.float16)
        torch.testing.assert_close(
            module.silu_mul_cuda(gate, up), expected, atol=1e-3, rtol=1e-3)


def test_silu_matches_under_cuda_graph_capture():
    torch.manual_seed(20260910)
    gate = torch.randn(33, 71, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    module = ext()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module.silu_mul_cuda(gate, up)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module.silu_mul_cuda(gate, up)
    # Replay must read the live inputs, not the values captured during warmup.
    gate.fill_(-1.75)
    up.fill_(2.25)
    graph.replay()
    torch.cuda.synchronize()
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)
    torch.testing.assert_close(captured, expected, atol=1e-3, rtol=1e-3)


def test_activation_bridge_rejects_invalid_inputs():
    module = ext()
    half = torch.randn(8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="shapes must match"):
        module.silu_mul_cuda(half, torch.randn(9, device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="dtype must match"):
        module.silu_mul_cuda(half, torch.randn(8, device="cuda", dtype=torch.float32))
    with pytest.raises(RuntimeError, match="dtype must be f32 or f16"):
        module.gelu_mul_cuda(half.to(torch.bfloat16), half.to(torch.bfloat16))
