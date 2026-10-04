"""CUDA MoE routing and expert-wise NINT grouped matmul tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.kernels.cuda._ext import ext  # noqa: E402
from mfq.kernels.cuda.moe import (  # noqa: E402
    MoeRoutePlan,
    add_shared_gate,
    geglu_split,
    grouped_matmul,
    sqrtsoftplus_weights,
    swiglu_split,
    to_gpu,
    topk,
    weighted_reduce,
    weighted_reduce_shared_gate,
)
from mfq.quantize.expert_nint import (  # noqa: E402
    dequantize_expertwise,
    quantize_expertwise,
)


def _mixed_weight(seed: int = 0, *, experts: int = 6, out: int = 10, k: int = 73):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.05, size=(experts, out, k)).astype(np.float32)
    catalog = (
        NintSpec(2, 16, 5),
        NintSpec(4, 24, 6),
        NintSpec(6, 24, 6),
        NintSpec(8, 24, 8),
        NintSpec(5, 28, 6),
    )
    specs = tuple(catalog[index % len(catalog)] for index in range(experts))
    tensor = quantize_expertwise(values, specs)
    return tensor, to_gpu(tensor)


def _ids(tokens: int, experts: int, routes: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(991 + tokens)
    scores = torch.randn(tokens, experts, device="cuda", generator=generator)
    return scores.topk(routes, dim=-1).indices.to(torch.int32).contiguous()


def _reference(weight, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    dense = torch.as_tensor(dequantize_expertwise(weight), device="cuda", dtype=torch.float32)
    tokens, routes = ids.shape
    out = dense.shape[1]
    result = torch.empty(tokens, routes, out, device="cuda", dtype=torch.float32)
    for token in range(tokens):
        for route in range(routes):
            expert = int(ids[token, route])
            source = x[token, route] if x.ndim == 3 else x[token]
            result[token, route] = source.float() @ dense[expert].T
    return result.to(torch.float16)


def test_moe_route_plan_builds_fine_coarse_and_wide_maps():
    experts = 17
    ids = _ids(1024, experts, 8)
    route = MoeRoutePlan.build(ids, experts)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=experts).to(torch.int32)

    assert route.mma_tile_m == 64
    assert route.wide_tile_m == 128
    torch.testing.assert_close(route.counts, counts, rtol=0, atol=0)
    for tile_m, bounds, tile_experts in (
        (8, route.tile_bounds, route.tile_experts),
        (64, route.mma_tile_bounds, route.mma_tile_experts),
        (128, route.wide_tile_bounds, route.wide_tile_experts),
    ):
        tiles = torch.div(counts + tile_m - 1, tile_m, rounding_mode="floor")
        expected_bounds = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                tiles.cumsum(0).to(torch.int32),
            )
        )
        expected_experts = torch.repeat_interleave(
            torch.arange(experts, device="cuda", dtype=torch.int32), tiles.long()
        )
        torch.testing.assert_close(bounds, expected_bounds, rtol=0, atol=0)
        torch.testing.assert_close(
            tile_experts[: expected_experts.numel()], expected_experts, rtol=0, atol=0
        )

    legacy_maps = ext().moe_build_expert_maps_cuda(ids, experts, 8, 64)
    assert len(legacy_maps) == 7
    (
        legacy_ids_dst,
        legacy_expert_bounds,
        legacy_tile_bounds,
        legacy_tile_experts,
        legacy_counts,
        legacy_mma_bounds,
        legacy_mma_experts,
    ) = legacy_maps
    torch.testing.assert_close(legacy_counts, counts, rtol=0, atol=0)
    expected_expert_bounds = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            counts.cumsum(0).to(torch.int32),
        )
    )
    torch.testing.assert_close(
        legacy_expert_bounds, expected_expert_bounds, rtol=0, atol=0
    )
    grouped_experts = ids.reshape(-1).index_select(0, legacy_ids_dst.long())
    expected_grouped_experts = torch.repeat_interleave(
        torch.arange(experts, device="cuda", dtype=torch.int32), counts.long()
    )
    torch.testing.assert_close(grouped_experts, expected_grouped_experts, rtol=0, atol=0)
    torch.testing.assert_close(
        legacy_ids_dst.sort().values,
        torch.arange(ids.numel(), device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    for tile_m, bounds, tile_experts in (
        (8, legacy_tile_bounds, legacy_tile_experts),
        (64, legacy_mma_bounds, legacy_mma_experts),
    ):
        tiles = torch.div(counts + tile_m - 1, tile_m, rounding_mode="floor")
        expected_bounds = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                tiles.cumsum(0).to(torch.int32),
            )
        )
        expected_experts = torch.repeat_interleave(
            torch.arange(experts, device="cuda", dtype=torch.int32), tiles.long()
        )
        torch.testing.assert_close(bounds, expected_bounds, rtol=0, atol=0)
        torch.testing.assert_close(
            tile_experts[: expected_experts.numel()], expected_experts, rtol=0, atol=0
        )


def _canonical_grouped(weight, x: torch.Tensor, route: MoeRoutePlan) -> torch.Tensor:
    out = torch.empty(
        (route.tokens, route.routes, weight.out_per_expert),
        device=x.device,
        dtype=torch.float16,
    )
    input_rows = route.tokens * route.routes if x.ndim == 3 else route.tokens
    quantized: set[tuple[int, int]] = set()
    for pool in weight.pools:
        packed = pool.weight
        gs = int(packed["gs"])
        groups = int(packed["ng"])
        key = (gs, groups)
        qx, xscale = weight.activation_workspace(
            x, gs=gs, groups=groups, input_rows=input_rows
        )
        ext().mfe_nint_matmul_ws_cuda(
            packed["q_packed"],
            packed["row_q_bits"],
            packed["row_q_bit_offsets"],
            packed["sub_scale"],
            packed["sub_min"],
            packed["neuron_scale"],
            packed["neuron_min"],
            x,
            route.ids,
            pool.local_map(weight.n_experts, x.device),
            weight.n_experts,
            len(pool.expert_ids),
            weight.out_per_expert,
            gs,
            0,
            route.map_ready,
            key in quantized,
            out,
            qx,
            xscale,
            route.ids_dst,
            route.expert_bounds,
            route.tile_bounds,
            route.tile_experts,
            8,
            0,
        )
        quantized.add(key)
    return out


@pytest.mark.parametrize("tokens", [1, 4, 13])
def test_expertwise_nint_grouped_matmul_matches_reference(tokens):
    torch.manual_seed(10 + tokens)
    tensor, weight = _mixed_weight(tokens)
    x = torch.randn(tokens, tensor.neuron_len, device="cuda", dtype=torch.float16) * 0.1
    ids = _ids(tokens, tensor.n_experts, 3)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    actual = grouped_matmul(weight, x, route)
    expected = _reference(tensor, x, ids)
    relative = ((actual - expected).float().norm() / expected.float().norm()).item()
    assert relative < 0.025, f"tokens={tokens}, relative={relative}"


def test_expertwise_nint_grouped_down_input_matches_reference():
    torch.manual_seed(44)
    tensor, weight = _mixed_weight(44, out=7, k=61)
    tokens, routes = 17, 2
    x = torch.randn(tokens, routes, tensor.neuron_len, device="cuda", dtype=torch.float16) * 0.1
    ids = _ids(tokens, tensor.n_experts, routes)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    actual = grouped_matmul(weight, x, route)
    expected = _reference(tensor, x, ids)
    relative = ((actual - expected).float().norm() / expected.float().norm()).item()
    assert relative < 0.025, f"relative={relative}"


def test_expertwise_nint_large_routed_tensor_matches_reference():
    torch.manual_seed(145)
    tensor, weight = _mixed_weight(
        145,
        experts=11,
        out=257,
        k=1025,
    )
    tokens, routes = 33, 4
    x = torch.randn(
        tokens,
        routes,
        tensor.neuron_len,
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    ids = _ids(tokens, tensor.n_experts, routes)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    actual = grouped_matmul(weight, x, route)
    expected = _reference(tensor, x, ids)
    relative = ((actual - expected).float().norm() / expected.float().norm()).item()
    assert relative < 0.025, f"relative={relative}"


def test_expertwise_nint_wide_prefill_tail_tiles_match_reference():
    torch.manual_seed(146)
    tensor, weight = _mixed_weight(
        146,
        experts=17,
        out=17,
        k=29,
    )
    tokens, routes = 1024, 8
    x = torch.randn(
        tokens,
        routes,
        tensor.neuron_len,
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    ids = _ids(tokens, tensor.n_experts, routes)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    assert route.wide_tile_m == 128

    actual = grouped_matmul(weight, x, route)
    dense = torch.as_tensor(
        dequantize_expertwise(tensor), device="cuda", dtype=torch.float32
    )
    selected = dense.index_select(0, ids.long().reshape(-1)).reshape(
        tokens, routes, tensor.out_per_expert, tensor.neuron_len
    )
    expected = torch.einsum("trk,trok->tro", x.float(), selected).half()
    relative = ((actual - expected).float().norm() / expected.float().norm()).item()
    assert relative < 0.015, f"relative={relative}"


@pytest.mark.parametrize("tokens", [1, 2])
def test_expertwise_nint_uses_the_canonical_launch(tokens):
    torch.manual_seed(140 + tokens)
    tensor, weight = _mixed_weight(tokens, experts=8, out=17, k=97)
    ids = _ids(tokens, tensor.n_experts, 3)
    route = MoeRoutePlan.build(ids, tensor.n_experts)
    x = torch.randn(tokens, tensor.neuron_len, device="cuda", dtype=torch.float16) * 0.1
    actual = grouped_matmul(weight, x, route)
    canonical = _canonical_grouped(weight, x, route)
    torch.testing.assert_close(actual, canonical, rtol=0, atol=0)


def test_moe_topk_softmax_matches_torch():
    torch.manual_seed(71)
    logits = torch.randn(9, 64, device="cuda", dtype=torch.float32)
    ids, weights = topk(logits, 6)
    expected_weights, expected_ids = torch.softmax(logits, dim=-1).topk(6, dim=-1)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)


def test_moe_topk_m1_256_delayed_softmax_matches_torch():
    torch.manual_seed(71256)
    logits = torch.randn(1, 256, device="cuda", dtype=torch.float32)
    ids, weights = topk(logits, 8, delayed_softmax=True)
    expected_values, expected_ids = logits.topk(8, dim=-1)
    expected_weights = torch.softmax(expected_values, dim=-1)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)


def test_moe_topk_sigmoid_bias_and_normalization_matches_torch():
    torch.manual_seed(72)
    logits = torch.randn(7, 48, device="cuda", dtype=torch.float16)
    bias = torch.randn(48, device="cuda", dtype=torch.float32) * 0.05
    ids, weights = topk(logits, 4, use_sigmoid=True, normalize=True, bias=bias)
    transformed = torch.sigmoid(logits.float())
    expected_ids = (transformed + bias).topk(4, dim=-1).indices
    expected_weights = transformed.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("rows", "dtype"),
    [(1, torch.float32), (7, torch.float16)],
)
def test_moe_topk_sqrtsoftplus_matches_deepseek_v4(rows, dtype):
    torch.manual_seed(730 + rows)
    logits = torch.randn(rows, 256, device="cuda", dtype=dtype) * 3.0
    bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.05
    ids, weights = topk(
        logits,
        6,
        use_sqrt_softplus=True,
        normalize=True,
        bias=bias,
        scale=1.5,
    )
    transformed = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_ids = (transformed + bias).topk(6, dim=-1).indices
    expected_weights = transformed.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.5
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=4e-5, atol=4e-6)


def test_moe_topk_sqrtsoftplus_extreme_logits_are_finite():
    logits = torch.linspace(-100.0, 100.0, 256, device="cuda").reshape(1, -1)
    ids, weights = topk(
        logits,
        6,
        use_sqrt_softplus=True,
        normalize=True,
        scale=1.5,
    )
    transformed = torch.nn.functional.softplus(logits).sqrt()
    expected_ids = transformed.topk(6, dim=-1).indices
    expected_weights = transformed.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.5
    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_moe_hash_route_sqrtsoftplus_weights_match_torch(dtype):
    torch.manual_seed(740)
    logits = torch.randn(11, 256, device="cuda", dtype=dtype) * 2.0
    ids = torch.randint(0, 256, (11, 6), device="cuda", dtype=torch.int32)
    actual = sqrtsoftplus_weights(logits, ids, scale=1.5)
    transformed = torch.nn.functional.softplus(logits.float()).sqrt()
    expected = transformed.gather(1, ids.long())
    expected = expected / expected.sum(dim=-1, keepdim=True) * 1.5
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_moe_weighted_reduce_uses_fp32_accumulation():
    torch.manual_seed(73)
    pair = torch.randn(11, 8, 37, device="cuda", dtype=torch.float16)
    weights = torch.softmax(torch.randn(11, 8, device="cuda"), dim=-1)
    actual = weighted_reduce(pair, weights)
    expected = (pair.float() * weights[:, :, None]).sum(dim=1).half()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_moe_fused_swiglu_split_matches_torch():
    torch.manual_seed(74)
    gate_up = torch.randn(13, 8, 66, device="cuda", dtype=torch.float16)
    actual = swiglu_split(gate_up)
    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up).half()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_moe_fused_geglu_split_matches_torch():
    torch.manual_seed(740)
    gate_up = torch.randn(13, 8, 66, device="cuda", dtype=torch.float16)
    actual = geglu_split(gate_up)
    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.gelu(gate, approximate="tanh") * up).half()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_moe_fused_shared_gate_matches_torch():
    torch.manual_seed(75)
    routed = torch.randn(17, 73, device="cuda", dtype=torch.float16)
    shared = torch.randn_like(routed)
    gate = torch.randn(17, 1, device="cuda", dtype=torch.float32)
    actual = add_shared_gate(routed, shared, gate)
    expected = (routed.float() + torch.sigmoid(gate) * shared.float()).half()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_moe_reduce_shared_gate_is_bit_exact_to_two_kernels():
    torch.manual_seed(76)
    pair = torch.randn(11, 8, 37, device="cuda", dtype=torch.float16)
    weights = torch.softmax(torch.randn(11, 8, device="cuda"), dim=-1)
    shared = torch.randn(11, 37, device="cuda", dtype=torch.float16)
    gate = torch.randn(11, 1, device="cuda", dtype=torch.float32)
    actual = weighted_reduce_shared_gate(pair, weights, shared, gate)
    expected = add_shared_gate(weighted_reduce(pair, weights), shared, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
