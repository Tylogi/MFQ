"""CUDA MoE routing and expert-wise NINT ``mul_mat_id`` primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from mfq.kernels.cuda._ext import ext
from mfq.kernels.cuda.activation import silu_mul


@dataclass
class NintExpertPool:
    """One homogeneous execution cohort inside an expert-wise weight tensor.

    ``expert_ids[local]`` owns rows
    ``[local * out_per_expert:(local + 1) * out_per_expert]`` in ``weight``.
    """

    weight: dict
    expert_ids: tuple[int, ...]
    _local_maps: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def local_map(self, n_experts: int, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._local_maps.get(key)
        if cached is not None and cached.device == device:
            return cached
        mapping = torch.full((n_experts,), -1, dtype=torch.int32)
        if self.expert_ids:
            global_ids = torch.tensor(self.expert_ids, dtype=torch.int64)
            mapping[global_ids] = torch.arange(len(self.expert_ids), dtype=torch.int32)
        cached = mapping.to(device=device, non_blocking=True)
        self._local_maps[key] = cached
        return cached


@dataclass
class CompactExpertPool:
    """One native compact-format execution cohort."""

    family: str
    weight: dict
    expert_ids: tuple[int, ...]
    _local_maps: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def local_map(self, n_experts: int, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._local_maps.get(key)
        if cached is not None and cached.device == device:
            return cached
        mapping = torch.full((n_experts,), -1, dtype=torch.int32)
        if self.expert_ids:
            global_ids = torch.tensor(self.expert_ids, dtype=torch.int64)
            mapping[global_ids] = torch.arange(len(self.expert_ids), dtype=torch.int32)
        cached = mapping.to(device=device, non_blocking=True)
        self._local_maps[key] = cached
        return cached


@dataclass
class ExpertWiseNintWeight:
    """A logical ``[experts, out, in]`` tensor split into precision cohorts."""

    n_experts: int
    out_per_expert: int
    neuron_len: int
    pools: tuple[NintExpertPool, ...]
    _activation_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_experts <= 0 or self.out_per_expert <= 0 or self.neuron_len <= 0:
            raise ValueError("expert-wise NINT dimensions must be positive")
        if not self.pools:
            raise ValueError("expert-wise NINT weight must contain at least one pool")
        owners = [-1] * self.n_experts
        for pool_index, pool in enumerate(self.pools):
            if not pool.expert_ids:
                raise ValueError("expert-wise NINT pool cannot be empty")
            g = pool.weight
            expected_rows = len(pool.expert_ids) * self.out_per_expert
            if int(g["out"]) != expected_rows:
                raise ValueError(f"pool {pool_index} has {g['out']} rows; expected {expected_rows}")
            if int(g["neuron_len"]) != self.neuron_len:
                raise ValueError(f"pool {pool_index} input width mismatch")
            bits = int(g.get("bits", 4))
            if bits not in range(1, 9):
                raise ValueError(f"pool {pool_index} uses unsupported NINT{bits}")
            if g.get("row_q_bits") is None or g.get("row_q_bit_offsets") is None:
                raise ValueError(f"pool {pool_index} lacks canonical NINT row metadata")
            for expert in pool.expert_ids:
                if not 0 <= expert < self.n_experts:
                    raise ValueError(f"expert id {expert} is outside [0, {self.n_experts})")
                if owners[expert] != -1:
                    raise ValueError(f"expert {expert} appears in more than one precision pool")
                owners[expert] = pool_index
        missing = [expert for expert, owner in enumerate(owners) if owner < 0]
        if missing:
            raise ValueError(f"expert-wise NINT pools do not cover experts {missing[:16]}")

    @classmethod
    def homogeneous(
        cls,
        weight: dict,
        n_experts: int,
        out_per_expert: int | None = None,
    ) -> ExpertWiseNintWeight:
        if out_per_expert is None:
            if int(weight["out"]) % n_experts:
                raise ValueError("flattened expert rows are not divisible by n_experts")
            out_per_expert = int(weight["out"]) // n_experts
        return cls(
            n_experts=int(n_experts),
            out_per_expert=int(out_per_expert),
            neuron_len=int(weight["neuron_len"]),
            pools=(NintExpertPool(weight, tuple(range(int(n_experts)))),),
        )

    def activation_workspace(
        self,
        x: torch.Tensor,
        *,
        gs: int,
        groups: int,
        input_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_pad = groups * gs
        key = (str(x.device), input_rows, k_pad, groups)
        cached = self._activation_workspaces.get(key)
        if cached is not None and cached[0].device == x.device:
            return cached
        cached = (
            torch.empty((input_rows, k_pad), device=x.device, dtype=torch.int8),
            torch.empty((input_rows, groups), device=x.device, dtype=torch.float32),
        )
        self._activation_workspaces[key] = cached
        return cached


@dataclass
class ExpertWiseMixedWeight:
    """A logical expert tensor dispatched by native precision-family cohorts."""

    n_experts: int
    out_per_expert: int
    neuron_len: int
    pools: tuple[CompactExpertPool, ...]
    _activation_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_experts <= 0 or self.out_per_expert <= 0 or self.neuron_len <= 0:
            raise ValueError("expert-wise mixed dimensions must be positive")
        if not self.pools:
            raise ValueError("expert-wise mixed weight must contain at least one pool")
        owners = [-1] * self.n_experts
        for pool_index, pool in enumerate(self.pools):
            if pool.family not in {
                "nint",
                "nint8_zero",
                "nvq",
                "nepq",
                "mxfp4",
                "mxfp4_sq",
                "mxfp8_sq",
                "fp8_128_sq",
            }:
                raise ValueError(f"unsupported expert cohort family: {pool.family}")
            if not pool.expert_ids:
                raise ValueError("expert-wise mixed pool cannot be empty")
            g = pool.weight
            if int(g["neuron_len"]) != self.neuron_len:
                raise ValueError(f"pool {pool_index} input width mismatch")
            if pool.family == "nint" and (
                g.get("row_q_bits") is None
                or g.get("row_q_bit_offsets") is None
            ):
                raise ValueError(f"NINT pool {pool_index} lacks canonical row metadata")
            if pool.family == "nepq":
                if (
                    int(g["n_experts"]) != len(pool.expert_ids)
                    or int(g["out_per_expert"]) != self.out_per_expert
                ):
                    raise ValueError(f"NEPQ pool {pool_index} shape mismatch")
            else:
                expected_rows = len(pool.expert_ids) * self.out_per_expert
                if int(g["out"]) != expected_rows:
                    raise ValueError(
                        f"pool {pool_index} has {g['out']} rows; expected {expected_rows}"
                    )
            for expert in pool.expert_ids:
                if not 0 <= expert < self.n_experts:
                    raise ValueError(f"expert id {expert} is outside [0, {self.n_experts})")
                if owners[expert] != -1:
                    raise ValueError(f"expert {expert} appears in more than one precision pool")
                owners[expert] = pool_index
        missing = [expert for expert, owner in enumerate(owners) if owner < 0]
        if missing:
            raise ValueError(f"expert-wise mixed pools do not cover experts {missing[:16]}")

    def activation_workspace(
        self,
        x: torch.Tensor,
        *,
        gs: int,
        groups: int,
        input_rows: int,
        transform_key: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_pad = groups * gs
        key = (str(x.device), input_rows, k_pad, groups, transform_key)
        cached = self._activation_workspaces.get(key)
        if cached is not None and cached[0].device == x.device:
            return cached
        cached = (
            torch.empty((input_rows, k_pad), device=x.device, dtype=torch.int8),
            torch.empty((input_rows, groups), device=x.device, dtype=torch.float32),
        )
        self._activation_workspaces[key] = cached
        return cached


@dataclass(frozen=True)
class MoeRoutePlan:
    """GPU route metadata shared by gate, up, and down expert projections."""

    ids: torch.Tensor
    n_experts: int
    ids_dst: torch.Tensor
    expert_bounds: torch.Tensor
    tile_bounds: torch.Tensor
    tile_experts: torch.Tensor
    mma_tile_bounds: torch.Tensor
    mma_tile_experts: torch.Tensor
    mma_tile_m: int
    wide_tile_bounds: torch.Tensor
    wide_tile_experts: torch.Tensor
    wide_tile_m: int
    counts: torch.Tensor
    cursors: torch.Tensor

    @property
    def tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def routes(self) -> int:
        return int(self.ids.shape[1])

    @property
    def map_ready(self) -> bool:
        return self.tokens <= 8 or self.ids_dst.numel() == self.ids.numel()

    @classmethod
    def build(cls, ids: torch.Tensor, n_experts: int) -> MoeRoutePlan:
        if ids.ndim != 2:
            raise ValueError("ids must have [tokens, routes] shape")
        ids = ids.contiguous().to(device=ids.device, dtype=torch.int32)
        if int(ids.shape[0]) > 8:
            rows_per_expert = max(
                1, (ids.numel() + int(n_experts) - 1) // int(n_experts)
            )
            use_coarse_mma = rows_per_expert >= 4
            mma_tile_m = (
                16 if rows_per_expert <= 16 else 32 if rows_per_expert <= 32 else 64
            )
            if not use_coarse_mma:
                mapped = ext().moe_build_expert_map_cuda(ids, int(n_experts), 8)
            elif rows_per_expert > 64:
                mapped = ext().moe_build_expert_maps_cuda(
                    ids, int(n_experts), 8, mma_tile_m, 128
                )
            else:
                mapped = ext().moe_build_expert_maps_cuda(
                    ids, int(n_experts), 8, mma_tile_m
                )
            ids_dst, expert_bounds, tile_bounds, tile_experts, counts = mapped[:5]
            if use_coarse_mma:
                mma_tile_bounds, mma_tile_experts = mapped[5:7]
                if rows_per_expert > 64:
                    wide_tile_bounds, wide_tile_experts = mapped[7:9]
                    wide_tile_m = 128
                else:
                    wide_tile_bounds = mma_tile_bounds
                    wide_tile_experts = mma_tile_experts
                    wide_tile_m = mma_tile_m
            else:
                mma_tile_bounds = tile_bounds
                mma_tile_experts = tile_experts
                mma_tile_m = 8
                wide_tile_bounds = tile_bounds
                wide_tile_experts = tile_experts
                wide_tile_m = 8
            cursors = torch.empty(0, device=ids.device, dtype=torch.int32)
        else:
            empty = torch.empty(0, device=ids.device, dtype=torch.int32)
            ids_dst = empty
            expert_bounds = empty
            tile_bounds = empty
            tile_experts = empty
            mma_tile_bounds = empty
            mma_tile_experts = empty
            mma_tile_m = 8
            wide_tile_bounds = empty
            wide_tile_experts = empty
            wide_tile_m = 8
            counts = empty
            cursors = empty
        return cls(
            ids=ids,
            n_experts=int(n_experts),
            ids_dst=ids_dst,
            expert_bounds=expert_bounds,
            tile_bounds=tile_bounds,
            tile_experts=tile_experts,
            mma_tile_bounds=mma_tile_bounds,
            mma_tile_experts=mma_tile_experts,
            mma_tile_m=mma_tile_m,
            wide_tile_bounds=wide_tile_bounds,
            wide_tile_experts=wide_tile_experts,
            wide_tile_m=wide_tile_m,
            counts=counts,
            cursors=cursors,
        )


def _nint_prefill_route_tiles(
    route: MoeRoutePlan,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pairs = route.tokens * route.routes
    rows_per_expert = max(1, (pairs + route.n_experts - 1) // route.n_experts)
    if rows_per_expert < 4:
        return route.tile_bounds, route.tile_experts, 8
    if rows_per_expert > 64 and route.wide_tile_m == 128:
        return route.wide_tile_bounds, route.wide_tile_experts, 128
    if route.mma_tile_m in (16, 32, 64):
        return route.mma_tile_bounds, route.mma_tile_experts, route.mma_tile_m
    return route.tile_bounds, route.tile_experts, 8


def topk(
    logits: torch.Tensor,
    top_k: int,
    *,
    use_sigmoid: bool = False,
    use_sqrt_softplus: bool = False,
    normalize: bool = False,
    delayed_softmax: bool = False,
    bias: torch.Tensor | None = None,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the router transform and return contiguous int32 ids/f32 weights."""

    logits = logits.reshape(-1, logits.shape[-1]).contiguous()
    if bias is not None:
        bias = bias.contiguous().to(device=logits.device, dtype=torch.float32)
    ids, weights = ext().moe_topk_cuda(
        logits,
        int(top_k),
        bool(use_sigmoid),
        bool(use_sqrt_softplus),
        bool(normalize),
        bool(delayed_softmax),
        bias,
        float(norm_floor),
        float(scale),
    )
    return ids, weights


def sqrtsoftplus_weights(
    logits: torch.Tensor,
    ids: torch.Tensor,
    *,
    norm_floor: float = 1e-20,
    scale: float = 1.0,
) -> torch.Tensor:
    """Gather hash-selected router weights using DeepSeek V4's transform."""

    logits = logits.reshape(-1, logits.shape[-1]).contiguous()
    ids = ids.reshape(logits.shape[0], -1).contiguous().to(device=logits.device, dtype=torch.int32)
    return ext().moe_sqrtsoftplus_weights_cuda(
        logits,
        ids,
        float(norm_floor),
        float(scale),
    )


ExpertWiseWeight = ExpertWiseNintWeight | ExpertWiseMixedWeight


def _prepare_input(weight: ExpertWiseWeight, x: torch.Tensor) -> torch.Tensor:
    if x.ndim not in (2, 3):
        raise ValueError("expert input must have [T,K] or [T,R,K] shape")
    x = x.contiguous().to(torch.float16)
    if x.shape[-1] > weight.neuron_len:
        raise ValueError(f"input width {x.shape[-1]} exceeds {weight.neuron_len}")
    if x.shape[-1] < weight.neuron_len:
        x = torch.nn.functional.pad(x, (0, weight.neuron_len - x.shape[-1]))
    return x


def _grouped_matmul_mixed(
    weight: ExpertWiseMixedWeight,
    x: torch.Tensor,
    route: MoeRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from mfq.kernels.cuda.nepq_matmul import (
        _prepare_routed_input as prepare_nepq_routed_input,
    )
    from mfq.kernels.cuda.nepq_matmul import nepq_grouped_matmul_pool
    from mfq.kernels.cuda.nvq_matmul import nvq_grouped_matmul_pool

    if route.n_experts != weight.n_experts:
        raise ValueError("route and weight expert counts differ")
    x = _prepare_input(weight, x)
    if x.shape[0] != route.tokens:
        raise ValueError("input and route token counts differ")
    if x.ndim == 3 and x.shape[1] != route.routes:
        raise ValueError("routed input and route counts differ")
    if x.device != route.ids.device:
        raise ValueError("input and route tensors must be on the same device")
    expected = (route.tokens, route.routes, weight.out_per_expert)
    if out is None:
        out = torch.empty(expected, device=x.device, dtype=torch.float16)
    elif (
        tuple(out.shape) != expected
        or out.dtype != torch.float16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous float16 {expected} on {x.device}")

    input_rows = route.tokens * route.routes if x.ndim == 3 else route.tokens
    prepared_inputs: dict[tuple, torch.Tensor] = {("identity",): x}
    quantized: set[tuple] = set()
    nint_pool_phase = 0
    for pool in weight.pools:
        g = pool.weight
        expert_local = pool.local_map(weight.n_experts, x.device)
        if pool.family == "mxfp4_sq":
            ext().mxfp4_sq_moe_matmul_cuda(
                g["blob"],
                g["row_q"],
                g["row_symbol_byte_offsets"],
                g["row_auxiliary"],
                x,
                route.ids,
                expert_local,
                int(g["bits"]),
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                weight.neuron_len,
                int(g["matrix_scale_base"]),
                int(g["q_sum"]),
                int(g["sq4_rows"]),
                out,
            )
            continue
        if pool.family == "mxfp8_sq":
            ext().mxfp8_sq_moe_matmul_cuda(
                g["blob"],
                g["row_q"],
                g["row_symbol_byte_offsets"],
                x,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                weight.neuron_len,
                int(g["block_rows"]),
                int(g["block_columns"]),
                int(g["scale_rows"]),
                int(g["scale_columns"]),
                int(g["palettes_offset"]),
                int(g["symbols_offset"]),
                int(g["scales_offset"]),
                out,
            )
            continue
        if pool.family == "fp8_128_sq":
            ext().fp8_128_sq_moe_matmul_cuda(
                g["blob"],
                g["row_q"],
                g["row_symbol_byte_offsets"],
                x,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                weight.neuron_len,
                int(g["scale_kind"]),
                int(g["palettes_offset"]),
                int(g["symbols_offset"]),
                int(g["scales_offset"]),
                out,
            )
            continue
        if pool.family == "mxfp4":
            ext().mxfp4_moe_grouped_matmul_pool_f16_cuda(
                g["values"],
                g["scales"],
                x,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                weight.neuron_len,
                out,
                route.ids_dst,
                route.expert_bounds,
                route.tile_bounds,
                route.tile_experts,
            )
            continue
        if pool.family == "nepq" and int(g.get("rotation_block", 0)):
            transform_key = (
                "hadamard",
                int(g["rotation_block"]),
                int(g.get("rotation_seed", 0)),
            )
            value = prepared_inputs.get(transform_key)
            if value is None:
                value = prepare_nepq_routed_input(g, x)
                prepared_inputs[transform_key] = value
        else:
            transform_key = ("identity",)
            value = x
        gs = 32 if pool.family == "nint8_zero" else int(g["gs"])
        groups = int(g["ng"])
        activation_key = (transform_key, gs, groups)
        qx, xscale = weight.activation_workspace(
            value,
            gs=gs,
            groups=groups,
            input_rows=input_rows,
            transform_key=transform_key,
        )
        input_quantized = activation_key in quantized
        activation_quantized_after_call = True
        if pool.family == "nint":
            nint_tile_bounds, nint_tile_experts, nint_tile_m = (
                _nint_prefill_route_tiles(route)
            )
            ext().mfe_nint_matmul_ws_cuda(
                g["q_packed"],
                g["row_q_bits"],
                g["row_q_bit_offsets"],
                g["sub_scale"],
                g["sub_min"],
                g["neuron_scale"],
                g["neuron_min"],
                value,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                gs,
                0,
                route.map_ready,
                input_quantized,
                out,
                qx,
                xscale,
                route.ids_dst,
                route.expert_bounds,
                nint_tile_bounds,
                nint_tile_experts,
                nint_tile_m,
                nint_pool_phase,
            )
            nint_pool_phase += 1
            activation_quantized_after_call = input_quantized or nint_tile_m == 8
        elif pool.family == "nint8_zero":
            ext().nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                g["q"],
                g["scale"],
                value,
                route.ids,
                expert_local,
                weight.n_experts,
                len(pool.expert_ids),
                weight.out_per_expert,
                route.map_ready,
                input_quantized,
                False,
                out,
                qx,
                xscale,
                route.counts,
                route.cursors,
                route.ids_dst,
                route.expert_bounds,
                route.tile_bounds,
                route.tile_experts,
                8,
            )
        elif pool.family == "nvq":
            nvq_grouped_matmul_pool(
                g,
                value,
                route,
                expert_local,
                len(pool.expert_ids),
                out=out,
                qx=qx,
                xscale=xscale,
                input_quantized=input_quantized,
            )
        else:
            nepq_grouped_matmul_pool(
                g,
                value,
                route,
                expert_local,
                len(pool.expert_ids),
                out=out,
                qx=qx,
                xscale=xscale,
                input_quantized=input_quantized,
                input_prepared=True,
            )
        if activation_quantized_after_call:
            quantized.add(activation_key)
    return out


def grouped_matmul(
    weight: ExpertWiseWeight,
    x: torch.Tensor,
    route: MoeRoutePlan,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run expert-wise NINT ``mul_mat_id`` and return ``[T,R,out]`` pairs."""

    if isinstance(weight, ExpertWiseMixedWeight):
        return _grouped_matmul_mixed(weight, x, route, out=out)
    if route.n_experts != weight.n_experts:
        raise ValueError("route and weight expert counts differ")
    x = _prepare_input(weight, x)
    if x.shape[0] != route.tokens:
        raise ValueError("input and route token counts differ")
    if x.ndim == 3 and x.shape[1] != route.routes:
        raise ValueError("routed input and route counts differ")
    if x.device != route.ids.device:
        raise ValueError("input and route tensors must be on the same device")
    if out is None:
        out = torch.empty(
            (route.tokens, route.routes, weight.out_per_expert),
            device=x.device,
            dtype=torch.float16,
        )
    else:
        expected = (route.tokens, route.routes, weight.out_per_expert)
        if tuple(out.shape) != expected or out.dtype != torch.float16 or out.device != x.device:
            raise ValueError(f"out must be float16 {expected} on {x.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    input_rows = route.tokens * route.routes if x.ndim == 3 else route.tokens
    quantized_groups: set[tuple[int, int]] = set()
    nint_tile_bounds, nint_tile_experts, nint_tile_m = (
        _nint_prefill_route_tiles(route)
    )
    nint_pool_phase = 0
    for pool in weight.pools:
        g = pool.weight
        gs = int(g["gs"])
        groups = int(g["ng"])
        key = (gs, groups)
        qx, xscale = weight.activation_workspace(x, gs=gs, groups=groups, input_rows=input_rows)
        ext().mfe_nint_matmul_ws_cuda(
            g["q_packed"],
            g["row_q_bits"],
            g["row_q_bit_offsets"],
            g["sub_scale"],
            g["sub_min"],
            g["neuron_scale"],
            g["neuron_min"],
            x,
            route.ids,
            pool.local_map(weight.n_experts, x.device),
            weight.n_experts,
            len(pool.expert_ids),
            weight.out_per_expert,
            gs,
            0,
            route.map_ready,
            key in quantized_groups,
            out,
            qx,
            xscale,
            route.ids_dst,
            route.expert_bounds,
            nint_tile_bounds,
            nint_tile_experts,
            nint_tile_m,
            nint_pool_phase,
        )
        nint_pool_phase += 1
        if key in quantized_groups or nint_tile_m == 8:
            quantized_groups.add(key)
    return out


def weighted_reduce(pair_output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce ``[T,R,O]`` expert outputs with router weights in FP32."""

    return ext().moe_weighted_reduce_cuda(
        pair_output.contiguous().to(torch.float16),
        weights.contiguous().to(device=pair_output.device, dtype=torch.float32),
    )


def swiglu_split(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to contiguous ``[..., 2 * width]`` expert output."""

    return ext().moe_swiglu_split_cuda(gate_up.contiguous().to(torch.float16))


def geglu_split(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply tanh-approximate GeGLU to contiguous ``[..., 2 * width]`` output."""

    return ext().moe_geglu_split_cuda(gate_up.contiguous().to(torch.float16))


def add_shared_gate(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute ``routed + sigmoid(gate_logits) * shared`` in one CUDA kernel."""

    return ext().moe_add_shared_gate_cuda(
        routed.contiguous().to(torch.float16),
        shared.contiguous().to(device=routed.device, dtype=torch.float16),
        gate_logits.reshape(-1, 1).contiguous().to(device=routed.device, dtype=torch.float32),
    )


def weighted_reduce_shared_gate(
    pair_output: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    """Reduce routed outputs and add the gated shared expert in one kernel."""

    return ext().moe_weighted_reduce_shared_gate_cuda(
        pair_output.contiguous().to(torch.float16),
        weights.contiguous().to(device=pair_output.device, dtype=torch.float32),
        shared.contiguous().to(device=pair_output.device, dtype=torch.float16),
        gate_logits.reshape(-1, 1).contiguous().to(device=pair_output.device, dtype=torch.float32),
    )


def moe_ffn(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    gate: ExpertWiseWeight,
    up: ExpertWiseWeight,
    down: ExpertWiseWeight,
    *,
    top_k_count: int,
    use_sigmoid: bool = False,
    use_sqrt_softplus: bool = False,
    normalize: bool = False,
    delayed_softmax: bool = False,
    router_bias: torch.Tensor | None = None,
    router_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute a complete routed SwiGLU FFN with expert-wise precision."""

    if gate.n_experts != up.n_experts or gate.n_experts != down.n_experts:
        raise ValueError("MoE FFN expert counts differ")
    if gate.out_per_expert != up.out_per_expert:
        raise ValueError("MoE gate/up widths differ")
    if down.neuron_len != gate.out_per_expert:
        raise ValueError("MoE down input width does not match gate/up output width")
    ids, weights = topk(
        router_logits,
        top_k_count,
        use_sigmoid=use_sigmoid,
        use_sqrt_softplus=use_sqrt_softplus,
        normalize=normalize,
        delayed_softmax=delayed_softmax,
        bias=router_bias,
        scale=router_scale,
    )
    route = MoeRoutePlan.build(ids, gate.n_experts)
    gate_pair = grouped_matmul(gate, x, route)
    up_pair = grouped_matmul(up, x, route)
    hidden = silu_mul(gate_pair, up_pair)
    down_pair = grouped_matmul(down, hidden, route)
    return weighted_reduce(down_pair, weights), ids, weights


def pools_from_groups(groups: Iterable[tuple[dict, Iterable[int]]]) -> tuple[NintExpertPool, ...]:
    """Build pools while preserving each cohort's local expert row order."""

    return tuple(NintExpertPool(weight, tuple(int(v) for v in ids)) for weight, ids in groups)


def _payload_tensor(payload: bytes, device: str | torch.device) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(
        device=device,
        non_blocking=True,
    ).contiguous()


def _to_gpu_mxfp4_sq(tensor, device: str | torch.device) -> dict:
    from mfq.formats.mxfp4_sq import pack_mxfp4_sq, unpack_mxfp4_sq

    payload = pack_mxfp4_sq(tensor)
    parsed = unpack_mxfp4_sq(payload)
    row_q = tuple(int(value) for value in parsed.row_q_bits)
    symbol_offsets: list[int] = []
    auxiliary_rows: list[int] = []
    symbol_offset = 0
    sq_row = 0
    native_row = 0
    for q in row_q:
        symbol_offsets.append(symbol_offset)
        symbol_offset += int(parsed.input_size) * q // 8
        if q == 4:
            auxiliary_rows.append(native_row)
            native_row += 1
        else:
            auxiliary_rows.append(sq_row)
            sq_row += 1
    return {
        "blob": _payload_tensor(payload, device),
        "row_q": torch.tensor(row_q, dtype=torch.uint8, device=device),
        "row_symbol_byte_offsets": torch.tensor(
            symbol_offsets, dtype=torch.int32, device=device
        ),
        "row_auxiliary": torch.tensor(
            auxiliary_rows, dtype=torch.int32, device=device
        ),
        "bits": int(parsed.bits),
        "out": int(parsed.output_size),
        "neuron_len": int(parsed.input_size),
        "matrix_scale_base": int(parsed.matrix_scale_base),
        "q_sum": sum(row_q),
        "sq4_rows": native_row,
    }


def _to_gpu_fp8_sq(tensor, device: str | torch.device) -> dict:
    from mfq.formats.fp8_sq import parse_fp8_sq_layout

    layout = parse_fp8_sq_layout(tensor.dtype, tensor.payload)
    return {
        "blob": _payload_tensor(tensor.payload, device),
        "row_q": torch.tensor(layout.row_q_bits, dtype=torch.uint8, device=device),
        "row_symbol_byte_offsets": torch.tensor(
            layout.row_symbol_byte_offsets, dtype=torch.int32, device=device
        ),
        "out": int(layout.shape[0]),
        "neuron_len": int(layout.shape[1]),
        "block_rows": int(layout.block_shape[0]),
        "block_columns": int(layout.block_shape[1]),
        "scale_rows": int(layout.scale_shape[0]),
        "scale_columns": int(layout.scale_shape[1]),
        "scale_kind": {"F8_E8M0": 1, "BF16": 2, "F16": 3, "F32": 4}[
            layout.scale_dtype
        ],
        "palettes_offset": int(layout.palettes_offset),
        "symbols_offset": int(layout.symbols_offset),
        "scales_offset": int(layout.scales_offset),
    }


def to_gpu(
    tensor, device: str | torch.device = "cuda"
) -> ExpertWiseNintWeight | ExpertWiseMixedWeight:
    """Upload an :class:`MfeTensor` as native execution cohorts."""

    from mfq.formats.mfe import MfeTensor
    from mfq.formats.fp8_sq import Fp8_128SqTensor, Mxfp8SqTensor
    from mfq.formats.mx import MxTensor
    from mfq.formats.mxfp4_sq import Mxfp4SqTensor
    from mfq.formats.nepq import NepqTensor
    from mfq.formats.nint import NintTensor
    from mfq.formats.nint8_zero import Nint8ZeroTensor
    from mfq.kernels.cuda.mx_matmul import to_gpu_mx
    from mfq.kernels.cuda.nepq_matmul import to_gpu_nepq
    from mfq.kernels.cuda.nint8_zero_matmul import to_gpu_nint8_zero
    from mfq.kernels.cuda.nvq_matmul import to_gpu_nvq
    from mfq.kernels.torch_backend import to_gpu as nint_to_gpu

    if not isinstance(tensor, MfeTensor):
        raise TypeError("to_gpu expects MfeTensor")
    if all(isinstance(pool.tensor, NintTensor) for pool in tensor.pools):
        pools = tuple(
            NintExpertPool(
                nint_to_gpu(pool.tensor, device=device, layout="deploy"),
                tuple(int(value) for value in pool.expert_ids),
            )
            for pool in tensor.pools
        )
        return ExpertWiseNintWeight(
            n_experts=tensor.n_experts,
            out_per_expert=tensor.out_per_expert,
            neuron_len=tensor.neuron_len,
            pools=pools,
        )

    mixed_pools: list[CompactExpertPool] = []
    for pool in tensor.pools:
        expert_ids = tuple(int(value) for value in pool.expert_ids)
        if isinstance(pool.tensor, NintTensor):
            family = "nint"
            packed = nint_to_gpu(pool.tensor, device=device, layout="deploy")
        elif isinstance(pool.tensor, Nint8ZeroTensor):
            family = "nint8_zero"
            packed = to_gpu_nint8_zero(pool.tensor, device=device)
        elif isinstance(pool.tensor, NepqTensor):
            family = "nepq"
            packed = to_gpu_nepq(pool.tensor, device=device)
        elif isinstance(pool.tensor, MxTensor):
            if pool.tensor.dtype != "MXFP4":
                raise ValueError("MFE CUDA execution supports MXFP4 expert pools")
            family = "mxfp4"
            packed = to_gpu_mx(pool.tensor, device=device)
        elif isinstance(pool.tensor, Mxfp4SqTensor):
            family = "mxfp4_sq"
            packed = _to_gpu_mxfp4_sq(pool.tensor, device)
        elif isinstance(pool.tensor, Mxfp8SqTensor):
            family = "mxfp8_sq"
            packed = _to_gpu_fp8_sq(pool.tensor, device)
        elif isinstance(pool.tensor, Fp8_128SqTensor):
            family = "fp8_128_sq"
            packed = _to_gpu_fp8_sq(pool.tensor, device)
        else:
            family = "nvq"
            packed = to_gpu_nvq(pool.tensor, device=device)
        mixed_pools.append(
            CompactExpertPool(
                family=family,
                weight=packed,
                expert_ids=expert_ids,
            )
        )
    return ExpertWiseMixedWeight(
        n_experts=tensor.n_experts,
        out_per_expert=tensor.out_per_expert,
        neuron_len=tensor.neuron_len,
        pools=tuple(mixed_pools),
    )
