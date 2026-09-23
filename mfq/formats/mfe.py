"""MFE -- Mixed Format Experts container definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from mfq.formats.mx import MxTensor
from mfq.formats.compat import (
    FP8_128SQ_DTYPE,
    MXFP4_SQ_DTYPE,
    MXFP8_SQ_DTYPE,
    NINT_DTYPE,
)
from mfq.formats.fp8_sq import Fp8_128SqTensor, Mxfp8SqTensor
from mfq.formats.nepq import NepqTensor
from mfq.formats.mxfp4_sq import Mxfp4SqTensor, concatenate_mxfp4_sq_rows
from mfq.formats.nint import (
    NINT_K_SELECTOR_BITS,
    NintTensor,
    metadata_envelope_spec,
)
from mfq.formats.nint8_zero import Nint8ZeroTensor
from mfq.formats.npq0_l import Npq0LTensor
from mfq.formats.npq0_s import Npq0STensor
from mfq.formats.nvq import NvqJscTensor, NvqTensor
from mfq.formats.nvq1_l import Nvq1LTensor
from mfq.formats.nvq1_s import Nvq1STensor

ExpertPoolTensor: TypeAlias = (
    NintTensor
    | Nint8ZeroTensor
    | NvqTensor
    | NvqJscTensor
    | Npq0LTensor
    | Npq0STensor
    | Nvq1LTensor
    | Nvq1STensor
    | NepqTensor
    | MxTensor
    | Mxfp4SqTensor
    | Mxfp8SqTensor
    | Fp8_128SqTensor
    | np.ndarray
)


def expert_tensor_family(tensor: ExpertPoolTensor) -> str:
    """Return the public precision-family label for one cohort tensor."""

    if isinstance(tensor, Nint8ZeroTensor):
        return "NINT8-0"
    if isinstance(tensor, MxTensor):
        if tensor.dtype not in {"MXFP4", "MXFP8"}:
            raise ValueError(f"MFE supports native MXFP4/MXFP8 expert pools, got {tensor.dtype}")
        return tensor.dtype
    if isinstance(tensor, Mxfp4SqTensor):
        return MXFP4_SQ_DTYPE
    if isinstance(tensor, Mxfp8SqTensor):
        return MXFP8_SQ_DTYPE
    if isinstance(tensor, Fp8_128SqTensor):
        return FP8_128SQ_DTYPE
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.dtype(np.float16):
            return "F16"
        # BF16 is represented by io.BFloat16Array, a tagged uint16 ndarray.
        # Avoid importing io here because io owns the MFE codec and imports
        # this module.
        if tensor.dtype == np.dtype("<u2") and type(tensor).__name__ == "BFloat16Array":
            return "BF16"
        raise ValueError(f"MFE dense expert pools support BF16/F16, got {tensor.dtype}")
    if isinstance(tensor, NintTensor):
        return NINT_DTYPE
    if isinstance(tensor, NepqTensor):
        return tensor.spec.label
    if isinstance(tensor, Nvq1LTensor):
        return "NVQ1-L"
    if isinstance(tensor, Nvq1STensor):
        return "NVQ1-S"
    if isinstance(tensor, Npq0LTensor):
        return "NPQ0-L"
    if isinstance(tensor, Npq0STensor):
        return "NPQ0-S"
    if isinstance(tensor, NvqJscTensor):
        return {
            "e8_256": "NVQ2J",
            "e8_1024": "NVQ2J-L",
            "e8_4096": "NVQ2J-XL",
            "d4_256": "NVQ3J",
            "d4_512": "NVQ3J-512",
            "d4_1024": "NVQ3J-L",
        }[tensor.spec.codebook]
    if isinstance(tensor, NvqTensor):
        family = {
            "e8_256": "NVQ2",
            "d4_256": "NVQ3",
        }.get(tensor.spec.codebook)
        if family is None:
            raise ValueError(f"{tensor.spec.codebook} requires an NvqJscTensor expert profile")
        return family
    raise TypeError(f"unsupported MFE cohort tensor: {type(tensor)!r}")


@dataclass(frozen=True)
class MfePool:
    """One homogeneous precision cohort and its global expert IDs."""

    expert_ids: np.ndarray
    tensor: ExpertPoolTensor


def merge_nint_pools(
    pools: tuple[MfePool, ...],
    out_per_expert: int,
    neuron_len: int,
) -> tuple[MfePool, ...]:
    """Coalesce compatible NINT pools without changing their quantized values.

    NINTv2 stores q and k per neuron, so q/k preset differences do not require
    separate runtime cohorts.  The group size remains a tensor-level control;
    pools with different group sizes therefore stay separate.
    """

    by_groupsize: dict[int, list[tuple[int, MfePool]]] = {}
    for index, pool in enumerate(pools):
        if isinstance(pool.tensor, NintTensor):
            by_groupsize.setdefault(pool.tensor.spec.groupsize, []).append(
                (index, pool)
            )

    replacements: dict[int, MfePool] = {}
    consumed: set[int] = set()
    for group in by_groupsize.values():
        remaining = sorted(
            group,
            key=lambda item: (
                int(np.asarray(item[1].tensor.row_sub_bits).min()),
                item[0],
            ),
        )
        compatible_groups: list[list[tuple[int, MfePool]]] = []
        while remaining:
            seed = remaining.pop(0)
            minimum_k = int(np.asarray(seed[1].tensor.row_sub_bits).min())
            maximum_k = minimum_k + (1 << NINT_K_SELECTOR_BITS) - 1
            compatible = [seed]
            deferred: list[tuple[int, MfePool]] = []
            for entry in remaining:
                if int(np.asarray(entry[1].tensor.row_sub_bits).max()) <= maximum_k:
                    compatible.append(entry)
                else:
                    deferred.append(entry)
            compatible_groups.append(compatible)
            remaining = deferred

        for compatible in compatible_groups:
            if len(compatible) < 2:
                continue
            tensors = [pool.tensor for _, pool in compatible]

            expert_ids = np.concatenate(
                [
                    np.asarray(pool.expert_ids, dtype=np.int32).reshape(-1)
                    for _, pool in compatible
                ]
            )
            expert_order = np.argsort(expert_ids, kind="stable")
            row_order = (
                expert_order[:, None] * int(out_per_expert)
                + np.arange(int(out_per_expert), dtype=np.int64)[None, :]
            ).reshape(-1)

            def merged_rows(name: str) -> np.ndarray:
                values = np.concatenate(
                    [np.asarray(getattr(tensor, name)) for tensor in tensors], axis=0
                )
                return np.ascontiguousarray(values[row_order])

            row_q_bits = merged_rows("row_q_bits").astype(np.uint8, copy=False)
            row_sub_bits = merged_rows("row_sub_bits").astype(np.uint8, copy=False)
            merged = NintTensor(
                spec=metadata_envelope_spec(
                    tensors[0].spec.groupsize,
                    row_q_bits,
                    row_sub_bits,
                ),
                shape=(expert_ids.size * int(out_per_expert), int(neuron_len)),
                axis=0,
                q=merged_rows("q"),
                neuron_scale=merged_rows("neuron_scale"),
                neuron_min=merged_rows("neuron_min"),
                sub_scale=merged_rows("sub_scale"),
                sub_min=merged_rows("sub_min"),
                neuron_len=int(neuron_len),
                row_sub_bits=row_sub_bits,
                row_q_bits=row_q_bits,
            )
            first = min(index for index, _ in compatible)
            replacements[first] = MfePool(
                expert_ids=np.ascontiguousarray(expert_ids[expert_order]),
                tensor=merged,
            )
            consumed.update(
                index for index, _ in compatible if index != first
            )

    if not replacements:
        return pools
    return tuple(
        replacements.get(index, pool)
        for index, pool in enumerate(pools)
        if index not in consumed
    )


def merge_mxfp4_sq_pools(
    pools: tuple[MfePool, ...],
    out_per_expert: int,
    neuron_len: int,
) -> tuple[MfePool, ...]:
    """Coalesce MXFP4-SQ cohorts into per-neuron q streams.

    q=1/2/3/4 is row metadata in the unified format.  Only the matrix E8M0
    scale base remains tensor-level, so pools with distinct bases stay
    separate.  Packed values and row order are copied without requantization.
    """

    by_base: dict[int, list[tuple[int, MfePool]]] = {}
    for index, pool in enumerate(pools):
        if isinstance(pool.tensor, Mxfp4SqTensor):
            if pool.tensor.input_size != int(neuron_len):
                raise ValueError("MXFP4-SQ MFE pool width does not match its container")
            by_base.setdefault(pool.tensor.matrix_scale_base, []).append((index, pool))

    replacements: dict[int, MfePool] = {}
    consumed: set[int] = set()
    for compatible in by_base.values():
        if len(compatible) < 2:
            continue
        tensors = [pool.tensor for _, pool in compatible]
        expert_ids = np.concatenate(
            [
                np.asarray(pool.expert_ids, dtype=np.int32).reshape(-1)
                for _, pool in compatible
            ]
        )
        expert_order = np.argsort(expert_ids, kind="stable")
        row_order = (
            expert_order[:, None] * int(out_per_expert)
            + np.arange(int(out_per_expert), dtype=np.int64)[None, :]
        ).reshape(-1)
        first = min(index for index, _ in compatible)
        replacements[first] = MfePool(
            expert_ids=np.ascontiguousarray(expert_ids[expert_order]),
            tensor=concatenate_mxfp4_sq_rows(tensors, row_order),
        )
        consumed.update(index for index, _ in compatible if index != first)

    if not replacements:
        return pools
    return tuple(
        replacements.get(index, pool)
        for index, pool in enumerate(pools)
        if index not in consumed
    )


@dataclass(frozen=True)
class MfeTensor:
    """One logical ``[experts, out, in]`` tensor with per-expert precision."""

    shape: tuple[int, int, int]
    pools: tuple[MfePool, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("MFE shape must be [experts, out, in]")
        n_experts, out_per_expert, neuron_len = (int(value) for value in self.shape)
        if not self.pools:
            raise ValueError("MFE must contain at least one precision pool")
        owners = np.full(n_experts, -1, dtype=np.int32)
        for pool_index, pool in enumerate(self.pools):
            expert_ids = np.ascontiguousarray(pool.expert_ids, dtype=np.int32).reshape(-1)
            if expert_ids.size == 0:
                raise ValueError("MFE precision pools cannot be empty")
            if np.any(expert_ids < 0) or np.any(expert_ids >= n_experts):
                raise ValueError(f"MFE pool {pool_index} contains an invalid expert id")
            if np.unique(expert_ids).size != expert_ids.size:
                raise ValueError(f"MFE pool {pool_index} repeats an expert id")
            if np.any(owners[expert_ids] >= 0):
                raise ValueError("an expert belongs to multiple MFE pools")
            owners[expert_ids] = pool_index

            tensor = pool.tensor
            expert_tensor_family(tensor)
            if isinstance(tensor, NepqTensor):
                expected_shape = (expert_ids.size, out_per_expert, neuron_len)
                valid = tuple(tensor.shape) == expected_shape
            elif isinstance(
                tensor,
                (MxTensor, Mxfp4SqTensor, Mxfp8SqTensor, Fp8_128SqTensor, np.ndarray),
            ):
                expected_shape = (expert_ids.size * out_per_expert, neuron_len)
                valid = tuple(tensor.shape) == expected_shape
            else:
                expected_shape = (expert_ids.size * out_per_expert, neuron_len)
                valid = tuple(tensor.shape) == expected_shape and tensor.axis == 0
            if not valid:
                raise ValueError(
                    f"MFE pool {pool_index} tensor shape {tensor.shape} must be {expected_shape}"
                )
        missing = np.flatnonzero(owners < 0)
        if missing.size:
            raise ValueError(f"MFE pools do not cover experts {missing[:16].tolist()}")

    @property
    def n_experts(self) -> int:
        return int(self.shape[0])

    @property
    def out_per_expert(self) -> int:
        return int(self.shape[1])

    @property
    def neuron_len(self) -> int:
        return int(self.shape[2])

    @property
    def expert_profiles(self) -> tuple[str, ...]:
        result = [""] * self.n_experts
        for pool in self.pools:
            profile = expert_tensor_family(pool.tensor)
            for expert in np.asarray(pool.expert_ids).reshape(-1):
                result[int(expert)] = profile
        return tuple(result)
