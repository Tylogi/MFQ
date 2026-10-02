"""Joint AlphaQ allocation for routed experts and dense matrices.

The caller supplies the full-file fixed cost, including PLE, native tensors
outside the allocation, runtime assets and the container. Dense native choices
remain explicit dtype overrides rather than invented quantized precisions.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from mfq.calibration.alphaq import (
    AlphaQTensorStatistics, _hull_allocation, alphaq_builtin_candidates,
    alphaq_candidates, alphaq_importance,
)
from mfq.calibration.artifact import (
    CalibrationScheme, ExpertPrecision, ExpertSelection, ExpertTensorSelection, TensorSelection,
)
from mfq.calibration.ew_solver import EwCandidate, EwCandidateTable, EwItemKey


@dataclass(frozen=True)
class DenseChoice:
    key: EwItemKey
    profile: str
    variable_storage_bits: int
    distortion: float
    precision: ExpertPrecision | None = None
    native_dtype: str | None = None
    pool_key: str = ""
    pool_storage_bits: int = 0

    def __post_init__(self):
        if (self.precision is None) == (self.native_dtype is None):
            raise ValueError("dense choice requires either precision or native dtype")
        if self.native_dtype is not None and (
            self.native_dtype not in {"BF16", "F16", "F32"} or self.distortion != 0
        ):
            raise ValueError("native dense choice must preserve source dtype with zero error")
        if self.pool_key or self.pool_storage_bits:
            raise ValueError("dense choices contain their complete payload, including tables")
        if self.variable_storage_bits <= 0 or self.variable_storage_bits % 8:
            raise ValueError("dense payload cost must be positive whole bytes")
        if not math.isfinite(self.distortion) or self.distortion < 0:
            raise ValueError("dense distortion must be finite and nonnegative")


@dataclass(frozen=True)
class JointAllocation:
    scheme: CalibrationScheme
    native_overrides: dict[str, str]
    selected: dict[EwItemKey, EwCandidate | DenseChoice]
    report: dict


def _imatrix_precision(precision: ExpertPrecision) -> ExpertPrecision:
    if precision.nint_spec is None:
        return precision
    options = {**dict(precision.options), "imatrix_weighted": True}
    return replace(precision, options=tuple(options.items()))


def dense_choices(
    statistics: Sequence[AlphaQTensorStatistics],
    source_dtypes: Mapping[str, str],
    profiles: Sequence[str] | None = None,
) -> tuple[DenseChoice, ...]:
    """Determine canonical dense byte costs without fitting any weights."""
    from mfq.formats.nint import NintSpec
    from mfq.tools.quantize_hf_to_mfq import TensorPlan, _plan_blob_nbytes

    if set(source_dtypes) != {s.name for s in statistics}:
        raise ValueError("source dtypes must cover the dense statistics exactly")
    choices = []
    for stat in statistics:
        if stat.shape[0] != 1:
            raise ValueError("dense statistics must describe one matrix")
        shape = stat.shape[1:]
        for candidate in alphaq_builtin_candidates([stat], profiles).candidates:
            precision = _imatrix_precision(candidate.precision)
            dtype = (f"NINT{precision.nint_spec.bits}" if precision.nint_spec
                     else precision.family)
            plan = TensorPlan(stat.name, "", shape, source_dtypes[stat.name], dtype,
                              target_spec=precision.nint_spec, target_precision=precision)
            bits = 8 * _plan_blob_nbytes(plan, NintSpec())
            choices.append(DenseChoice(candidate.key, candidate.profile, bits,
                                       candidate.distortion, precision))
        dtype = source_dtypes[stat.name]
        if dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unsupported native dense source dtype: {dtype}")
        bits = 8 * (20 + math.prod(shape) * (4 if dtype == "F32" else 2))
        choices.append(DenseChoice(EwItemKey(stat.name, stat.layer, stat.projection, 0),
                                   "NATIVE", bits, 0.0, native_dtype=dtype))
    return tuple(choices)


def allocate_joint(
    expert_statistics: Sequence[AlphaQTensorStatistics],
    expert_candidates: EwCandidateTable,
    dense_statistics: Sequence[AlphaQTensorStatistics],
    dense_candidates: Sequence[DenseChoice],
    *,
    model_weight_count: int,
    maximum_file_bytes: int,
    fixed_file_bytes: int,
    expert_exposure: Mapping[str, float],
    router_multipliers: Mapping[EwItemKey, float] | None = None,
) -> JointAllocation:
    """Allocate one full-file byte ceiling with analytical AlphaQ distortion.

    Expert exposure is top-k / expert count; dense exposure is one. All
    importance values share the expert AlphaQ scale. Storage, rather than a
    hand-set minimum precision, determines the cost of a small-matrix upgrade.
    Every affine NINT selection explicitly requires imatrix-weighted fitting,
    including NINT8; the original recipe-only NINT8 policy is unchanged.
    """
    if (not isinstance(model_weight_count, int) or model_weight_count <= 0
            or not isinstance(maximum_file_bytes, int) or maximum_file_bytes <= 0
            or not isinstance(fixed_file_bytes, int) or fixed_file_bytes < 0):
        raise ValueError("full-file budget and weight count must be valid integers")
    experts = {s.name: s for s in expert_statistics}
    dense = {s.name: s for s in dense_statistics}
    if (len(experts) != len(expert_statistics) or len(dense) != len(dense_statistics)
            or experts.keys() & dense.keys()):
        raise ValueError("expert and dense statistics require unique disjoint tensor names")
    if set(experts) != set(expert_candidates.tensors) or set(expert_exposure) != set(experts):
        raise ValueError("expert statistics, candidates and exposures must have identical scope")
    for name, spec in expert_candidates.tensors.items():
        if experts[name].shape != (spec.n_experts, spec.rows_per_expert, spec.columns):
            raise ValueError(f"expert shape mismatch: {name}")
        if not math.isfinite(expert_exposure[name]) or not 0 < expert_exposure[name] <= 1:
            raise ValueError("expert exposure must be finite and in (0,1]")
    if any(s.shape[0] != 1 for s in dense_statistics):
        raise ValueError("dense statistics must describe one matrix per tensor")
    allocated_weights = sum(math.prod(s.shape) for s in (*expert_statistics, *dense_statistics))
    if model_weight_count < allocated_weights:
        raise ValueError("model parameter count is smaller than the allocated tensors")

    scored = alphaq_candidates(replace(expert_candidates, candidates=tuple(
        replace(c, precision=_imatrix_precision(c.precision))
        for c in expert_candidates.candidates
    )))
    importance = alphaq_importance(expert_statistics)
    weights = importance.weights_for(scored.items)
    if router_multipliers is None:
        router_multipliers = dict.fromkeys(weights, 1.0)
    if set(router_multipliers) != set(weights) or any(
        not math.isfinite(v) or v < 0 for v in router_multipliers.values()
    ):
        raise ValueError("router multipliers must be finite, nonnegative and cover all experts")
    weights = {k: w * expert_exposure[k.tensor] * router_multipliers[k] for k, w in weights.items()}
    choices = defaultdict(list)
    for candidate in scored.candidates:
        choices[candidate.key].append(candidate)
    median = float(np.median(np.concatenate([s.alpha for s in expert_statistics])))
    scale = importance.metadata["importance_scale"]
    for stat in dense_statistics:
        key = EwItemKey(stat.name, stat.layer, stat.projection, 0)
        raw = median / stat.alpha[0] * stat.variance[0]
        weights[key] = raw / scale if scale > 0 else raw
    expected_dense = {EwItemKey(s.name, s.layer, s.projection, 0) for s in dense_statistics}
    if {c.key for c in dense_candidates} != expected_dense:
        raise ValueError("dense candidates must cover the exact dense statistic scope")
    for candidate in dense_candidates:
        if candidate.precision is not None:
            candidate = replace(candidate, precision=_imatrix_precision(candidate.precision))
        choices[candidate.key].append(candidate)
    if any(len({c.profile for c in options}) != len(options) for options in choices.values()):
        raise ValueError("candidate profiles must be unique per matrix")
    pools = {}
    for options in choices.values():
        for candidate in options:
            key = (candidate.key.tensor, candidate.pool_key)
            value = (candidate.pool_storage_bits, candidate.precision if candidate.pool_key else None)
            if key in pools and pools[key] != value:
                raise ValueError("shared pool cost or precision differs")
            pools[key] = value
    tensor_fixed = sum(t.fixed_storage_bits for t in scored.tensors.values())
    available = (maximum_file_bytes - fixed_file_bytes) * 8 - tensor_fixed
    reserved = sum(bits for bits, _ in pools.values())
    selected, lower = _hull_allocation(choices, weights, available - reserved)
    if reserved:
        _, lower = _hull_allocation(choices, weights, available)
    members = defaultdict(list)
    for key, candidate in selected.items():
        members[(key.tensor, candidate.pool_key)].append(key)
    charges = {min(keys): pools[pool][0] for pool, keys in members.items()}
    used_pools = sum(charges.values())
    storage = sum(c.variable_storage_bits for c in selected.values()) + used_pools + tensor_fixed
    if storage + fixed_file_bytes * 8 > maximum_file_bytes * 8:
        raise RuntimeError("joint allocation exceeds its full-file budget")
    expert_selections = {}
    for name, tensor in scored.tensors.items():
        selections = []
        for expert in range(tensor.n_experts):
            key = EwItemKey(name, tensor.layer, tensor.projection, expert)
            c = selected[key]
            loss = weights[key] * c.distortion
            selections.append(ExpertSelection(
                expert, c.precision.nint_spec, c.variable_storage_bits + charges.get(key, 0)
                + (tensor.fixed_storage_bits if expert == 0 else 0), loss, loss, c.precision,
            ))
        expert_selections[name] = ExpertTensorSelection(
            name, tensor.group, tensor.n_experts, tensor.rows_per_expert,
            tensor.columns, tuple(selections),
        )
    dense_selections, native, native_bits = {}, {}, 0
    for key in expected_dense:
        c = selected[key]
        if c.native_dtype is not None:
            native[key.tensor] = c.native_dtype
            native_bits += c.variable_storage_bits
        else:
            stat = dense[key.tensor]
            loss = weights[key] * c.distortion
            dense_selections[key.tensor] = TensorSelection(
                key.tensor, "dense", c.precision.nint_spec, *stat.shape[1:],
                c.variable_storage_bits, loss, loss, c.precision,
            )
    objective = sum(weights[k] * c.distortion for k, c in selected.items())
    metadata = dict(
        method="AlphaQ", distortion="analytical", dense_mode="joint",
        formula="AlphaQ importance * exposure * analytical per-weight distortion",
        integer_optimality_proven=False, solver="lower-hull relaxation + feasible integer rounding",
        native_distortion=0, quality_status="not evaluated; surrogate only",
    )
    scheme = CalibrationScheme(None, "AlphaQ-full-file", (maximum_file_bytes-fixed_file_bytes)*8,
                               dense_selections, metadata, {}, expert_selections=expert_selections)
    if scheme.storage_bits + native_bits != storage:
        raise RuntimeError("scheme and native payload costs do not match the solver")
    report = dict(
        **metadata, maximum_file_bytes=maximum_file_bytes, fixed_file_bytes=fixed_file_bytes,
        model_weight_count=model_weight_count, model_storage_bits=storage+fixed_file_bytes*8,
        model_bpw=(storage+fixed_file_bytes*8)/model_weight_count,
        native_dense_storage_bits=native_bits, objective=objective,
        relaxation_lower_bound=lower, relative_gap=max(0.0, objective-lower)/objective if objective else 0.0,
        reserved_pool_bits=reserved, used_pool_bits=used_pools,
        expert_selected_counts=dict(Counter(c.profile for k,c in selected.items() if k.tensor in experts)),
        dense_selected_counts=dict(Counter(c.profile for k,c in selected.items() if k.tensor in dense)),
    )
    return JointAllocation(scheme, native, selected, report)
