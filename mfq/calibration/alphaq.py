"""Calibration-free AlphaQ importance and MFQ expert allocation.

Implements the official code-default FARMS / top-10%-Hill / gamma=1
objective. All complete 128x128 blocks are used deterministically (no
random block cap). No model execution, candidate fitting or text is needed.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertSelection,
    ExpertTensorSelection,
    nint_expert_precision,
)
from mfq.calibration.ew_solver import (
    EwBudget,
    EwCandidate,
    EwCandidateTable,
    EwItemKey,
    EwSolveResult,
    EwTensorSpec,
    ImportanceEntry,
    ImportanceTable,
    solve_ew_budget,
)

STATISTICS_FORMAT = "mfq.alphaq-statistics.v1"
METHOD = {
    "method": "AlphaQ",
    "formula": "median(alpha) / alpha * variance * 2**(-2*b)",
    "gamma": 1,
    "block_size": 128,
    "tail_fraction": 0.1,
    "block_policy": "all complete blocks; full matrix when either dimension < 128",
    "calibration_tokens": 0,
    "reference": "https://github.com/Superone77/AlphaQ",
}


@dataclass(frozen=True)
class AlphaQTensorStatistics:
    name: str
    layer: int
    projection: str
    shape: tuple[int, int, int]
    alpha: tuple[float, ...]
    variance: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("AlphaQ requires positive [experts, rows, columns] shapes")
        if not self.name or not self.projection or self.layer < 0:
            raise ValueError("invalid AlphaQ tensor identity")
        if len(self.alpha) != self.shape[0] or len(self.variance) != self.shape[0]:
            raise ValueError(f"AlphaQ statistics count differs from expert count: {self.name}")
        if any(not math.isfinite(a) or a < 1 for a in self.alpha):
            raise ValueError("AlphaQ alpha values must be finite and >= 1")
        if any(not math.isfinite(v) or v < 0 for v in self.variance):
            raise ValueError("AlphaQ variances must be finite and nonnegative")


def alphaq_weight_statistics(weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """Batched FARMS/Hill statistics for a floating [E,O,I] torch tensor.

    Uses batched FP32 singular values; no TF32 Gram products, approximate
    randomized SVD or CPU decoding. Batch size is reduced only after OOM.
    Tiny/rank-deficient/constant matrices have a finite deterministic result.
    """
    import torch

    if not isinstance(weights, torch.Tensor) or weights.ndim != 3:
        raise ValueError("AlphaQ weights must be a floating [E,O,I] torch tensor")
    if not weights.is_floating_point() or min(weights.shape) <= 0:
        raise ValueError("AlphaQ weights must be nonempty and floating point")
    with torch.no_grad():
        w = weights.detach().float()
        if not bool(torch.isfinite(w).all()):
            raise ValueError("AlphaQ weights contain NaN or infinity")
        experts, rows, columns = w.shape
        variance = w.var(dim=(-2, -1), correction=0).double().cpu().numpy()
        if min(rows, columns) >= 128:
            nr, nc = rows // 128, columns // 128
            blocks = (
                w[:, : nr * 128, : nc * 128]
                .reshape(experts, nr, 128, nc, 128)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1, 128, 128)
            )
        else:
            blocks = w
        # Keep the relatively small spectra on the device. Failed batches
        # do not discard spectra from preceding batches.
        spectra = torch.empty(
            (blocks.shape[0], min(blocks.shape[-2:])), dtype=torch.float32, device=w.device
        )
        start, batch = 0, blocks.shape[0]
        while start < blocks.shape[0]:
            count = min(batch, blocks.shape[0] - start)
            try:
                spectra[start : start + count] = torch.linalg.svdvals(
                    blocks[start : start + count]
                ).square()
            except torch.OutOfMemoryError:
                if count == 1:
                    raise
                batch = max(1, count // 2)
                continue
            start += count
        eigen = spectra.reshape(experts, -1).double().cpu().numpy()
    eigen.sort(axis=1)
    n = eigen.shape[1]
    if n < 2:
        return np.ones(experts, dtype=np.float64), variance
    k = min(n - 1, max(10, int(n * 0.1)))
    threshold = np.maximum(eigen[:, -k - 1], 1e-12)
    tail = np.maximum(eigen[:, -k:], 1e-12)
    denominator = np.maximum(np.log(tail / threshold[:, None]).sum(axis=1), 1e-12)
    return 1.0 + k / denominator, variance


def alphaq_importance(statistics: Sequence[AlphaQTensorStatistics]) -> ImportanceTable:
    """Return globally scaled weights; relative layer/projection scales survive."""
    if not statistics or len({s.name for s in statistics}) != len(statistics):
        raise ValueError("AlphaQ requires nonempty, unique tensor statistics")
    alpha = np.concatenate([np.asarray(s.alpha) for s in statistics])
    variance = np.concatenate([np.asarray(s.variance) for s in statistics])
    raw = np.median(alpha) / alpha * variance
    scale = float(raw.max())
    # An all-zero-variance model is a well-defined zero objective. It receives
    # the minimum-cost feasible assignment, not artificial importance.
    scores = raw / scale if scale > 0 else raw
    entries, offset = [], 0
    for tensor in statistics:
        for expert in range(tensor.shape[0]):
            entries.append(
                ImportanceEntry(
                    tensor.layer,
                    expert,
                    tensor.projection,
                    tensor.name,
                    float(scores[offset]),
                    None,
                )
            )
            offset += 1
    return ImportanceTable(
        "score", tuple(entries), "none", "linear_percentile", {**METHOD, "importance_scale": scale}
    )


def alphaq_candidates(candidates: EwCandidateTable) -> EwCandidateTable:
    """Replace empirical distortions with the AlphaQ analytical bit surrogate.

    NINT uses nominal q bits; vector/native formats use supplied effective
    payload BPW. Storage costs (including fixed pools) are kept unchanged.
    """
    scored = []
    for candidate in candidates.candidates:
        spec = candidate.precision.nint_spec
        bits = float(spec.bits) if spec is not None else candidate.effective_bpw
        if not math.isfinite(bits) or bits <= 0:
            raise ValueError(f"invalid AlphaQ bit width: {candidate.key}/{candidate.profile}")
        loss = math.exp2(-2 * bits) if hasattr(math, "exp2") else 2.0 ** (-2 * bits)
        scored.append(replace(candidate, distortion=loss, validation_distortion=loss))
    return replace(candidates, candidates=tuple(scored), metadata={**candidates.metadata, **METHOD})


def alphaq_nint_candidates(
    statistics: Sequence[AlphaQTensorStatistics], profiles: Sequence[str] | None = None
) -> EwCandidateTable:
    """Build standard NINT choices without quantizing any weights.

    Budget counts byte-rounded per-expert payload, neuron q/k selectors and
    the expert ID. Fixed MFE/container headers are excluded and labelled.
    Cohort packing can save a few bytes compared with per-expert rounding.
    """
    from mfq.calibration.evaluator import NINT_EXPERT_PROFILES
    from mfq.formats.nint import NINT_K_SELECTOR_BITS, NINT_Q_SELECTOR_BITS

    names = tuple(NINT_EXPERT_PROFILES) if profiles is None else tuple(profiles)
    if not names or len(set(names)) != len(names) or set(names) - NINT_EXPERT_PROFILES.keys():
        raise ValueError("profiles must be distinct standard NINT expert profiles")
    tensors, candidates = {}, []
    for s in statistics:
        experts, rows, columns = s.shape
        tensors[s.name] = EwTensorSpec(
            s.name, s.name, s.layer, s.projection, experts, rows, columns, 0, None
        )
        for profile in names:
            spec = NINT_EXPERT_PROFILES[profile]
            groups = (columns + spec.groupsize - 1) // spec.groupsize
            payload_bytes = (
                4 * rows
                + (rows * NINT_K_SELECTOR_BITS + 7) // 8
                + (rows * NINT_Q_SELECTOR_BITS + 7) // 8
                + 2 * ((rows * groups * spec.sub_bits + 7) // 8)
                + (rows * groups * spec.groupsize * spec.bits + 7) // 8
            )
            for expert in range(experts):
                key = EwItemKey(s.name, s.layer, s.projection, expert)
                candidates.append(
                    EwCandidate(
                        key,
                        profile,
                        nint_expert_precision(spec),
                        (payload_bytes + 4) * 8,
                        profile,
                        0,
                        0.0,
                        0.0,
                        payload_bytes * 8 / (rows * columns),
                    )
                )
    table = EwCandidateTable(
        tensors,
        tuple(candidates),
        None,
        {
            "storage_accounting": "per-expert byte-rounded NINTv2 payload + expert ID; excludes fixed container headers",
        },
    )
    return alphaq_candidates(table)


def _hull_allocation(
    choices: Mapping[EwItemKey, Sequence[EwCandidate]],
    weights: Mapping[EwItemKey, float],
    budget: int,
):
    """Single-budget convex-hull relaxation + feasible integer rounding.

    O(N*K*log(K) + N*K*log(N*K)); no model-sized MILP. The fractional
    relaxation is a lower bound, not a claim of integer optimality.
    """
    selected, events = {}, []
    actual, initial = 0, 0.0
    for key, candidates in choices.items():
        ordered = sorted(
            candidates,
            key=lambda c: (c.variable_storage_bits, weights[key] * c.distortion, c.profile),
        )
        hull: list[EwCandidate] = []
        for c in ordered:
            cost, loss = c.variable_storage_bits, weights[key] * c.distortion
            if hull and (
                cost == hull[-1].variable_storage_bits or loss >= weights[key] * hull[-1].distortion
            ):
                continue
            while len(hull) >= 2:
                a, b = hull[-2:]
                left = (
                    weights[key]
                    * (b.distortion - a.distortion)
                    / (b.variable_storage_bits - a.variable_storage_bits)
                )
                right = (
                    weights[key]
                    * (c.distortion - b.distortion)
                    / (c.variable_storage_bits - b.variable_storage_bits)
                )
                if right > left:
                    break
                hull.pop()
            hull.append(c)
        selected[key] = hull[0]
        actual += hull[0].variable_storage_bits
        initial += weights[key] * hull[0].distortion
        for a, b in zip(hull, hull[1:], strict=False):
            extra = b.variable_storage_bits - a.variable_storage_bits
            benefit = weights[key] * (a.distortion - b.distortion)
            events.append((-benefit / extra, len(events), key, a, b, extra, benefit))
    if actual > budget:
        raise ValueError(f"AlphaQ minimum payload {actual} bits exceeds budget {budget}")
    events.sort(key=lambda event: (event[0], event[1]))
    lower, fractional = initial, False
    for _, _, key, a, b, extra, benefit in events:
        if not fractional:
            if actual + extra > budget:
                lower -= (budget - actual) / extra * benefit
                fractional = True
            else:
                lower -= benefit
        if selected[key] == a and actual + extra <= budget:
            selected[key] = b
            actual += extra
    return selected, max(0.0, lower)


def allocate_alphaq(
    statistics: Sequence[AlphaQTensorStatistics],
    candidates: EwCandidateTable,
    budget: EwBudget,
    *,
    solver: str = "hull",
) -> EwSolveResult:
    """Allocate AlphaQ into the standard quantizer-consumable scheme.

    ``hull`` is the fast global upper-budget path. Shared nonzero pool
    charges, lower/shape/projection/layer constraints require explicit
    ``solver='exact'`` and reuse MFQ's existing joint EW solver.
    """
    importance = alphaq_importance(statistics)
    scored = alphaq_candidates(candidates)
    by_name = {s.name: s for s in statistics}
    for name, tensor in scored.tensors.items():
        if name not in by_name or by_name[name].shape != (
            tensor.n_experts,
            tensor.rows_per_expert,
            tensor.columns,
        ):
            raise ValueError(f"AlphaQ source/candidate shape or identity mismatch: {name}")
    if set(by_name) != set(scored.tensors):
        raise ValueError("AlphaQ statistics and candidates must cover the same tensors")
    weights = importance.weights_for(scored.items)
    if budget.model_weight_count < scored.routed_weight_count:
        raise ValueError("model weight count is smaller than routed weight count")
    if solver == "exact":
        result = solve_ew_budget(importance, scored, budget)
        metadata = {
            **result.scheme.metadata,
            **importance.metadata,
            "quality_status": "not evaluated; surrogate only",
        }
        return replace(
            result,
            scheme=replace(result.scheme, metadata=metadata),
            report={**result.report, **metadata},
        )
    if solver != "hull":
        raise ValueError("AlphaQ solver must be hull or exact")
    minimum, maximum = budget.total.resolve(budget.model_weight_count, "total")
    if (
        minimum not in (None, 0)
        or maximum is None
        or budget.projections
        or budget.layers
        or budget.shape_constraints
        or any(c.pool_storage_bits for c in scored.candidates)
    ):
        raise ValueError(
            "AlphaQ hull requires one upper budget and zero pool charges; use --solver exact for joint constraints"
        )
    fixed = sum(t.fixed_storage_bits for t in scored.tensors.values())
    available = maximum - budget.model_fixed_storage_bits - fixed
    choices: dict[EwItemKey, list[EwCandidate]] = defaultdict(list)
    for c in scored.candidates:
        choices[c.key].append(c)
    selected, lower = _hull_allocation(choices, weights, available)
    tensors = {}
    for name, t in scored.tensors.items():
        experts = []
        for expert in range(t.n_experts):
            key = EwItemKey(name, t.layer, t.projection, expert)
            c = selected[key]
            loss = weights[key] * c.distortion
            experts.append(
                ExpertSelection(
                    expert,
                    c.precision.nint_spec,
                    c.variable_storage_bits + (t.fixed_storage_bits if expert == 0 else 0),
                    loss,
                    loss,
                    c.precision,
                )
            )
        tensors[name] = ExpertTensorSelection(
            name, t.group, t.n_experts, t.rows_per_expert, t.columns, tuple(experts)
        )
    objective = sum(weights[key] * c.distortion for key, c in selected.items())
    metadata = {
        **importance.metadata,
        "solver": "lower-hull LP + feasible rounding",
        "integer_optimality_proven": False,
        "storage_accounting": scored.metadata.get(
            "storage_accounting", "candidate variable bits + tensor fixed bits"
        ),
        "quality_status": "not evaluated; surrogate only",
    }
    scheme = CalibrationScheme(
        None,
        budget.target_profile,
        maximum - budget.model_fixed_storage_bits,
        {},
        metadata,
        {},
        expert_selections=tensors,
    )
    report = {
        **metadata,
        "format": "mfq.alphaq-allocation.v1",
        "status": "feasible",
        "items": len(selected),
        "candidates": len(scored.candidates),
        "routed_weight_count": scored.routed_weight_count,
        "routed_storage_bits": scheme.storage_bits,
        "routed_bpw": scheme.bpw,
        "target_storage_bits": scheme.target_storage_bits,
        "model_storage_bits": scheme.storage_bits + budget.model_fixed_storage_bits,
        "model_bpw": (scheme.storage_bits + budget.model_fixed_storage_bits)
        / budget.model_weight_count,
        "objective": objective,
        "relaxation_lower_bound": lower,
        "absolute_gap": max(0.0, objective - lower),
        "relative_gap": max(0.0, objective - lower) / objective if objective else 0.0,
        "selected_counts": dict(Counter(c.profile for c in selected.values())),
    }
    return EwSolveResult(scheme, report, selected)


__all__ = [
    "AlphaQTensorStatistics",
    "alphaq_weight_statistics",
    "alphaq_importance",
    "alphaq_candidates",
    "alphaq_nint_candidates",
    "allocate_alphaq",
]
