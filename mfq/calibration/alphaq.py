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
    ExpertPrecision,
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
ALPHAQ_PROFILES = (
    "NVQ1-S", "NVQ1-L",
    "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
    "NVQ3J", "NVQ3J-512", "NVQ3J-L",
    "NINT4", "NINT5", "NINT6", "NINT8",
)
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


def alphaq_weight_statistics(weights: Any, *, backend: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Batched FARMS/Hill statistics for a floating [E,O,I] torch tensor.

    CUDA uses full-FP32 Gram products and a batched symmetric solver when
    supported; CPU, tiny matrices and numerically degenerate tails use SVD.
    ``backend='svd'`` is the reference path; ``'gram'`` also permits CPU
    comparisons. No block subsampling; batches shrink only after actual OOM.
    """
    import torch

    if not isinstance(weights, torch.Tensor) or weights.ndim != 3:
        raise ValueError("AlphaQ weights must be a floating [E,O,I] torch tensor")
    if not weights.is_floating_point() or min(weights.shape) <= 0:
        raise ValueError("AlphaQ weights must be nonempty and floating point")
    if backend not in {"auto", "svd", "gram"}:
        raise ValueError("AlphaQ spectrum backend must be auto, svd or gram")
    from mfq.calibration.alphaq_spectrum import cuda_library, gram_eigenvalues
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
        gram_backend = min(rows, columns) >= 128 and (
            backend == "gram" or (backend == "auto" and w.is_cuda and cuda_library() is not None)
        )
        # Keep the relatively small spectra on the device. Failed batches
        # do not discard spectra from preceding batches.
        spectra = torch.empty(
            (blocks.shape[0], min(blocks.shape[-2:])), dtype=torch.float32, device=w.device
        )
        # cuSOLVER's generic batch interface uses signed-int matrix indexing.
        batch = min(blocks.shape[0], 2147483647//(128*128)) if gram_backend else blocks.shape[0]
        start = 0
        while start < blocks.shape[0]:
            count = min(batch, blocks.shape[0] - start)
            try:
                part = blocks[start : start + count]
                spectra[start : start + count] = (
                    gram_eigenvalues(part) if gram_backend else torch.linalg.svdvals(part).square()
                )
            except torch.OutOfMemoryError:
                if count == 1:
                    raise
                batch = max(1, count // 2)
                continue
            start += count
        eigen = spectra.reshape(experts, -1).double().cpu().numpy()
        if gram_backend:
            ordered = np.sort(eigen, axis=1)
            tail_count = min(ordered.shape[1]-1, max(10, int(ordered.shape[1]*.1)))
            threshold = ordered[:, -tail_count-1]
            largest = ordered[:, -1]
            # Gram roundoff can dominate a low-rank tail or nearly constant
            # spectrum. Recompute those banks with the established SVD path.
            unstable = (threshold <= largest*128*np.finfo(np.float32).eps) | (
                largest-threshold <= largest*128*np.finfo(np.float32).eps)
            if unstable.any():
                ids = np.flatnonzero(unstable)
                exact_alpha, exact_variance = alphaq_weight_statistics(
                    weights[torch.as_tensor(ids, device=weights.device)], backend="svd")
    eigen.sort(axis=1)
    n = eigen.shape[1]
    if n < 2:
        return np.ones(experts, dtype=np.float64), variance
    k = min(n - 1, max(10, int(n * 0.1)))
    threshold = np.maximum(eigen[:, -k - 1], 1e-12)
    tail = np.maximum(eigen[:, -k:], 1e-12)
    denominator = np.maximum(np.log(tail / threshold[:, None]).sum(axis=1), 1e-12)
    alpha = 1.0 + k / denominator
    if gram_backend and unstable.any():
        alpha[ids], variance[ids] = exact_alpha, exact_variance
    return alpha, variance


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


def alphaq_builtin_candidates(
    statistics: Sequence[AlphaQTensorStatistics], profiles: Sequence[str] | None = None
) -> EwCandidateTable:
    """Canonical NVQ/NINT choices, including MFE headers and shared tables.

    Per-expert rounding is an upper bound on cohort stream packing. Tables
    are charged once per used tensor/profile pool, never once per expert.
    No weight fitting is needed to determine these costs.
    """
    from mfq.formats.io import _MFE_HDR
    from mfq.formats.nvq import jsc_payload_nbytes
    from mfq.formats.nvq1_l import NVQ1_L_T8_S3
    from mfq.formats.nvq1_s import NVQ1_S
    from mfq.tools.quantize_hf_to_mfq import _mixed_moe_blob_nbytes, _NVQ_SPECS

    names = ALPHAQ_PROFILES if profiles is None else tuple(profiles)
    if not names or len(set(names)) != len(names) or set(names) - set(ALPHAQ_PROFILES):
        raise ValueError("profiles must be distinct built-in AlphaQ profiles")
    nint_names = tuple(n for n in names if n.startswith("NINT"))
    nint = alphaq_nint_candidates(statistics, nint_names) if nint_names else None
    nint_costs = {} if nint is None else {
        (c.key.tensor, c.profile): c for c in nint.candidates
    }
    tensors, candidates = {}, []
    for s in statistics:
        experts, rows, columns = s.shape
        tensors[s.name] = EwTensorSpec(
            s.name, s.name, s.layer, s.projection, experts, rows, columns,
            _MFE_HDR.size * 8, None,
        )
        for profile in names:
            if profile.startswith("NINT"):
                base = nint_costs[s.name, profile]
                precision = base.precision
                payload = base.variable_storage_bits // 8 - 4
            else:
                # Use explicit bank counts so estimation and streaming agree.
                options = (("banks", 4 if profile.startswith("NVQ2J") else 2),) if "J" in profile else ()
                precision = ExpertPrecision(profile, options=options)
                if profile == "NVQ1-S":
                    payload = NVQ1_S.payload_nbytes(rows, columns, include_codebook=False)
                elif profile == "NVQ1-L":
                    payload = NVQ1_L_T8_S3.payload_nbytes(rows, columns)
                else:
                    payload = jsc_payload_nbytes(_NVQ_SPECS[profile], rows, columns)
            single = _mixed_moe_blob_nbytes((1, rows, columns), (precision,), None)
            pool = single - _MFE_HDR.size - payload - 4
            if pool < 0:
                raise RuntimeError(f"negative pool storage for {profile}")
            for expert in range(experts):
                candidates.append(EwCandidate(
                    EwItemKey(s.name, s.layer, s.projection, expert), profile,
                    precision, (payload + 4) * 8, profile, pool * 8,
                    0.0, 0.0, payload * 8 / (rows * columns),
                ))
    return alphaq_candidates(EwCandidateTable(tensors, tuple(candidates), None, {
        "storage_accounting": "per-expert byte-rounded streams + expert IDs + used pool headers/tables + MFE tensor headers; excludes outer container",
        "profiles": list(names),
    }))


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
    distortion: str = "analytical",
    importance_multipliers: Mapping[EwItemKey, float] | None = None,
) -> EwSolveResult:
    """Allocate AlphaQ into the standard quantizer-consumable scheme.

    ``hull`` is the fast global upper-budget path. Shared nonzero pool
    charges, lower/shape/projection/layer constraints require explicit
    ``solver='exact'`` and reuse MFQ's existing joint EW solver.
    """
    importance = alphaq_importance(statistics)
    if distortion == "analytical":
        scored = alphaq_candidates(candidates)
    elif distortion == "imatrix_sse":
        if candidates.metadata.get("distortion_metric") != "imatrix_weighted_sse":
            raise ValueError("imatrix_sse requires measured imatrix_weighted_sse candidates")
        if any(not math.isfinite(c.distortion) or c.distortion < 0
               for c in candidates.candidates):
            raise ValueError("candidate SSE must be finite and nonnegative")
        scored = candidates
        importance = replace(importance, metadata={
            **importance.metadata,
            "formula": "median(alpha) / alpha * variance * imatrix_weighted_sse",
            "distortion_metric": "imatrix_weighted_sse",
        })
    else:
        raise ValueError("AlphaQ distortion must be analytical or imatrix_sse")
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
    if importance_multipliers is not None:
        if set(importance_multipliers) != set(scored.items):
            raise ValueError("importance multipliers must cover every candidate item exactly")
        if any(not math.isfinite(v) or v < 0 for v in importance_multipliers.values()):
            raise ValueError("importance multipliers must be finite and nonnegative")
        importance = replace(importance, entries=tuple(
            replace(entry, score=entry.score * importance_multipliers[
                EwItemKey(entry.tensor, entry.layer, entry.projection, entry.expert_id)])
            for entry in importance.entries),
            metadata={**importance.metadata, "external_importance_multiplier": True,
                      "formula": "(" + importance.metadata["formula"] + ") * importance_multiplier"})
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
    ):
        raise ValueError(
            "AlphaQ hull requires one upper budget; use --solver exact for joint constraints"
        )
    fixed = sum(t.fixed_storage_bits for t in scored.tensors.values())
    available = maximum - budget.model_fixed_storage_bits - fixed
    choices: dict[EwItemKey, list[EwCandidate]] = defaultdict(list)
    for c in scored.candidates:
        choices[c.key].append(c)
    pools = {}
    for c in scored.candidates:
        pool_key = (c.key.tensor, c.pool_key)
        pool_spec = (c.pool_storage_bits, c.precision)
        if pool_key in pools and pools[pool_key] != pool_spec:
            raise ValueError(f"inconsistent AlphaQ pool storage or precision: {pool_key}")
        pools[pool_key] = pool_spec
    reserved_pool_bits = sum(bits for bits, _ in pools.values())
    try:
        selected, lower = _hull_allocation(choices, weights, available - reserved_pool_bits)
    except ValueError as exc:
        if reserved_pool_bits:
            raise ValueError(
                "all-pool reservation exceeds budget; use --solver exact to test shared-pool feasibility"
            ) from exc
        raise
    if reserved_pool_bits:
        # Ignoring activation costs is a relaxation of the original problem.
        # The bound from the reserved-budget solve is NOT a valid lower bound.
        _, lower = _hull_allocation(choices, weights, available)
    pool_members = defaultdict(list)
    for key, c in selected.items():
        pool_members[(key.tensor, c.pool_key)].append(key)
    pool_charges = {
        min(members): pools[pool][0] for pool, members in pool_members.items()
    }
    used_pool_bits = sum(pool_charges.values())
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
                    c.variable_storage_bits + pool_charges.get(key, 0)
                    + (t.fixed_storage_bits if expert == 0 else 0),
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
        "pool_accounting": "reserve all enabled pools during allocation; charge only selected pools",
        "reserved_pool_bits": reserved_pool_bits,
        "used_pool_bits": used_pool_bits,
        "unused_pool_reservation_bits": reserved_pool_bits - used_pool_bits,
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
    "ALPHAQ_PROFILES",
    "alphaq_weight_statistics",
    "alphaq_importance",
    "alphaq_candidates",
    "alphaq_nint_candidates",
    "alphaq_builtin_candidates",
    "allocate_alphaq",
]
