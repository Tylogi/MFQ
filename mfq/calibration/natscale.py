"""Calibration-free NatScale: imatrix exposure/energy and joint G/U/D choices."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.calibration.alphaq import alphaq_candidates
from mfq.calibration.alphaq_joint import JointAllocation, _imatrix_precision
from mfq.calibration.artifact import CalibrationScheme, ExpertSelection, ExpertTensorSelection
from mfq.calibration.ew_solver import EwItemKey


def _positive_logs(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('sensitivity scores must be a finite nonnegative vector')
    logs = np.log(values[values > 0])
    if not logs.size:
        raise ValueError('sensitivity scores have empty positive support')
    if np.ptp(logs) == 0:
        raise ValueError('sensitivity scores have constant positive support')
    return logs


def _contrast(logs, t):
    # Center and scale before exponentiation. No powers of unscaled scores
    # are formed, even for subnormal scores or very large exponents.
    x = logs - logs.max()
    q = np.exp(t*x)
    q /= q.sum()
    return float(t*(np.dot(q, x)-x.mean()))


def realized_contrast(scores, beta):
    return _contrast(_positive_logs(scores), 2*beta)


def natscale_exponent(scores, tau=1.0):
    """Solve Jeffreys(softmax(2 beta log(g)), uniform) = tau."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('NatScale tau must be finite and positive')
    logs = _positive_logs(scores)
    span = float(np.ptp(logs))
    scaled = (logs-logs.max())/span
    low, high = 0., 1.
    while _contrast(scaled, high) < tau:
        high *= 2
        if not math.isfinite(high):
            raise ValueError('cannot bracket the requested contrast')
    for _ in range(100):
        mid = (low+high)/2
        if _contrast(scaled, mid) < tau:
            low = mid
        else:
            high = mid
    beta = ((low+high)/2)/(2*span)
    if not math.isclose(_contrast(logs, 2*beta), tau, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('NatScale realized contrast differs from tau')
    return beta


def weibull_shape_exponent(scores):
    """Unweighted Weibull MLE on sqrt(g), returning k-hat / 4.

    Its likelihood equation k*(E_softmax(k z)[z]-mean(z))=1 is
    precisely the one-nat Jeffreys equation, with z=log(sqrt(g)).
    """
    return natscale_exponent(scores, 1.0)


@dataclass(frozen=True)
class HullSolution:
    selected: np.ndarray
    payload_bytes: int
    objective: float
    lower_bound: float
    scale: float
    row_offset: float


def solve_hulls(payloads, costs, budget):
    """Stable lower hull LP and deterministic feasible boundary rounding.

    Candidate arithmetic/pruning is vectorized. Only nondominated points
    enter the per-expert hull loop; Q-cubed Python candidate objects are avoided.
    """
    costs = np.asarray(costs, dtype=np.float64)
    raw_bytes = np.asarray(payloads)
    if (costs.ndim != 2 or min(costs.shape) <= 0 or not np.isfinite(costs).all()
            or (costs < 0).any() or not np.isfinite(raw_bytes).all()
            or (raw_bytes < 0).any() or not np.equal(raw_bytes, np.floor(raw_bytes)).all()
            or type(budget) is not int or budget < 0):
        raise ValueError('invalid NatScale payloads, costs or byte budget')
    payloads = np.broadcast_to(raw_bytes, costs.shape).astype(np.int64)
    scale = float(costs.max(axis=1).sum())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('NatScale costs have a degenerate global scale')
    minima = costs.min(axis=1)
    normalized = (costs-minima[:, None])/scale
    chosen = np.empty(costs.shape[0], dtype=np.int64)
    used, initial, events = 0, 0., []
    indices = np.arange(costs.shape[1])
    for row, losses in enumerate(normalized):
        sizes = payloads[row]
        ordered = np.lexsort((indices, losses, sizes))
        b, d = sizes[ordered], losses[ordered]
        unique = np.r_[True, b[1:] != b[:-1]]
        ordered = ordered[unique]
        d = losses[ordered]
        improving = np.r_[True, d[1:] < np.minimum.accumulate(d)[:-1]]
        hull = []
        for c in ordered[improving]:
            while len(hull) >= 2:
                a, mid = hull[-2:]
                left = (losses[mid]-losses[a])/(sizes[mid]-sizes[a])
                right = (losses[c]-losses[mid])/(sizes[c]-sizes[mid])
                if right > left:
                    break
                hull.pop()
            hull.append(int(c))
        chosen[row] = hull[0]
        used += int(sizes[hull[0]])
        initial += float(losses[hull[0]])
        for a, b in zip(hull, hull[1:]):
            extra = int(sizes[b]-sizes[a])
            benefit = float(losses[a]-losses[b])
            events.append((-benefit/extra, len(events), row, a, b, extra, benefit))
    if used > budget:
        raise ValueError(f'NatScale minimum payload {used} bytes exceeds budget {budget}')
    events.sort(key=lambda e: (e[0], e[1]))
    lower, fractional = initial, False
    for _, _, row, a, b, extra, benefit in events:
        if not fractional:
            if used+extra > budget:
                lower -= (budget-used)/extra*benefit
                fractional = True
            else:
                lower -= benefit
        if chosen[row] == a and used+extra <= budget:
            chosen[row] = b
            used += extra
    objective = float(normalized[np.arange(costs.shape[0]), chosen].sum())
    return HullSolution(chosen, used, objective, max(0., lower), scale, float(minima.sum()))


def projection_energies(statistics, scope, imatrix):
    """P = sum(input second moment * column energy); imatrix is normalized on load."""
    result = {}
    for stat in statistics:
        if stat.column_energy is None:
            raise ValueError(f'NatScale needs column energies; recollect statistics: {stat.name}')
        entry = imatrix.entries[scope.imatrix_entries[stat.name]]
        moments = np.asarray(entry.values, dtype=np.float64)
        if moments.shape != (stat.shape[0], stat.shape[2]):
            raise ValueError(f'NatScale imatrix shape differs: {stat.name}')
        if not np.isfinite(moments).all() or (moments < 0).any():
            raise ValueError('NatScale second moments must be finite and nonnegative')
        energy = np.sum(moments*stat.column_energy, axis=1)
        energy[np.asarray(entry.counts) == 0] = 0
        result[stat.name] = energy
    return result


def gradient_scores(path, keys, source):
    doc = json.loads(Path(path).read_text())
    if doc.get('format') != 'mfq.natscale-gradients.v1' or doc.get('source') != source:
        raise ValueError('NatScale gradients do not match source/schema')
    values = {}
    for entry in doc['experts']:
        key = (entry['layer'], entry['expert'])
        value = entry['second_moment']
        if (key in values or any(type(v) is not int or v < 0 for v in key)
                or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError('invalid or duplicate NatScale gradient expert')
        values[key] = value
    if set(values) != set(keys):
        raise ValueError('NatScale gradients must cover exactly all target experts')
    return np.array([values[key] for key in keys], dtype=np.float64)


def allocate_natscale(statistics, candidates, *, energies, exposures,
                       maximum_file_bytes, fixed_file_bytes, model_weight_count,
                       tau=1.0, sensitivity='uniform', errors=None, source=None):
    """Allocate complete Q^3 configurations, reserving shared pools once per tensor/profile."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('NatScale tau must be finite and positive')
    scored = alphaq_candidates(candidates)
    stats = {s.name: s for s in statistics}
    if (len(stats) != len(statistics) or set(stats) != set(scored.tensors)
            or set(energies) != set(stats) or set(exposures) != set(scored.items)):
        raise ValueError('NatScale statistics/candidate/energy/exposure scope differs')
    choices, layers = defaultdict(list), defaultdict(dict)
    for c in scored.candidates:
        choices[c.key].append(c)
    for stat in statistics:
        if stat.projection not in ('gate', 'up', 'down') or stat.projection in layers[stat.layer]:
            raise ValueError('NatScale requires unique separate gate/up/down banks')
        tensor = scored.tensors[stat.name]
        if stat.shape != (tensor.n_experts, tensor.rows_per_expert, tensor.columns):
            raise ValueError('NatScale candidate shape differs from statistics')
        layers[stat.layer][stat.projection] = stat
    if not layers or any(set(banks) != {'gate', 'up', 'down'} for banks in layers.values()):
        raise ValueError('NatScale requires gate/up/down in every routed layer')
    median = float(np.median(np.concatenate([s.alpha for s in statistics])))
    all_costs, all_bytes, identities, catalog, scores, freq = [], [], [], [], [], []
    for layer, banks in sorted(layers.items()):
        ordered = [banks[p] for p in ('gate', 'up', 'down')]
        gate, up, down = ordered
        count, width, hidden = gate.shape
        if up.shape != gate.shape or down.shape != (count, hidden, width):
            raise ValueError('NatScale expert dimensions are inconsistent')
        projected, sizes, decay, rho = [], [], [], []
        for stat in ordered:
            matrix = [choices[EwItemKey(stat.name, layer, stat.projection, e)] for e in range(count)]
            profiles = tuple(c.profile for c in matrix[0])
            if any(tuple(c.profile for c in row) != profiles for row in matrix):
                raise ValueError('NatScale candidates must share a profile order within a bank')
            projected.append(matrix)
            sizes.append(np.array([[c.variable_storage_bits//8 for c in row] for row in matrix]))
            decay.append(np.array([[c.distortion for c in row] for row in matrix]))
            if errors is None:
                rho.append(decay[-1])
            else:
                sampled = min(errors, key=lambda l: (abs(int(l)-layer), int(l)))
                values = errors[sampled][stat.projection]
                rho.append(np.broadcast_to([values[p] for p in profiles], (count, len(profiles))))
        combos = np.array(np.meshgrid(*[np.arange(len(m[0])) for m in projected], indexing='ij')).reshape(3, -1).T
        byte_rows = sum(sizes[p][:, combos[:, p]] for p in range(3))
        a = sum((median/np.array(stat.alpha)*np.array(stat.variance))[:, None]
                *decay[p][:, combos[:, p]] for p, stat in enumerate(ordered))/3
        pd = np.asarray(energies[down.name], dtype=np.float64)/hidden
        if pd.shape != (count,) or not np.isfinite(pd).all() or (pd < 0).any():
            raise ValueError('NatScale energies must be finite nonnegative per-expert vectors')
        f = np.array([exposures[EwItemKey(gate.name, layer, 'gate', e)] for e in range(count)])
        if not np.isfinite(f).all() or (f < 0).any() or (f > 1).any():
            raise ValueError('NatScale exposure must be in [0,1]')
        for stat in (up, down):
            if any(exposures[EwItemKey(stat.name, layer, stat.projection, e)] != f[e] for e in range(count)):
                raise ValueError('NatScale projection exposures disagree')
        r = sum(rho[p][:, combos[:, p]] for p in range(3))
        if not np.isfinite(r).all() or (r < 0).any():
            raise ValueError('NatScale relative errors must be finite and nonnegative')
        all_costs.append(a*f[:, None]*np.sqrt(pd[:, None]*r))
        all_bytes.append(byte_rows)
        catalog.extend([(projected, combos, e) for e in range(count)])
        identities.extend((layer, e) for e in range(count))
        scores.extend(pd); freq.extend(f)
    if sensitivity == 'uniform':
        beta, contrast, weights = 0., 0., np.ones(len(identities))
        log_weight_scale = 0.
    else:
        g = (np.asarray(scores) if sensitivity == 'imatrix-energy' else
             gradient_scores(sensitivity.removeprefix('gradient:'), identities, source)
             if sensitivity.startswith('gradient:') else None)
        if g is None:
            raise ValueError('NatScale sensitivity must be uniform, imatrix-energy or gradient:PATH')
        beta = natscale_exponent(g, tau)
        contrast = realized_contrast(g, beta)
        weights = np.zeros_like(g)
        logs = np.log(g[g > 0])
        log_weight_scale = float(beta*logs.max())
        weights[g > 0] = np.exp(beta*(logs-logs.max()))
    costs = np.concatenate(all_costs)*weights[:, None]
    payloads = np.concatenate(all_bytes)
    pools = {}
    for c in scored.candidates:
        key = (c.key.tensor, c.pool_key)
        if c.variable_storage_bits % 8 or c.pool_storage_bits % 8:
            raise ValueError('NatScale candidate costs must be whole bytes')
        if key in pools and pools[key] != c.pool_storage_bits:
            raise ValueError('NatScale shared pool costs disagree')
        pools[key] = c.pool_storage_bits
    fixed_bits = sum(t.fixed_storage_bits for t in scored.tensors.values())
    reserved = sum(pools.values())
    available = (maximum_file_bytes-fixed_file_bytes)*8-fixed_bits
    solution = solve_hulls(payloads, costs, (available-reserved)//8)
    # Ignoring pool activation costs gives a valid lower bound for the full
    # fixed-charge problem, including configurations outside reserved capacity.
    relaxed = solve_hulls(payloads, costs, available//8) if reserved else solution
    selected, selected_losses = {}, {}
    counts, layer_bits, layer_weights = defaultdict(Counter), Counter(), Counter()
    for row, index in enumerate(solution.selected):
        matrix, combos, expert = catalog[row]
        for p, ci in enumerate(combos[index]):
            c = matrix[p][expert][ci]
            selected[c.key] = c
            selected_losses[c.key] = float(costs[row, index]/3)
            counts[c.key.projection][c.profile] += 1
    members = defaultdict(list)
    for key, c in selected.items():
        members[(key.tensor, c.pool_key)].append(key)
    charges = {min(keys): pools[pool] for pool, keys in members.items()}
    expert_selections = {}
    for name, tensor in scored.tensors.items():
        selections = []
        for e in range(tensor.n_experts):
            key = EwItemKey(name, tensor.layer, tensor.projection, e)
            c = selected[key]
            bits = c.variable_storage_bits+charges.get(key, 0)+(tensor.fixed_storage_bits if e == 0 else 0)
            precision = _imatrix_precision(c.precision)
            loss = selected_losses[key]
            selections.append(ExpertSelection(e, precision.nint_spec, bits, loss, loss, precision))
            layer_bits[tensor.layer] += bits
            layer_weights[tensor.layer] += tensor.rows_per_expert*tensor.columns
        expert_selections[name] = ExpertTensorSelection(name, tensor.group, tensor.n_experts,
            tensor.rows_per_expert, tensor.columns, tuple(selections))
    objective = solution.objective
    lower = relaxed.lower_bound
    gap = max(0., objective-lower)
    metadata = dict(method='NatScale', error='analytic' if errors is None else 'measured',
        sensitivity=sensitivity, beta=beta, realized_J=contrast, tau=tau,
        sensitivity_log_weight_scale=log_weight_scale,
        objective=objective, relaxation_lower_bound=lower, lp_gap=gap,
        raw_objective=objective*solution.scale+solution.row_offset,
        raw_lp_gap=gap*solution.scale, global_scale=solution.scale, row_offset=solution.row_offset,
        integer_optimality_proven=False, quality_status='not evaluated; surrogate only',
        formula='w * mean(AlphaQ) * exposure * sqrt(P_down/H * sum(rho))')
    scheme = CalibrationScheme(None, 'NatScale-full-file', available, {}, metadata, {},
                               expert_selections=expert_selections)
    if scheme.storage_bits+fixed_file_bytes*8 > maximum_file_bytes*8:
        raise RuntimeError('NatScale allocation exceeds its whole-file ceiling')
    report = dict(**metadata, maximum_file_bytes=maximum_file_bytes, fixed_file_bytes=fixed_file_bytes,
        model_weight_count=model_weight_count, model_storage_bits=scheme.storage_bits+fixed_file_bytes*8,
        reserved_pool_bits=reserved, used_pool_bits=sum(charges.values()),
        selected_profile_counts={p: dict(v) for p, v in counts.items()},
        per_layer_bpw={str(l): layer_bits[l]/layer_weights[l] for l in layer_bits},
        exposure=dict(min=float(min(freq)), max=float(max(freq)), mean=float(np.mean(freq)),
                      zero=int(np.count_nonzero(np.asarray(freq) == 0)), experts=len(freq)),
        relative_gap=gap/objective if objective else 0.)
    return JointAllocation(scheme, {}, selected, report)
