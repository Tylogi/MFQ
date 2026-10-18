"""Independent mathematical and byte-budget checks for calibration-free NatScale."""
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import weibull_min

from mfq.calibration.alphaq import AlphaQTensorStatistics, alphaq_builtin_candidates
from mfq.calibration.ew_solver import EwItemKey
from mfq.calibration.natscale import (
    allocate_natscale, gradient_scores, natscale_exponent, projection_energies,
    realized_contrast, solve_hulls, weibull_shape_exponent,
)


@pytest.mark.parametrize('tau', [.01, 1., 12.])
def test_contrast_matches_direct_symmetric_kl_and_is_power_invariant(tau):
    g = np.array([0., .003, .2, .7, 9., 500.])
    beta = natscale_exponent(g, tau)
    q = g[g > 0]**(2*beta); q /= q.sum()
    u = np.full(len(q), 1/len(q))
    direct = np.sum(q*np.log(q/u)+u*np.log(u/q))
    assert direct == pytest.approx(tau, rel=1e-10)
    assert natscale_exponent(g**2.3, tau)*2.3 == pytest.approx(beta)
    assert natscale_exponent(g*1e40, tau) == pytest.approx(beta)


def test_weibull_mle_shape_and_exact_one_nat():
    g = np.array([.0001, .03, .4, .5, 2., 7., 100.])
    beta = weibull_shape_exponent(g)
    shape, _, _ = weibull_min.fit(np.sqrt(g), floc=0)
    assert beta == pytest.approx(shape/4, rel=1e-4)
    assert realized_contrast(g, beta) == pytest.approx(1., abs=1e-12)
    extreme = np.array([1e-300, 1e-100, 1e100, 1e300])
    assert realized_contrast(extreme, natscale_exponent(extreme)) == pytest.approx(1.)


@pytest.mark.parametrize('scores', [[], [0, 0], [1, 1], [0, 7, 7], [-1, 2], [1, np.nan]])
def test_invalid_contrast_support(scores):
    with pytest.raises(ValueError):
        natscale_exponent(scores)


@pytest.mark.parametrize('tau', [0, -1, np.inf, np.nan])
def test_invalid_contrast_target(tau):
    with pytest.raises(ValueError, match='tau'):
        natscale_exponent([1, 2], tau)


@pytest.mark.parametrize('seed', range(5))
def test_lower_hull_lp_certificate_against_scipy_and_brute_force(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 15, size=(3, 8))
    costs = rng.random((3, 8))
    budget = int(sizes.min(axis=1).sum()+8)
    result = solve_hulls(sizes, costs, budget)
    normalized = (costs-costs.min(axis=1, keepdims=True))/costs.max(axis=1).sum()
    linear = linprog(normalized.ravel(), A_ub=sizes.ravel()[None], b_ub=[budget],
        A_eq=np.kron(np.eye(3), np.ones((1, 8))), b_eq=np.ones(3), bounds=(0, None), method='highs')
    assert linear.success
    assert result.lower_bound == pytest.approx(linear.fun, abs=1e-12)
    optimum = min(sum(normalized[row, c] for row, c in enumerate(combos))
        for combos in itertools.product(range(8), repeat=3)
        if sum(sizes[row, c] for row, c in enumerate(combos)) <= budget)
    assert result.lower_bound <= optimum+1e-12 <= result.objective+1e-12
    assert result.payload_bytes == sum(sizes[r, c] for r, c in enumerate(result.selected)) <= budget
    assert np.array_equal(result.selected, solve_hulls(sizes, costs, budget).selected)


def test_hull_ties_boundary_rounding_and_degenerate_rejection():
    result = solve_hulls([1, 3, 3, 7], [[8, 4, 4, 1], [8, 4, 4, 1]], 5)
    assert result.selected.tolist() == [1, 0]
    assert result.payload_bytes == 4
    with pytest.raises(ValueError, match='minimum payload'):
        solve_hulls([2, 3], [[1, 0], [1, 0]], 3)
    with pytest.raises(ValueError, match='degenerate'):
        solve_hulls([1, 2], [[0, 0]], 3)


def allocation_inputs():
    stats = []
    energies, exposures = {}, {}
    for projection in ('gate', 'up', 'down'):
        shape = (3, 8, 128) if projection != 'down' else (3, 128, 8)
        name = f'model.block.3.mlp.experts.{projection}.weight'
        stat = AlphaQTensorStatistics(name, 3, projection, shape, (2., 3., 4.), (1., 2., .5))
        stats.append(stat)
        energies[name] = np.array([10., 20., 0.])
        exposures.update({EwItemKey(name, 3, projection, e): f for e, f in enumerate((.7, .3, 0.))})
    table = alphaq_builtin_candidates(stats, ['NINT4', 'NINT8'])
    return stats, table, energies, exposures


def test_complete_configuration_cost_and_exact_payload_accounting():
    stats, table, energies, exposures = allocation_inputs()
    result = allocate_natscale(stats, table, energies=energies, exposures=exposures,
        maximum_file_bytes=18000, fixed_file_bytes=1000, model_weight_count=9216)
    candidates = result.selected.values()
    pools = {(c.key.tensor, c.pool_key): c.pool_storage_bits for c in candidates}
    actual = sum(c.variable_storage_bits for c in candidates)+sum(pools.values())
    actual += sum(t.fixed_storage_bits for t in table.tensors.values())
    assert result.scheme.storage_bits == actual
    assert actual/8+1000 <= 18000
    assert result.report['exposure']['zero'] == 1
    assert result.report['beta'] == result.report['realized_J'] == 0
    assert result.report['lp_gap'] >= 0
    assert set(result.report['selected_profile_counts']) == {'gate', 'up', 'down'}
    # Independently evaluate the selected full configurations in raw units.
    median = np.median(np.concatenate([s.alpha for s in stats]))
    expected = 0.
    for e, f in enumerate((.7, .3, 0.)):
        cs = [result.selected[EwItemKey(s.name, 3, s.projection, e)] for s in stats]
        a = sum(median/s.alpha[e]*s.variance[e]*c.distortion for s, c in zip(stats, cs))/3
        expected += a*f*np.sqrt(energies[stats[-1].name][e]/128*sum(c.distortion for c in cs))
    assert result.report['raw_objective'] == pytest.approx(expected)
    for s in result.scheme.expert_selections.values():
        assert all(p.option('imatrix_weighted') is True for p in s.precisions)


def test_measured_errors_nearest_layer_and_energy_sensitivity():
    stats, table, energies, exposures = allocation_inputs()
    errors = {'2': {p: {'NINT4': .15, 'NINT8': .003} for p in ('gate', 'up', 'down')}}
    result = allocate_natscale(stats, table, energies=energies, exposures=exposures,
        errors=errors, sensitivity='imatrix-energy', maximum_file_bytes=18000,
        fixed_file_bytes=1000, model_weight_count=9216)
    assert result.report['realized_J'] == pytest.approx(1.)
    assert result.report['error'] == 'measured'


def test_gradient_schema_checks_identity_coverage_and_duplicates(tmp_path):
    path = tmp_path/'gradients.json'
    doc = dict(format='mfq.natscale-gradients.v1', source={'model': 'test'},
               experts=[dict(layer=3, expert=0, second_moment=.5)])
    path.write_text(json.dumps(doc))
    assert gradient_scores(path, [(3, 0)], doc['source']).tolist() == [.5]
    with pytest.raises(ValueError, match='source'):
        gradient_scores(path, [(3, 0)], {})
    doc['experts'] *= 2; path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='duplicate'):
        gradient_scores(path, [(3, 0)], doc['source'])
