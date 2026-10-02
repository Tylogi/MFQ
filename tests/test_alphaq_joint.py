"""Independent cost/objective checks for the public joint allocator."""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from mfq.calibration.alphaq import AlphaQTensorStatistics, alphaq_builtin_candidates
from mfq.calibration.alphaq_joint import DenseChoice, allocate_joint, dense_choices
from mfq.calibration.artifact import load_scheme, save_scheme
from mfq.calibration.ew_solver import EwItemKey


def inputs():
    experts = [AlphaQTensorStatistics("experts", 0, "gate", (2, 16, 640), (2., 4.), (1., .5))]
    dense = [AlphaQTensorStatistics("small", 0, "dense", (1, 1, 640), (3.,), (1.,)),
             AlphaQTensorStatistics("large", 0, "dense", (1, 16, 640), (2.,), (.3,))]
    profiles = ("NVQ2J", "NINT4", "NINT8")
    candidates = alphaq_builtin_candidates(experts, profiles)
    dc = dense_choices(dense, {s.name: "BF16" for s in dense}, profiles)
    return experts, candidates, dense, dc


def actual_cost(selected, table):
    pools = {(c.key.tensor, c.pool_key): c.pool_storage_bits for c in selected}
    return (sum(c.variable_storage_bits for c in selected) + sum(pools.values())
            + sum(t.fixed_storage_bits for t in table.tensors.values()))


@pytest.mark.parametrize("fraction", [.2, .55, .9])
def test_joint_budget_bound_matches_independent_enumeration(tmp_path, fraction):
    stats, table, ds, dc = inputs()
    all_candidates = (*table.candidates, *dc)
    keys = list(dict.fromkeys(c.key for c in all_candidates))
    options = [[c for c in all_candidates if c.key == k] for k in keys]
    fixed = 312
    reserved = sum({(c.key.tensor,c.pool_key): c.pool_storage_bits for c in all_candidates}.values())
    tensor_fixed = sum(t.fixed_storage_bits for t in table.tensors.values())
    minimum = sum(min(c.variable_storage_bits for c in choices) for choices in options)
    maximum = sum(max(c.variable_storage_bits for c in choices) for choices in options)
    cap = fixed + (minimum+reserved+tensor_fixed+int((maximum-minimum)*fraction)+7)//8
    result = allocate_joint(stats, table, ds, dc, model_weight_count=32000,
                            maximum_file_bytes=cap, fixed_file_bytes=fixed,
                            expert_exposure={"experts": .1})
    # Independently reconstruct the global importance scale and exposure.
    median = np.median([2.,4.])
    scale = max(median/2*1, median/4*.5)
    weights = {keys[0]: (median/2*1)/scale*.1,
               keys[1]: (median/4*.5)/scale*.1}
    for stat in ds:
        weights[EwItemKey(stat.name,0,"dense",0)] = median/stat.alpha[0]*stat.variance[0]/scale
    feasible = [sum(weights[c.key]*c.distortion for c in selected)
                for selected in itertools.product(*options)
                if actual_cost(selected,table)+fixed*8 <= cap*8]
    optimum = min(feasible)
    assert result.report["objective"] >= optimum - 1e-12
    assert result.report["relaxation_lower_bound"] <= optimum + 1e-12
    exact = actual_cost(result.selected.values(),table)+fixed*8
    assert result.report["model_storage_bits"] == exact <= cap*8
    assert result.native_overrides["small"] == "BF16"
    assert "small" not in result.scheme.selections
    save_scheme(tmp_path/"joint.json", result.scheme)
    restored = load_scheme(tmp_path/"joint.json")
    assert restored.storage_bits+result.report["native_dense_storage_bits"]+fixed*8 == exact
    for choice in result.selected.values():
        if choice.precision is not None and choice.precision.nint_spec is not None:
            assert choice.precision.option("imatrix_weighted") is True


def test_dense_costs_are_complete_payloads_and_native_is_explicit():
    _, _, _, choices = inputs()
    assert all(c.pool_storage_bits == 0 and not c.pool_key for c in choices)
    small = next(c for c in choices if c.key.tensor == "small" and c.profile == "NATIVE")
    assert small.variable_storage_bits == (20+640*2)*8
    assert small.precision is None and small.native_dtype == "BF16"
    with pytest.raises(ValueError,match="zero error"):
        replace(small,distortion=1.)


def test_joint_rejects_scope_and_budget_errors():
    es, ec, ds, dc = inputs()
    kwargs = dict(model_weight_count=32000, maximum_file_bytes=100000,
                  fixed_file_bytes=312, expert_exposure={"experts": .1})
    with pytest.raises(ValueError,match="scope"):
        allocate_joint(es,ec,ds,dc[:-4],**kwargs)
    with pytest.raises(ValueError,match="parameter count"):
        allocate_joint(es,ec,ds,dc,**{**kwargs,"model_weight_count":1})
    with pytest.raises(ValueError,match="router multipliers"):
        allocate_joint(es,ec,ds,dc,router_multipliers={},**kwargs)
    with pytest.raises(ValueError,match="exceeds budget"):
        allocate_joint(es,ec,ds,dc,**{**kwargs,"maximum_file_bytes":1})


def test_zero_importance_uses_minimum_cost_without_fake_floor():
    es, ec, ds, dc = inputs()
    es = [replace(s,variance=(0.,)*s.shape[0]) for s in es]
    ds = [replace(s,variance=(0.,)) for s in ds]
    result = allocate_joint(es,ec,ds,dc,model_weight_count=32000,
                            maximum_file_bytes=100000,fixed_file_bytes=312,
                            expert_exposure={"experts": .1})
    assert result.report["objective"] == 0
    for key, chosen in result.selected.items():
        assert chosen.variable_storage_bits == min(
            c.variable_storage_bits for c in (*ec.candidates,*dc) if c.key == key)
