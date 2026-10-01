from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from mfq import cli
from mfq.calibration.alphaq import (
    AlphaQTensorStatistics,
    allocate_alphaq,
    alphaq_candidates,
    alphaq_importance,
    alphaq_nint_candidates,
    alphaq_weight_statistics,
)
from mfq.calibration.alphaq_source import collect_alphaq
from mfq.calibration.artifact import ExpertPrecision, load_scheme
from mfq.calibration.ew_solver import (
    EwBudget,
    RateBounds,
)
from mfq.formats.nint import NintSpec

torch = pytest.importorskip("torch")


def _oracle(w):
    result = []
    for x in w:
        if min(x.shape) >= 128:
            blocks = [
                x[r : r + 128, c : c + 128]
                for r in range(0, x.shape[0] - 127, 128)
                for c in range(0, x.shape[1] - 127, 128)
            ]
        else:
            blocks = [x]
        eig = np.sort(
            np.concatenate(
                [np.linalg.svd(b.astype(np.float64), compute_uv=False) ** 2 for b in blocks]
            )
        )
        if len(eig) < 2:
            result.append(1.0)
            continue
        k = min(len(eig) - 1, max(10, int(len(eig) * 0.1)))
        result.append(
            1 + k / max(np.log(np.maximum(eig[-k:], 1e-12) / max(eig[-k - 1], 1e-12)).sum(), 1e-12)
        )
    return np.asarray(result), w.var(axis=(-2, -1), dtype=np.float64)


@pytest.mark.parametrize("shape", [(3, 256, 384), (2, 145, 267), (3, 17, 31), (2, 1, 1)])
def test_batched_spectral_statistics_match_scalar_svd(shape):
    w = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    actual = alphaq_weight_statistics(torch.from_numpy(w))
    expected = _oracle(w)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_on_batched_real_block_geometry():
    w = torch.randn((4, 512, 512), generator=torch.Generator().manual_seed(8))
    expected = alphaq_weight_statistics(w)
    actual = alphaq_weight_statistics(w.cuda())
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-4)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-6)


def test_degenerate_nonfinite_and_invalid_weights():
    a, v = alphaq_weight_statistics(torch.zeros(2, 128, 128))
    assert np.isfinite(a).all() and (v == 0).all()
    with pytest.raises(ValueError, match="NaN"):
        alphaq_weight_statistics(torch.full((1, 128, 128), float("nan")))
    with pytest.raises(ValueError, match="floating"):
        alphaq_weight_statistics(torch.zeros(1, 2, 3, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"\[E,O,I\]"):
        alphaq_weight_statistics(torch.zeros(2, 3))


def test_oom_retry_preserves_all_blocks(monkeypatch):
    original = torch.linalg.svdvals
    calls = []

    def fail_large(blocks):
        calls.append(blocks.shape[0])
        if blocks.shape[0] > 2:
            raise torch.OutOfMemoryError("injected")
        return original(blocks)

    w = torch.randn((3, 128, 256), generator=torch.Generator().manual_seed(1))
    expected = alphaq_weight_statistics(w)
    monkeypatch.setattr(torch.linalg, "svdvals", fail_large)
    actual = alphaq_weight_statistics(w)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5)
    assert calls[0] == 6 and sum(c for c in calls if c <= 2) == 6


def _statistics(experts=3):
    return (
        AlphaQTensorStatistics(
            "model.block.0.mlp.experts.down.weight",
            0,
            "down",
            (experts, 128, 128),
            tuple(2.0 + x for x in range(experts)),
            tuple(0.01 * (x + 1) for x in range(experts)),
        ),
    )


def _budget(table, max_bits):
    return EwBudget(
        "AlphaQ-test", table.routed_weight_count, 0, RateBounds(max_bits=max_bits), {}, {}, (), {}
    )


def test_importance_matches_code_default_without_per_layer_normalization():
    s = _statistics()
    second = replace(
        s[0],
        name="model.block.1.mlp.experts.down.weight",
        layer=1,
        variance=tuple(10 * x for x in s[0].variance),
    )
    table = alphaq_importance((*s, second))
    scores = np.array([e.score for e in table.entries])
    expected = np.tile(1 / np.asarray(s[0].alpha), 2) * np.r_[s[0].variance, second.variance]
    np.testing.assert_allclose(scores, expected / expected.max())
    np.testing.assert_allclose(scores[3:] / scores[:3], 10)
    assert table.score_normalization == "none"


def test_nint_candidate_payload_matches_v2_serializer():
    from mfq.formats.io import _NINT_HDR, pack_nint
    from mfq.formats.nint import NintTensor

    stats = (replace(_statistics(1)[0], shape=(1, 5, 49)),)
    table = alphaq_nint_candidates(stats)
    for c in table.candidates:
        spec = c.precision.nint_spec
        groups = math.ceil(49 / spec.groupsize)
        tensor = NintTensor(
            spec,
            (5, 49),
            0,
            np.zeros((5, groups, spec.groupsize), dtype=np.uint8),
            np.ones(5),
            np.ones(5),
            np.zeros((5, groups), dtype=np.uint8),
            np.zeros((5, groups), dtype=np.uint8),
            49,
        )
        header_bytes = _NINT_HDR.size + 4 + 2 * 8 + 8
        payload = len(pack_nint(tensor)) - header_bytes
        assert c.variable_storage_bits == (payload + 4) * 8
        assert c.distortion == 2.0 ** (-2 * spec.bits)
        assert c.effective_bpw == payload * 8 / (5 * 49)
    assert {c.profile: c.precision.nint_spec for c in table.candidates}["NINT5"] == NintSpec(
        5, 28, 7
    )


def test_vector_candidates_use_payload_bpw_and_keep_artifacts_and_costs():
    table = alphaq_nint_candidates(_statistics(1), ("NINT2",))
    c = replace(
        table.candidates[0],
        profile="NVQ2J-L",
        effective_bpw=2.3125,
        precision=ExpertPrecision("NVQ2J-L", artifact="tables/c.npz"),
        pool_storage_bits=6400,
        distortion=123.0,
        validation_distortion=321.0,
    )
    scored = alphaq_candidates(replace(table, candidates=(c,)))
    assert scored.candidates[0].distortion == 2.0 ** (-4.625)
    assert scored.candidates[0].variable_storage_bits == c.variable_storage_bits
    assert scored.candidates[0].pool_storage_bits == c.pool_storage_bits
    assert scored.candidates[0].precision == c.precision


def test_hull_solver_bounds_bruteforce_optimum_and_spends_global_budget():
    stats = _statistics()
    table = alphaq_nint_candidates(stats, ("NINT2", "NINT3", "NINT4"))
    items = table.items
    choices = [[c for c in table.candidates if c.key == key] for key in items]
    weights = alphaq_importance(stats).weights_for(items)
    for capacity in (150000, 170000, 190000):
        result = allocate_alphaq(stats, table, _budget(table, capacity))
        options = [
            (
                sum(weights[c.key] * c.distortion for c in combination),
                sum(c.variable_storage_bits for c in combination),
            )
            for combination in itertools.product(*choices)
        ]
        optimum = min(loss for loss, cost in options if cost <= capacity)
        assert result.scheme.storage_bits <= capacity
        assert result.report["relaxation_lower_bound"] <= optimum + 1e-12
        assert result.report["objective"] >= optimum - 1e-12
        assert result.scheme.storage_bits == sum(
            c.variable_storage_bits for c in result.selected.values()
        )


def test_hull_relaxation_matches_general_lp_on_unequal_costs():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(14)
    stats = _statistics(12)
    base = alphaq_nint_candidates(stats, ("NINT2", "NINT3", "NINT4", "NINT5"))
    for _ in range(8):
        table = replace(
            base,
            candidates=tuple(
                replace(c, variable_storage_bits=int(rng.integers(100, 900)))
                for c in base.candidates
            ),
        )
        result = allocate_alphaq(stats, table, _budget(table, 5200))
        weights = alphaq_importance(stats).weights_for(table.items)
        costs = [c.variable_storage_bits for c in table.candidates]
        losses = [weights[c.key] * c.distortion for c in table.candidates]
        equal = np.array([[int(c.key == key) for c in table.candidates] for key in table.items])
        lp = scipy.linprog(
            losses, A_ub=[costs], b_ub=[5200], A_eq=equal, b_eq=np.ones(12), bounds=(0, 1)
        )
        assert lp.success
        assert result.report["relaxation_lower_bound"] == pytest.approx(lp.fun, abs=1e-10)


def test_exact_solver_keeps_pool_activation_charges():
    pytest.importorskip("scipy")
    stats = _statistics(2)
    table = alphaq_nint_candidates(stats, ("NINT2", "NINT4"))
    table = replace(
        table, candidates=tuple(replace(c, pool_storage_bits=200) for c in table.candidates)
    )
    budget = _budget(table, 120000)
    with pytest.raises(ValueError, match="--solver exact"):
        allocate_alphaq(stats, table, budget)
    result = allocate_alphaq(stats, table, budget, solver="exact")
    expected = sum(c.variable_storage_bits for c in result.selected.values())
    expected += 200 * len({c.pool_key for c in result.selected.values()})
    assert result.scheme.storage_bits == expected <= 120000
    assert result.scheme.metadata["method"] == "AlphaQ"


def test_no_silent_shape_mismatch_or_infeasible_budget():
    stats = _statistics()
    table = alphaq_nint_candidates(stats)
    with pytest.raises(ValueError, match="exceeds budget"):
        allocate_alphaq(stats, table, _budget(table, 1))
    with pytest.raises(ValueError, match="shape"):
        allocate_alphaq((replace(stats[0], shape=(3, 128, 129)),), table, _budget(table, 9999999))
    with pytest.raises(ValueError, match="same tensors"):
        allocate_alphaq((*stats, replace(stats[0], name="extra")), table, _budget(table, 9999999))


def _hf_model(tmp_path, *, separate=False):
    save_file = pytest.importorskip("safetensors.torch").save_file
    root = tmp_path / "model"
    root.mkdir()
    model_type = "glm5_next" if separate else "qwen3_5_moe"
    config = {"model_type": model_type, "num_hidden_layers": 1}
    if separate:
        config = {"model_type": model_type, "text_config": config}
    (root / "config.json").write_text(json.dumps(config))
    g = torch.Generator().manual_seed(3)
    if separate:
        tensors = {
            f"model.language_model.layers.0.mlp.experts.{e}.{p}_proj.weight": torch.randn(
                128, 128, generator=g
            ).bfloat16()
            for e in range(3)
            for p in ("gate", "up", "down")
        }
    else:
        tensors = {
            f"model.language_model.layers.0.mlp.experts.{p}_proj": torch.randn(
                3, rows, 128, generator=g
            ).bfloat16()
            for p, rows in (("gate_up", 256), ("down", 128))
        }
    save_file(tensors, root / "model.safetensors")
    return root


@pytest.mark.parametrize("separate", [False, True])
def test_cli_hf_scheme_is_consumed_by_actual_quantizer_and_cache_reused(
    tmp_path, monkeypatch, separate
):
    from mfq.calibration import alphaq_source
    from mfq.tools.quantize_hf_to_mfq import _plan

    model = _hf_model(tmp_path, separate=separate)
    output = tmp_path / "scheme.json"
    stats = tmp_path / "stats.json"
    args = [
        "calibrate",
        "alphaq",
        "--model",
        str(model),
        "--output",
        str(output),
        "--statistics",
        str(stats),
        "--target-bpw",
        "3.5",
        "--device",
        "cpu",
    ]
    assert cli.main(args) == 0
    scheme = load_scheme(output)
    plans = _plan(model, True, None, "f32", scheme)
    allocated = [p for p in plans if p.target_dtype == "MFE"]
    assert len(allocated) == (3 if separate else 2)
    assert all(p.expert_precisions is not None for p in allocated)
    assert all(len(p.expert_precisions) == 3 for p in allocated)
    assert scheme.bpw <= 3.5
    assert scheme.metadata["quality_status"] == "not evaluated; surrogate only"
    assert json.loads(stats.read_text())["complete"]

    def forbidden(*args, **kwargs):
        pytest.fail("repeated spectral computation")

    monkeypatch.setattr(alphaq_source, "alphaq_weight_statistics", forbidden)
    args[args.index(str(output))] = str(tmp_path / "second.json")
    args[args.index("3.5")] = "4.5"
    assert cli.main(args) == 0
    with pytest.raises(FileExistsError):
        cli.main(args)


def test_mfq_source_and_cache_identity(tmp_path):
    from mfq.formats.header import FileHeader
    from mfq.formats.io import save

    name = "model.block.0.mlp.experts.down.weight"
    path = tmp_path / "source.mfq"
    save(
        path,
        FileHeader(model_arch="test", num_tensors=1),
        {name: np.random.default_rng(9).normal(size=(2, 128, 128)).astype(np.float32)},
    )
    stats = tmp_path / "stats.json"
    values = collect_alphaq(path, stats, device="cpu")
    assert values[0].name == name and values[0].shape == (2, 128, 128)
    raw = json.loads(stats.read_text())
    raw["identity"]["model"] = "another-model"
    stats.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different source"):
        collect_alphaq(path, stats, device="cpu")


def test_optional_progress_io_does_not_fail_collection(monkeypatch):
    from mfq.calibration.alphaq_source import _progress

    def broken(*args, **kwargs):
        raise BrokenPipeError("injected")

    monkeypatch.setattr("builtins.print", broken)
    _progress(done=1)


def test_required_statistics_write_failure_propagates(tmp_path, monkeypatch):
    model = _hf_model(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("injected statistics failure")

    monkeypatch.setattr("mfq.calibration.alphaq_source._atomic_json", denied)
    with pytest.raises(PermissionError, match="statistics failure"):
        collect_alphaq(model, tmp_path / "statistics.json", device="cpu")


def test_zero_variance_gets_cheapest_candidate():
    stats = (replace(_statistics(2)[0], variance=(0.0, 0.0)),)
    table = alphaq_nint_candidates(stats)
    result = allocate_alphaq(stats, table, _budget(table, 9999999))
    assert {c.profile for c in result.selected.values()} == {"NINT2"}
    assert result.report["objective"] == result.report["relaxation_lower_bound"] == 0


def test_cli_quantizes_scheme_into_mfe(tmp_path):
    from mfq.formats.header import FileHeader
    from mfq.formats.io import open_mmap, save
    from mfq.formats.mfe import MfeTensor

    source = tmp_path / "source.mfq"
    scheme = tmp_path / "alphaq.json"
    target = tmp_path / "quantized.mfq"
    name = "model.block.0.mlp.experts.down.weight"
    save(
        source,
        FileHeader(model_arch="test", num_tensors=1),
        {name: np.random.default_rng(4).normal(size=(2, 8, 32)).astype(np.float32)},
    )
    assert (
        cli.main(
            [
                "calibrate",
                "alphaq",
                "--model",
                str(source),
                "--output",
                str(scheme),
                "--target-bpw",
                "5.0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "quantize",
                str(source),
                str(target),
                "--scheme",
                str(scheme),
                "--backend",
                "cpu",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    expected = load_scheme(scheme).expert_selections[name].precisions
    with open_mmap(target) as store:
        value = store[name]
        assert isinstance(value, MfeTensor)
        assert value.shape == (2, 8, 32)
        actual = {}
        for pool in value.pools:
            for local, expert in enumerate(pool.expert_ids):
                q = pool.tensor.row_q_bits[local * 8 : (local + 1) * 8]
                actual[int(expert)] = int(q[0])
                assert (q == q[0]).all()
        assert actual == {e: p.nint_spec.bits for e, p in enumerate(expected)}


def test_public_facade_and_cli_validation(tmp_path):
    from mfq.calibration import allocate_alphaq as public

    assert public is allocate_alphaq
    with pytest.raises(ValueError, match="must differ"):
        cli.main(
            [
                "calibrate",
                "alphaq",
                "--model",
                "unused",
                "--output",
                str(tmp_path / "same.json"),
                "--statistics",
                str(tmp_path / "same.json"),
                "--target-bpw",
                "3",
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        cli.main(
            [
                "calibrate",
                "alphaq",
                "--model",
                "unused",
                "--output",
                str(tmp_path / "ok.json"),
                "--target-bpw",
                "nan",
            ]
        )


def test_invalid_solver_contract_rejected_before_collecting_weights(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("statistics ran for an unsupported solver contract")

    monkeypatch.setattr("mfq.calibration.alphaq_source.collect_alphaq", forbidden)
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {"format": "mfq.ew-budget.v1", "model_weight_count": 100, "total": {"target_bpw": 3.0}}
        )
    )
    with pytest.raises(ValueError, match="--solver exact"):
        cli.main(
            [
                "calibrate",
                "alphaq",
                "--model",
                "unused",
                "--output",
                str(tmp_path / "out.json"),
                "--budget",
                str(path),
            ]
        )
