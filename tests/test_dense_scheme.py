"""Calibration descriptors survive planning and normal HF conversion."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mfq.calibration.artifact import (
    CalibrationScheme, ExpertPrecision, TensorSelection, load_scheme, save_scheme,
)
from mfq.formats.io import load_mmap, pack_tensor_payload
from mfq.formats.nint import NintSpec
from mfq.quantize.expert_nint import quantize_flat_cohort
from mfq.quantize.nvq_jsc import NvqJscConfig, initial_jsc_tables
from mfq.tools import quantize_hf_to_mfq as hf


SOURCE = "model.language_model.layers.0.mlp.down_proj.weight"
CANONICAL = "model.block.0.mlp.down.weight"
PROFILES = ("NVQ1-S", "NVQ1-L", "NVQ2J", "NVQ2J-L", "NVQ2J-XL",
            "NVQ3J", "NVQ3J-512", "NVQ3J-L")


def scheme_for(name, precision, shape=(8, 640)):
    selection = TensorSelection(name, "dense", precision.nint_spec, *shape,
                                1, 0.0, 0.0, precision=precision)
    return CalibrationScheme(None, "ALPHAQ", 1, {name: selection}, {}, {})


def model_at(root):
    root.mkdir()
    values = torch.from_numpy(np.random.default_rng(43).normal(size=(8, 640))).bfloat16()
    save_file({SOURCE: values}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    return values


@pytest.mark.parametrize("name", [SOURCE, CANONICAL])
def test_scheme_alias_preserves_descriptor_and_native_override(tmp_path, name):
    root = tmp_path / "model"
    model_at(root)
    precision = ExpertPrecision("NVQ3J-L", artifact="tables.npz", options=(("banks", 2),))
    scheme = scheme_for(name, precision)
    plan = hf._plan(root, True, None, "F16", scheme)
    assert len(plan) == 1
    assert plan[0].name == CANONICAL
    assert plan[0].target_dtype == "NVQ3J-L"
    assert plan[0].target_precision == precision
    native = hf._apply_tensor_precision_overrides(plan, {CANONICAL: "BF16"})[0]
    assert native.target_dtype == "BF16"
    assert native.target_precision is None
    assert native.target_spec is None
    assert hf._plan_blob_nbytes(native, NintSpec()) == 20 + 8 * 640 * 2


def test_duplicate_scheme_alias_is_rejected(tmp_path):
    root = tmp_path / "model"
    model_at(root)
    scheme = scheme_for(SOURCE, ExpertPrecision("NVQ1-S"))
    scheme.selections[CANONICAL] = replace(scheme.selections[SOURCE], name=CANONICAL)
    with pytest.raises(ValueError, match="duplicate source/canonical"):
        hf._plan(root, True, None, "F16", scheme)


@pytest.mark.parametrize("family", PROFILES)
def test_dense_scheme_stream_matches_canonical_payload(tmp_path, family):
    values = torch.from_numpy(np.random.default_rng(17).normal(size=(16, 640))).bfloat16()
    importance = np.linspace(0.3, 2.1, 640, dtype=np.float32)
    artifact = None
    artifact_name = None
    options = (("assignment_refine_steps", 1), ("search_steps", 7), ("group_chunk", 1024))
    if "J" in family:
        artifact = initial_jsc_tables(NvqJscConfig(banks=2, spec=hf._NVQ_SPECS[family]))
        artifact_name = "selected.npz"
        np.savez(tmp_path / artifact_name, scale_lut=artifact.scale_lut,
                 bank_for_state=artifact.bank_for_state, codebooks=artifact.codebooks)
    precision = ExpertPrecision(family, artifact=artifact_name, options=options)
    path = tmp_path / "scheme.json"
    save_scheme(path, scheme_for(CANONICAL, precision, (16, 640)))
    assert load_scheme(path).selections[CANONICAL].descriptor == precision
    plan = hf.TensorPlan(CANONICAL, "", (16, 640), "BF16", family,
                         target_precision=precision)
    expected = pack_tensor_payload(quantize_flat_cohort(
        values.float().numpy(), precision, artifact=artifact, importance=importance, device="cpu"
    ))[1]
    calls = []

    def importance_rows(start, end):
        calls.append((start, end))
        return importance

    output = tmp_path / "dense.bin"
    size = hf._write_flat_family_axis0_blob(
        values, (16, 640), precision, output, 8, "cpu", "cpu", tmp_path,
        importance_rows=importance_rows,
    )
    assert calls == [(0, 8), (8, 16)]
    assert output.read_bytes() == expected
    assert size == hf._plan_blob_nbytes(plan, NintSpec(), tmp_path)


def test_normal_converter_uses_dense_scheme_artifact(tmp_path):
    root = tmp_path / "model"
    values = model_at(root)
    family = "NVQ3J-L"
    artifact = initial_jsc_tables(NvqJscConfig(banks=2, spec=hf._NVQ_SPECS[family]))
    np.savez(tmp_path / "selected.npz", scale_lut=artifact.scale_lut,
             bank_for_state=artifact.bank_for_state, codebooks=artifact.codebooks)
    precision = ExpertPrecision(family, artifact="selected.npz", options=(
        ("assignment_refine_steps", 1), ("search_steps", 7), ("group_chunk", 1024),
    ))
    scheme_path = tmp_path / "scheme.json"
    save_scheme(scheme_path, scheme_for(CANONICAL, precision))
    output = tmp_path / "model.mfq"
    args = hf.build_parser().parse_args([
        "--input", str(root), "--output", str(output), "--calibration-scheme", str(scheme_path),
        "--quant-backend", "cpu", "--device", "cpu", "--row-chunk", "4",
    ])
    hf.convert(args)
    expected = pack_tensor_payload(quantize_flat_cohort(
        values.float().numpy(), precision, artifact=artifact, device="cpu"
    ))[1]
    _, store = load_mmap(output)
    try:
        assert pack_tensor_payload(store[CANONICAL])[1] == expected
    finally:
        store.close()


def test_uniform_expert_selection_uses_generic_descriptor():
    precision = ExpertPrecision("NVQ2J-XL", artifact="tables.npz")
    scheme = scheme_for("experts", precision, (16, 640))
    assert hf._glm_expert_precisions("experts", (2, 8, 640), scheme) == (precision, precision)
