"""Explicit weighted NINT8 survives canonical, streamed and normal HF paths."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mfq.calibration.artifact import CalibrationScheme, ExpertPrecision, TensorSelection, save_scheme
from mfq.calibration.evaluator import NINT_EXPERT_PROFILES
from mfq.formats.io import load_mmap, pack_tensor_payload, unpack_mfe
from mfq.quantize.expert_nint import quantize_flat_cohort
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.quantize.nint_quant import quantize
from mfq.tools import quantize_hf_to_mfq as hf


SPEC = NINT_EXPERT_PROFILES['NINT8']
LEGACY = ExpertPrecision('NINT', nint_spec=SPEC)
WEIGHTED = replace(LEGACY, options=(('imatrix_weighted', True),))
SOURCE = 'model.language_model.layers.0.mlp.down_proj.weight'
NAME = 'model.block.0.mlp.down.weight'


def data():
    weight = torch.from_numpy(np.random.default_rng(31).normal(size=(8, 640))).bfloat16()
    importance = np.geomspace(.001, 1000., 640).astype(np.float32)
    return weight, importance


def imatrix_at(path, name, importance):
    return ImportanceMatrix(path, {name: ImportanceEntry(importance[None], np.array([16]))},
                            ('fixed-test',), 1, 640, False)


def test_weighted_nint8_canonical_bytes_and_legacy_policy():
    weight, importance = data()
    rows = weight.float().numpy()
    weighted = pack_tensor_payload(quantize(rows, SPEC, axis=0, importance=importance))[1]
    plain = pack_tensor_payload(quantize(rows, SPEC, axis=0))[1]
    assert weighted != plain
    assert pack_tensor_payload(quantize_flat_cohort(
        rows, WEIGHTED, importance=importance, device='cpu'))[1] == weighted
    assert pack_tensor_payload(quantize_flat_cohort(
        rows, LEGACY, importance=importance, device='cpu'))[1] == plain
    with pytest.raises(ValueError, match='requires importance'):
        quantize_flat_cohort(rows, WEIGHTED, device='cpu')


@pytest.mark.parametrize('mixed', [False, True])
def test_streamed_mfe_keeps_weighted_and_plain_pools_and_exact_cost(tmp_path, mixed):
    weight, importance = data()
    source = torch.stack((weight, weight))
    precisions = (WEIGHTED, LEGACY) if mixed else (WEIGHTED, WEIGHTED)
    output = tmp_path/'experts.bin'
    nbytes = hf._write_mixed_moe_axis0_blob(
        source, (2, 8, 640), (2, 8, 640), precisions, output, 8, 'cpu', 'cpu', None,
        importance=importance)
    assert nbytes == hf._mixed_moe_blob_nbytes((2, 8, 640), precisions, None)
    tensor = unpack_mfe(output.read_bytes())
    assert len(tensor.pools) == (2 if mixed else 1)
    for pool in tensor.pools:
        ids = list(pool.expert_ids)
        expected = quantize(source[ids].reshape(-1, 640).float().numpy(), SPEC, axis=0,
                            importance=importance if precisions[ids[0]] == WEIGHTED else None)
        assert pack_tensor_payload(pool.tensor)[1] == pack_tensor_payload(expected)[1]


@pytest.mark.parametrize('precision', [LEGACY, WEIGHTED])
def test_normal_dense_converter_preserves_explicit_policy(tmp_path, monkeypatch, precision):
    weight, importance = data()
    root = tmp_path/'model'
    root.mkdir()
    save_file({SOURCE: weight}, root/'model.safetensors')
    (root/'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    selection = TensorSelection(NAME, 'dense', SPEC, 8, 640, 1, 0., 0., precision)
    scheme_path = tmp_path/'scheme.json'
    save_scheme(scheme_path, CalibrationScheme(None, 'joint', 1, {NAME: selection}, {}, {}))
    imatrix_path = tmp_path/'imatrix'
    imatrix_path.write_bytes(b'fixture')
    matrix = imatrix_at(imatrix_path, SOURCE, importance)
    monkeypatch.setattr(hf, 'load_importance_matrix', lambda _: matrix)
    output = tmp_path/'model.mfq'
    args = hf.build_parser().parse_args([
        '--input', str(root), '--output', str(output), '--calibration-scheme', str(scheme_path),
        '--imatrix', str(imatrix_path), '--quant-backend', 'cpu', '--device', 'cpu', '--row-chunk', '4'])
    hf.convert(args)
    _, store = load_mmap(output)
    try:
        expected = quantize(weight.float().numpy(), SPEC, axis=0,
                            importance=importance if precision == WEIGHTED else None)
        assert pack_tensor_payload(store[NAME])[1] == pack_tensor_payload(expected)[1]
    finally:
        store.close()
    if precision == WEIGHTED:
        args.imatrix = ''
        args.output = str(tmp_path/'missing.mfq')
        with pytest.raises(ValueError, match='requires --imatrix'):
            hf.convert(args)


def test_explicit_weighted_optional_embedding_requires_actual_binding(tmp_path):
    plan = hf.TensorPlan('token_embd.weight', '', (8, 640), 'BF16', 'NINT8',
                         target_spec=SPEC, target_precision=WEIGHTED)
    empty = ImportanceMatrix(tmp_path/'empty', {}, (), 0, 0, False)
    with pytest.raises(ValueError, match='missing'):
        hf._bind_hf_imatrix(empty, [plan])


@pytest.mark.parametrize('value', ['true', 1, .5])
def test_weighted_flag_requires_boolean(value):
    with pytest.raises(ValueError, match='boolean'):
        replace(LEGACY, options=(('imatrix_weighted', value),))
