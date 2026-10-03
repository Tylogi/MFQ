"""Whole-model scope, router counts and serialized file-budget checks."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from mfq.calibration.alphaq_model import full_file_budget, imatrix_router_multipliers, model_scope
from mfq.formats.assets import ASSET_DTYPE, RuntimeAsset
from mfq.formats.header import FileHeader
from mfq.formats.shards import write_blob_record_shards
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.tools.quantize_hf_to_mfq import BlobRecord, TensorPlan


def fixture():
    plans, entries = [], {}
    for layer, counts, tokens in [(0, [8, 2], 10), (1, [5, 5, 0, 10], 20)]:
        for projection in ('gate', 'up', 'down'):
            name = f'model.block.{layer}.mlp.experts.{projection}.weight'
            plans.append(TensorPlan(name, 'weights', (len(counts), 2, 640), 'BF16', 'MFE'))
            entries[name] = ImportanceEntry(np.ones((len(counts), 640)), np.array(counts))
        entries[f'router.{layer}'] = ImportanceEntry(np.ones((1, 640)), np.array([tokens]))
    for name, shape, dtype, with_imatrix in [
        ('model.block.0.mlp.router.weight', (2, 640), 'F32', True),
        ('model.block.0.attn.output.weight', (2, 640), 'BF16', True),
        ('model.output.weight', (2, 640), 'F16', False),
        ('model.block.0.position_embedding.key.weight', (2, 640), 'BF16', True),
        ('model.block.0.position_embedding.hash', (3,), 'I64', False),
        ('model.norm.weight', (640,), 'F32', False),
    ]:
        plans.append(TensorPlan(name, 'weights', shape, dtype, dtype))
        if with_imatrix:
            entries[name] = ImportanceEntry(np.ones((1, shape[-1])), np.array([10]))
    return plans, ImportanceMatrix(Path('test.imatrix'), entries, (), 0, 0, False)


def test_scope_includes_small_router_and_excludes_ple_and_integer_parameters():
    plans, imatrix = fixture()
    scope = model_scope(plans, imatrix)
    assert len(scope.experts) == 6
    assert {p.name for p in scope.dense} == {
        'model.block.0.mlp.router.weight', 'model.block.0.attn.output.weight'}
    assert [p.name for p in scope.native_dense] == ['model.output.weight']
    assert len(scope.ple) == 2
    assert scope.model_weight_count == 18*2*640 + 4*2*640 + 640


@pytest.mark.parametrize('square_root', [False, True])
def test_router_uses_actual_tokens_and_one_global_mean(square_root):
    plans, imatrix = fixture()
    scope = model_scope(plans, imatrix)
    factors, exposure = imatrix_router_multipliers(scope, imatrix, {0: 'router.0', 1: 'router.1'}, 1,
                                                   square_root=square_root)
    raw = np.array([.8, .2, .25, .25, 0., .5])
    if square_root:
        raw = np.sqrt(raw)
    expected = raw/raw.mean()
    for key, value in factors.items():
        assert value == pytest.approx(expected[(0 if key.layer == 0 else 2) + key.expert_id])
        assert exposure[key.tensor] == (1/2 if key.layer == 0 else 1/4)


@pytest.mark.parametrize('failure', ['missing', 'coverage', 'disagreement', 'nan', 'fractional', 'extra_layer', 'bool_topk'])
def test_invalid_imatrix_routing_cannot_silently_change_importance(failure):
    plans, imatrix = fixture()
    scope = model_scope(plans, imatrix)
    router = {0: 'router.0', 1: 'router.1'}
    name = scope.experts[0].name
    if failure == 'missing':
        router.pop(0)
    elif failure == 'extra_layer':
        router[2] = 'router.1'
    elif failure in {'coverage', 'disagreement', 'nan', 'fractional'}:
        values = {'coverage': [7, 2], 'disagreement': [7, 3], 'nan': [np.nan, 2], 'fractional': [7.5, 2.5]}
        imatrix.entries[name] = replace(imatrix.entries[name], counts=np.array(values[failure]))
    with pytest.raises(ValueError):
        imatrix_router_multipliers(scope, imatrix, router, True if failure == 'bool_topk' else 1)


def test_scope_rejects_missing_expert_and_mismatched_matrix_width():
    plans, imatrix = fixture()
    first = plans[0].name
    original = imatrix.entries.pop(first)
    with pytest.raises(ValueError, match='missing expert'):
        model_scope(plans, imatrix)
    imatrix.entries[first] = replace(original, values=np.ones((2, 32)))
    with pytest.raises(ValueError, match='shape differs'):
        model_scope(plans, imatrix)


@pytest.mark.parametrize('joint', [False, True])
def test_whole_file_budget_matches_written_container_and_native_override(tmp_path, joint):
    plans, imatrix = fixture()
    # An old recipe requested quantization, but absent imatrix must force native.
    plans = [replace(p, target_dtype='NINT4') if p.name == 'model.output.weight' else p for p in plans]
    scope = model_scope(plans, imatrix)
    assets = [RuntimeAsset('__mfq__/config.json', 'application/json', b'{"x":3}')]
    header = FileHeader(version=2, model_arch='test', extra={'policy': 'alphaq', 'unicode': '矩阵'})
    bpw = Decimal('16.000123456789')
    budget = full_file_budget(scope, header, assets, target_bpw=bpw, joint_dense=joint)
    assert budget.maximum_file_bytes == int(bpw*scope.model_weight_count/8)
    assert budget.native_overrides == {'model.output.weight': 'F16'}
    records, variable_bytes, short_tags = [], 0, 0
    payload = tmp_path/'payload.bin'
    payload.write_bytes(bytes(10000))
    records.append(BlobRecord(assets[0].name, ASSET_DTYPE, len(assets[0].data), payload))
    for item in plans:
        if item in scope.experts:
            dtype, size = 'MFE', 113
            variable_bytes += size
        elif joint and item in scope.dense:
            # A three-character native tag is smaller than the reserved four.
            dtype, size = 'F32', 20+4*np.prod(item.shape)
            variable_bytes += size
            short_tags += 1
        else:
            dtype = item.source_dtype
            size = 4+8*len(item.shape)+np.prod(item.shape)*{'BF16':2, 'F16':2, 'F32':4, 'I64':8}[dtype]
        records.append(BlobRecord(item.name, dtype, int(size), payload))
    output = tmp_path/'model.mfq'
    write_blob_record_shards(output, header, records)
    assert output.stat().st_size == budget.fixed_file_bytes + variable_bytes - short_tags


@pytest.mark.parametrize('bpw', ['NaN', 'Infinity', '-1', '0', 'bad'])
def test_invalid_bpw_is_rejected(bpw):
    plans, imatrix = fixture()
    with pytest.raises(ValueError, match='BPW'):
        full_file_budget(model_scope(plans, imatrix), FileHeader(), [], target_bpw=bpw)
