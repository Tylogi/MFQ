import json
import asyncio
import sys
from decimal import Decimal, ROUND_CEILING
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from safetensors.torch import save_file

from mfq.formats.io import open_mmap
from mfq.quantize.workbench import ModelSource, RecipeValidationError, prepare_recipe, validate_recipe, generate_recipe, fit_model, load_recipe, matrix_statistics, BatchGate, source_summary
from tests.test_alphaq_workflow import fixture
from mfq.server.protocol.models import CreateJobRequest
from mfq.server.protocol.quantization import QuantizationDirectories
from mfq.server.services.jobs import JobManager
from mfq.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore


def valid_target(model, candidates, preferred=12):
    target = preferred
    source = ModelSource(model)
    try:
        for _ in range(8):
            try:
                validate_recipe(model, target, candidates)
                return target
            except RecipeValidationError:
                pass
            scope, _, _, bounds = prepare_recipe(source, tuple(candidates), Decimal(str(target)), check_target=False)
            limit = min(bounds['maximum_file_bytes'], bounds['source_file_bytes'], scope.model_weight_count * 4)
            assert bounds['minimum_file_bytes'] <= limit
            size = max(bounds['minimum_file_bytes'], min(limit, int(Decimal(str(preferred)) * scope.model_weight_count / 8)))
            updated = float((Decimal(size) * 8 / scope.model_weight_count).quantize(Decimal('0.000000000001'), rounding=ROUND_CEILING))
            if updated == target:
                validate_recipe(model, target, candidates)
                return target
            target = updated
        pytest.fail('candidate budget did not stabilize')
    finally:
        source.close()


@pytest.mark.parametrize('backend', ['cpu', pytest.param('metal', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_data_free_recipe_and_complete_model(tmp_path, backend):
    root, _, count = fixture(tmp_path)
    recipe, output = tmp_path / 'recipe.json', tmp_path / 'output.mfq'
    result = generate_recipe(root, recipe, 12, ['NINT4', 'NINT8'], backend=backend)
    assert result['estimated_bpw'] <= 12
    scheme = load_recipe(recipe)
    assert scheme.target_profile == 'DF-V1-AlphaQ'
    assert scheme.expert_selections
    fitted = fit_model(root, recipe, output, backend=backend, row_chunk=8)
    assert fitted['total_bytes'] == output.stat().st_size <= count * 12 // 8
    store = open_mmap(output)
    try:
        assert len(store.records) >= 6
        assert all(store.records[f'model.block.0.mlp.experts.{p}.weight'].dtype == 'MFE' for p in ('gate', 'up', 'down'))
        assert 'model.output.weight' in store.records
    finally:
        store.close()
    with pytest.raises((ValueError, RuntimeError), match='precision|quantized'):
        ModelSource(output)
    with pytest.raises(FileExistsError):
        fit_model(root, recipe, output, backend='cpu')


def test_recipe_strict_validation_and_source_identity(tmp_path):
    root, _, _ = fixture(tmp_path)
    recipe = tmp_path / 'recipe.json'
    generate_recipe(root, recipe, 12, ['NINT4', 'NINT8'], backend='cpu')
    config = root / 'config.json'
    config.write_text(config.read_text() + '\n')
    with pytest.raises(ValueError, match='different source'):
        fit_model(root, recipe, tmp_path / 'bad.mfq', backend='cpu')
    recipe.write_text('{"format":"a","format":"b"}')
    with pytest.raises(ValueError, match='duplicate'):
        load_recipe(recipe)


@pytest.mark.parametrize('target,candidates,code', [
    (4, [], 'quantization_empty_candidates'),
    (12, ['NINT4'], 'quantization_target_above_candidates'),
    (5, ['NINT8'], 'quantization_target_below_candidates'),
    (24, ['NINT4', 'NINT8'], 'quantization_target_above_source'),
    (4, ['NINT4'], 'quantization_target_below_candidates'),
    (0, ['NINT4'], 'quantization_invalid_target'),
    (float('nan'), ['NINT4'], 'quantization_invalid_target'),
    (None, ['NINT4'], 'quantization_invalid_target'),
    (33, ['NINT4'], 'quantization_invalid_target'),
    (4, None, 'invalid_candidates'),
    (4, [['NINT4']], 'invalid_candidates'),
])
def test_recipe_boundary_errors_precede_weight_reads_and_leave_no_output(tmp_path, monkeypatch, target, candidates, code):
    root, _, _ = fixture(tmp_path)
    def forbidden(*args, **kwargs): pytest.fail('invalid recipe must not read weights or collect statistics')
    monkeypatch.setattr(ModelSource, 'rows', forbidden)
    monkeypatch.setattr('mfq.quantize.workbench.matrix_statistics', forbidden)
    output = tmp_path / 'invalid.json'
    with pytest.raises(RecipeValidationError) as error:
        generate_recipe(root, output, target, candidates, backend='cpu')
    assert error.value.code == code
    assert not output.exists()
    if code in {'quantization_target_above_candidates', 'quantization_target_below_candidates'}:
        assert error.value.details['minimum_bpw'] > 4
        assert error.value.details['maximum_bpw'] >= error.value.details['minimum_bpw']


def test_recipe_preflight_uses_metadata_and_accepts_single_candidate_endpoint(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    def forbidden(*args, **kwargs): pytest.fail('preflight must not read weights')
    monkeypatch.setattr(ModelSource, 'rows', forbidden)
    bounds = validate_recipe(root, 12, ['NINT4', 'NINT8'])
    assert bounds['minimum_bpw'] < 12 < bounds['maximum_bpw']
    source = ModelSource(root)
    try:
        a = prepare_recipe(source, ('NINT4',), Decimal(4), check_target=False)[3]
        b = prepare_recipe(source, ('NINT4',), Decimal('9.123456789123'), check_target=False)[3]
        assert a['minimum_file_bytes'] == b['minimum_file_bytes']
        assert a['maximum_file_bytes'] == b['maximum_file_bytes']
        count = source.parameters()
    finally:
        source.close()
    target = valid_target(root, ['NINT4'])
    endpoint = validate_recipe(root, target, ['NINT4'])
    assert endpoint['minimum_bpw'] <= target <= endpoint['maximum_bpw'] + 8 / count


@pytest.mark.parametrize('precision', ['fp8', 'mxfp8', 'mxfp4'])
def test_recipe_rejects_targets_above_native_source_precision(tmp_path, monkeypatch, precision):
    root = native_source_fixture(tmp_path, precision)
    monkeypatch.setattr(ModelSource, 'rows', lambda *args, **kwargs: pytest.fail('source precision check must be metadata-only'))
    with pytest.raises(RecipeValidationError) as error:
        validate_recipe(root, 12, ['NINT4', 'NINT8'])
    assert error.value.code == 'quantization_target_above_source'
    assert error.value.details['source_bpw'] < 12


def test_recipe_reports_when_no_target_can_fit_the_candidate_set(tmp_path):
    root = native_source_fixture(tmp_path, 'mxfp4')
    with pytest.raises(RecipeValidationError) as error:
        validate_recipe(root, 5, ['NINT4', 'NINT8'])
    assert error.value.code == 'quantization_no_feasible_budget'
    assert error.value.details['minimum_bpw'] > error.value.details['source_bpw']


def test_recipe_api_returns_explicit_errors_without_registering_jobs(tmp_path):
    import httpx
    from mfq.server.api import create_app
    from mfq.server.services.service import ServerService
    from tests.test_server_service import FakeBackend
    async def run():
        root, _, _ = fixture(tmp_path)
        catalog = ModelCatalog([root])
        tools = ToolJobHandlers(catalog, ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        store = SessionStore(tmp_path / 'jobs.sqlite3')
        manager = JobManager(store, tools.handlers())
        service = ServerService(store, FakeBackend(), jobs=manager, catalog=catalog, tool_handlers=tools)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
                for target, candidates, code in [(4, [], 'quantization_empty_candidates'),
                    (12, ['NINT4'], 'quantization_target_above_candidates'),
                    (5, ['NINT8'], 'quantization_target_below_candidates'),
                    (24, ['NINT4', 'NINT8'], 'quantization_target_above_source')]:
                    response = await client.post('/api/v1/jobs', json={'kind': 'quantization.recipe', 'payload': {
                        'input': str(root), 'output': str(tmp_path / 'invalid.json'), 'target_bpw': target, 'candidates': candidates}})
                    assert response.status_code == 422, response.text
                    assert response.json()['error']['code'] == code
                    if candidates:
                        assert response.json()['error']['details']['target_bpw'] == target
                assert (await client.get('/api/v1/jobs')).json()['data'] == []
        finally:
            await service.aclose()
        assert not (tmp_path / 'invalid.json').exists()
    asyncio.run(run())


def test_rejects_nvfp4_hf_source(tmp_path):
    root, _, _ = fixture(tmp_path)
    config = root / 'config.json'
    value = json.loads(config.read_text())
    value['quantization_config'] = {'quant_method': 'nvfp4'}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='NVFP4'):
        ModelSource(root)


def native_source_fixture(tmp_path, precision, varying=False):
    from tests.test_hf_source_store import _write_raw_safetensors
    root = tmp_path / precision; root.mkdir()
    config = {'model_type': 'deepseek_v4', 'num_hidden_layers': 1, 'n_routed_experts': 2,
        'num_experts_per_tok': 1, 'hidden_size': 128, 'moe_intermediate_size': 128,
        'quantization_config': {'quant_method': precision, 'weight_block_size': [128, 128]}}
    (root / 'config.json').write_text(json.dumps(config))
    tensors = {}
    for expert in range(2):
        for part in ('w1', 'w2', 'w3'):
            name = f'layers.0.ffn.experts.{expert}.{part}'
            if precision == 'mxfp4':
                data = np.full((128, 64), 0x22 + expert, dtype=np.uint8)
                if varying: data[::2] ^= 0x11
                scale = np.full((128, 4), 127, dtype=np.uint8)
                tensors[name + '.weight'] = ('I8', data.shape, data.tobytes())
                tensors[name + '.scale'] = ('F8_E8M0', scale.shape, scale.tobytes())
            else:
                data = np.full((128, 128), 0x38 + expert, dtype=np.uint8)
                if varying: data[::2] += 2
                scale = np.full((1, 1), 128 if precision == 'mxfp8' else 2, dtype=np.uint8 if precision == 'mxfp8' else np.float32)
                tensors[name + '.weight'] = ('F8_E4M3', data.shape, data.tobytes())
                tensors[name + '.scale'] = ('F8_E8M0' if precision == 'mxfp8' else 'F32', scale.shape, scale.tobytes())
    dense = np.linspace(-1, 1, 32 * 128, dtype=np.float32).reshape(32, 128)
    if varying: dense.fill(0)
    tensors['head.weight'] = ('F32', dense.shape, dense.tobytes())
    _write_raw_safetensors(root / 'model.safetensors', tensors)
    return root


@pytest.mark.parametrize('precision', ['fp8', 'mxfp8', 'mxfp4'])
def test_native_low_precision_source_allocates_and_fits_complete_model(tmp_path, precision):
    root = native_source_fixture(tmp_path, precision)
    summary = source_summary(root)
    assert summary['parameters'] == 6 * 128 * 128 + 32 * 128
    assert precision.upper() in summary['source_precisions']
    source = ModelSource(root)
    try:
        bank = next(item for item in source.plans if item.name.endswith('.experts.gate.weight'))
        assert bank.shape == (2, 128, 128)
        with source.rows(bank, BatchGate()) as reader:
            row = reader.read_rows(0, 1, device='cpu')
        torch.testing.assert_close(row, torch.full((1, 128), 1.0 if precision == 'mxfp4' else 2.0))
    finally:
        source.close()
    recipe, output = tmp_path / 'recipe.json', tmp_path / 'model.mfq'
    candidates = ['NVQ1-S'] if precision == 'mxfp4' else ['NINT4', 'NINT8']
    target = valid_target(root, candidates)
    generate_recipe(root, recipe, target, candidates, backend='cpu')
    result = fit_model(root, recipe, output, backend='cpu', row_chunk=128)
    assert result['actual_bpw'] <= target
    store = open_mmap(output)
    try:
        assert all(store.records[f'model.block.0.mlp.experts.{p}.weight'].dtype == 'MFE' for p in ('gate', 'up', 'down'))
    finally:
        store.close()


@pytest.mark.parametrize('precision,profile', [
    *(('mxfp4', f'MXFP4-SQ-{bits}') for bits in ('1', '2', '3', 'F')),
    *(('mxfp8', f'MXFP8-SQ-{bits}') for bits in range(1, 9)),
    *(('fp8', f'FP8-SQ-{bits}') for bits in range(1, 9)),
])
def test_sq_candidates_fit_native_codes_without_dequantization(tmp_path, monkeypatch, precision, profile):
    from mfq.formats.io import unpack_tensor_payload
    from mfq.tools import quantize_hf_to_mfq as q
    exact = profile.endswith(('-F', '-8'))
    root = native_source_fixture(tmp_path, precision, varying=exact)
    summary = source_summary(root)
    assert profile in summary['eligible_candidates']
    other = 'FP8-SQ-4' if precision != 'fp8' else 'MXFP8-SQ-4'
    assert other not in summary['eligible_candidates']
    recipe, output = tmp_path / 'sq.json', tmp_path / 'sq.mfq'
    candidates = [profile, 'NINT4'] if exact else [profile]
    target = valid_target(root, candidates)
    generate_recipe(root, recipe, target, candidates, backend='cpu')
    scheme = load_recipe(recipe)
    expected_bits = 4 if profile.endswith('-F') else int(profile[-1])
    assert all(entry.descriptor.option('q') == expected_bits for bank in scheme.expert_selections.values() for entry in bank.selections)
    def forbidden(*args, **kwargs):
        raise AssertionError('native SQ fitting must never dequantize source weights')
    monkeypatch.setattr(q._ScaledFp8TensorSlice, 'read_rows', forbidden)
    monkeypatch.setattr(q._Mxfp4TensorSlice, 'read_rows', forbidden)
    result = fit_model(root, recipe, output, backend='cpu')
    assert result['actual_bpw'] <= target
    store = open_mmap(output)
    try:
        for projection in ('gate', 'up', 'down'):
            name = f'model.block.0.mlp.experts.{projection}.weight'
            bank = unpack_tensor_payload('MFE', store.read_blob(name))
            for pool in bank.pools:
                assert set(pool.tensor.row_q_bits) == {expected_bits}
    finally:
        store.close()


@pytest.mark.parametrize('precision,profile', [
    ('mxfp4', 'MXFP4-SQ-2'),
    *(('mxfp8', f'MXFP8-SQ-{bits}') for bits in (1, 2, 3, 5)),
    *(('fp8', f'FP8-SQ-{bits}') for bits in (1, 2, 3, 5)),
])
def test_dense_sq_candidates_preserve_source_contract(tmp_path, monkeypatch, precision, profile):
    from mfq.tools import quantize_hf_to_mfq as q
    from tests.test_hf_source_store import _write_raw_safetensors
    root = native_source_fixture(tmp_path, precision)
    name = 'head.weight'
    if precision == 'mxfp4':
        tensors = {name: ('I8', (128, 64), bytes([0x22]) * (128 * 64)), 'head.scale': ('F8_E8M0', (128, 4), bytes([127]) * 512)}
    else:
        tensors = {name: ('F8_E4M3', (128, 128), bytes([0x38]) * (128 * 128)), 'head.scale': (
            'F8_E8M0' if precision == 'mxfp8' else 'F32', (1, 1), bytes([127]) if precision == 'mxfp8' else np.array([2.], dtype=np.float32).tobytes())}
    _write_raw_safetensors(root / 'model.safetensors', tensors)
    recipe = tmp_path / 'dense-sq.json'
    generate_recipe(root, recipe, valid_target(root, [profile]), [profile], backend='cpu')
    assert load_recipe(recipe).selections['model.output.weight'].descriptor.option('q') == int(profile[-1])
    def forbidden(*args, **kwargs): raise AssertionError('SQ must read packed source storage')
    monkeypatch.setattr(q._ScaledFp8TensorSlice, 'read_rows', forbidden)
    monkeypatch.setattr(q._Mxfp4TensorSlice, 'read_rows', forbidden)
    output = tmp_path / 'dense-sq.mfq'
    fit_model(root, recipe, output, backend='cpu')
    store = open_mmap(output)
    try:
        assert store.records['model.output.weight'].dtype == ('FP8-128SQ' if precision == 'fp8' else precision.upper() + '-SQ')
    finally:
        store.close()


@pytest.mark.parametrize('profile', ['MXFP4-SQ-F', *(
    f'{family}-{bits}' for family in ('MXFP8-SQ', 'FP8-SQ') for bits in (1, 2, 3))])
def test_rejects_sq_candidates_on_bf16_sources(tmp_path, profile):
    root, _, _ = fixture(tmp_path)
    assert not any('-SQ-' in name for name in source_summary(root)['eligible_candidates'])
    with pytest.raises(ValueError, match='matching native source'):
        generate_recipe(root, tmp_path / 'invalid.json', 12, ['NINT4', profile], backend='cpu')


def test_mixed_native_source_limits_sq_per_expert(tmp_path):
    from mfq.tools import quantize_hf_to_mfq as q
    from mfq.calibration.alphaq import AlphaQTensorStatistics
    from mfq.quantize.workbench import workbench_expert_candidates
    from tests.test_hf_source_store import _write_raw_safetensors
    root = native_source_fixture(tmp_path, 'mxfp4')
    data = {}
    for expert in range(2):
        for projection in ('w1', 'w2', 'w3'):
            name = f'layers.0.ffn.experts.{expert}.{projection}'
            if expert == 0:
                data[name + '.weight'] = ('F32', (128, 128), np.ones((128, 128), dtype=np.float32).tobytes())
            else:
                data[name + '.weight'] = ('I8', (128, 64), bytes([0x22]) * 8192)
                data[name + '.scale'] = ('F8_E8M0', (128, 4), bytes([127]) * 512)
    _write_raw_safetensors(root / 'model.safetensors', data)
    source = ModelSource(root)
    try:
        experts = list(source.plans)
        statistics = [AlphaQTensorStatistics(item.name, 0, item.name.split('.')[-2], item.shape, (2., 2.), (1., 1.)) for item in experts]
        candidates = workbench_expert_candidates(source, statistics, experts, ['NINT4', 'MXFP4-SQ-2'])
        sq = [entry for entry in candidates.candidates if entry.profile == 'MXFP4-SQ-2']
        assert len(sq) == 3 and {entry.key.expert_id for entry in sq} == {1}
    finally:
        source.close()
    recipe = tmp_path / 'mixed.json'
    generate_recipe(root, recipe, valid_target(root, ['NINT4', 'MXFP4-SQ-2'], 8), ['NINT4', 'MXFP4-SQ-2'], backend='cpu')
    fit_model(root, recipe, tmp_path / 'mixed.mfq', backend='cpu')


@pytest.mark.parametrize('precision', ['fp8', 'mxfp8', 'mxfp4'])
@pytest.mark.parametrize('direct_io', [False, True])
def test_calibration_index_reads_native_values_and_logical_shape(tmp_path, monkeypatch, precision, direct_io):
    from mfq.calibration.qwen35 import HfSafetensorIndex
    root = native_source_fixture(tmp_path, precision)
    monkeypatch.setenv('MFQ_SAFETENSORS_DIRECT_IO', '1' if direct_io else '0')
    index = HfSafetensorIndex(root)
    name = 'layers.0.ffn.experts.0.w1.weight'
    assert index.shape(name) == (128, 128)
    value = index.tensor(name, row_start=7, row_end=9, dtype=torch.bfloat16)
    torch.testing.assert_close(value, torch.full((2, 128), 1. if precision == 'mxfp4' else 2., dtype=torch.bfloat16))


@pytest.mark.parametrize('block', [(128, 128), (32, 32), (1, 32)])
def test_native_mxfp8_mfq_source_preserves_and_transcodes_exact_storage(tmp_path, monkeypatch, block):
    from mfq.formats.io import save, unpack_tensor_payload
    from mfq.formats.header import FileHeader
    from mfq.formats.mx import MxTensor
    from mfq.formats.fp8_sq import decode_fp8_sq_codes, fp8_sq_scale_bytes
    from mfq.quantize.mfq_source import FullPrecisionMfqTensorSource
    values = np.full((128, 128), 0x38, dtype=np.uint8)
    scales = np.arange((128 // block[0]) * (128 // block[1]), dtype=np.uint8).reshape(128 // block[0], 128 // block[1]) % 4 + 125
    path = tmp_path / 'native.mfq'
    save(path, FileHeader(version=2, model_arch='qwen3', extra={'hf_config': {'model_type': 'qwen3'}}), {
        'model.output.weight': MxTensor('MXFP8', (128, 128), values, scales)})
    source = ModelSource(path)
    try:
        item = source.plans[0]
        with source.rows(item, BatchGate()) as reader:
            actual = reader.read_rows(7, 9, device='cpu')
        expanded = np.repeat(np.repeat(np.exp2(scales.astype(np.float32) - 127), block[0], axis=0), block[1], axis=1)
        torch.testing.assert_close(actual, torch.from_numpy(expanded[7:9]))
        copied = tmp_path / 'copied.blob'
        _, size = source.write_native(item, copied, 8, BatchGate())
        assert size == len(source.checkpoint.store.read_blob('model.output.weight'))
        assert copied.read_bytes() == source.checkpoint.store.read_blob('model.output.weight')
    finally:
        source.close()
    recipe = tmp_path / 'native-sq.json'
    generate_recipe(path, recipe, 7, ['MXFP8-SQ-5'], backend='cpu')
    def forbidden(*args, **kwargs): raise AssertionError('native SQ must not decode the original MFQ')
    monkeypatch.setattr(FullPrecisionMfqTensorSource, 'read_rows', forbidden)
    fit_model(path, recipe, tmp_path / 'sq.mfq', backend='cpu')


def test_native_fp8_mfq_uses_original_scale_and_sq_codes(tmp_path, monkeypatch):
    from mfq.formats.header import FileHeader
    from mfq.formats.shards import write_blob_record_shards
    from mfq.formats.fp8_sq import decode_fp8_sq_codes, fp8_sq_scale_bytes
    from mfq.formats.io import unpack_tensor_payload
    from mfq.tools import quantize_hf_to_mfq as q
    path = tmp_path / 'native-fp8.mfq'
    values = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), .5)
    record_names = ['model.output.weight', 'model.output.weight_scale']
    blobs = []
    for name, tensor, dtype in zip(record_names, [values, scale], ['F8_E4M3', 'F32'], strict=True):
        blob = tmp_path / (dtype + '.blob')
        rows = SimpleNamespace(read_rows=lambda start, end, *, device: tensor[start:end])
        size = q._write_float8_e4m3_axis0_blob(rows, tensor.shape, blob, 8) if dtype == 'F8_E4M3' else q._dense_blob_from_tensor(tensor, blob, dtype)
        blobs.append(q.BlobRecord(name, dtype, size, blob))
    write_blob_record_shards(path, FileHeader(version=2, model_arch='qwen3', extra={'hf_config': {'model_type': 'qwen3'}}), blobs)
    source = ModelSource(path)
    try:
        assert source.parameters() == values.numel()
        item = next(item for item in source.plans if item.name == record_names[0])
        assert source.native_dtype(item) == 'F32'
        with source.rows(item, BatchGate()) as reader:
            torch.testing.assert_close(reader.read_rows(0, 1, device='cpu'), torch.full((1, 128), .5))
    finally:
        source.close()
    generate_recipe(path, tmp_path / 'sq.json', 7, ['FP8-SQ-5'], backend='cpu')
    def forbidden(*args, **kwargs): raise AssertionError('FP8 SQ must use native codes, not decoded values')
    monkeypatch.setattr(q._ScaledFp8TensorSlice, 'read_rows', forbidden)
    fit_model(path, tmp_path / 'sq.json', tmp_path / 'sq.mfq', backend='cpu')


@pytest.mark.parametrize('precision,profile', [('MXFP4', 'MXFP4-SQ-2'), ('MXFP8', 'MXFP8-SQ-6')])
def test_native_mfq_expert_sq_never_materializes_float_weights(tmp_path, monkeypatch, precision, profile):
    from mfq.formats.io import save
    from mfq.formats.header import FileHeader
    from mfq.formats.mx import MxTensor
    from mfq.quantize.mfq_source import FullPrecisionMfqTensorSource
    tensors = {}
    for expert in range(2):
        for part in ('w1', 'w2', 'w3'):
            name = f'layers.0.ffn.experts.{expert}.{part}.weight'
            shape = (128, 64 if precision == 'MXFP4' else 128)
            scale_shape = (128, 4) if precision == 'MXFP4' else (1, 1)
            tensors[name] = MxTensor(precision, (128, 128), np.full(shape, 0x22 if precision == 'MXFP4' else 0x38, dtype=np.uint8), np.full(scale_shape, 127, dtype=np.uint8))
    path = tmp_path / 'native-experts.mfq'
    save(path, FileHeader(version=2, model_arch='deepseek_v4', extra={'hf_config': {
        'model_type': 'deepseek_v4', 'num_hidden_layers': 1, 'n_routed_experts': 2, 'num_experts_per_tok': 1}}), tensors)
    recipe = tmp_path / 'expert-sq.json'
    generate_recipe(path, recipe, valid_target(path, [profile]), [profile], backend='cpu')
    assert len(load_recipe(recipe).expert_selections) == 3
    def forbidden(*args, **kwargs): raise AssertionError('SQ fitting must retain original MFQ codes and E8M0 scales')
    monkeypatch.setattr(FullPrecisionMfqTensorSource, 'read_rows', forbidden)
    fit_model(path, recipe, tmp_path / 'expert-sq.mfq', backend='cpu')


def test_fp8_ple_stays_native_with_its_original_scale(tmp_path):
    root, _, _ = fixture(tmp_path)
    config = json.loads((root / 'config.json').read_text())
    config['text_config']['num_hidden_layers'] = 2
    (root / 'config.json').write_text(json.dumps(config))
    from safetensors.torch import load_file
    from mfq.formats.io import unpack_tensor_payload
    tensors = load_file(root / 'model.safetensors')
    weight_name = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight'
    scale_name = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale'
    weight = torch.arange(32 * 128).reshape(32, 128).remainder(9).float().to(torch.float8_e4m3fn)
    tensors[weight_name] = weight
    tensors[scale_name] = torch.tensor([.125], dtype=torch.bfloat16)
    save_file(tensors, root / 'model.safetensors')
    recipe = tmp_path / 'ple.json'
    generate_recipe(root, recipe, valid_target(root, ['NINT4']), ['NINT4'], backend='cpu')
    fit_model(root, recipe, tmp_path / 'ple.mfq', backend='cpu')
    store = open_mmap(tmp_path / 'ple.mfq')
    try:
        name = next(name for name in store.records if '.ngram.' in name and name.endswith('.weight'))
        assert store.records[name].dtype == 'F8_E4M3'
        assert store.records[name].nbytes == 20 + weight.numel()
        assert any('weight_scale' in name for name in store.records)
    finally:
        store.close()


def test_fp8_packed_gate_up_and_down_preserve_expert_scale_boundaries(tmp_path):
    root, _, _ = fixture(tmp_path)
    values = {}
    for projection, rows in [('gate_up_proj', 16), ('down_proj', 8)]:
        name = f'model.language_model.layers.0.mlp.experts.{projection}'
        values[name] = torch.ones(3, rows, 128).to(torch.float8_e4m3fn)
        values[name + '_scale_inv'] = torch.tensor([1., 2., 4.]).reshape(3, 1, 1)
    save_file(values, root / 'model.safetensors')
    source = ModelSource(root)
    try:
        for plan in source.plans:
            assert plan.shape == (3, 8, 128)
            with source.rows(plan, BatchGate()) as reader:
                actual = reader.read_rows(0, 24, device='cpu')
            torch.testing.assert_close(actual[:, 0], torch.tensor([1.] * 8 + [2.] * 8 + [4.] * 8))
            assert source.sq_precision(plan, 'FP8-SQ-8') is not None
    finally:
        source.close()
    recipe = tmp_path / 'packed.json'
    generate_recipe(root, recipe, valid_target(root, ['NINT4', 'NINT8']), ['NINT4', 'NINT8'], backend='cpu')
    fit_model(root, recipe, tmp_path / 'packed.mfq', backend='cpu')
    generate_recipe(root, tmp_path / 'packed-sq.json', valid_target(root, ['FP8-SQ-4']), ['FP8-SQ-4'], backend='cpu')
    fit_model(root, tmp_path / 'packed-sq.json', tmp_path / 'packed-sq.mfq', backend='cpu')


def test_data_free_nvq_expert_fitting_uses_canonical_codec(tmp_path):
    root, _, _ = fixture(tmp_path)
    recipe, output = tmp_path / 'nvq.json', tmp_path / 'nvq.mfq'
    target = valid_target(root, ['NVQ1-S'])
    generate_recipe(root, recipe, target, ['NVQ1-S'], backend='cpu')
    fitted = fit_model(root, recipe, output, backend='cpu', row_chunk=8)
    assert fitted['actual_bpw'] <= target
    store = open_mmap(output)
    try:
        assert all(store.records[f'model.block.0.mlp.experts.{p}.weight'].dtype == 'MFE' for p in ('gate', 'up', 'down'))
    finally:
        store.close()


def test_streaming_statistics_match_canonical_alphaq():
    from mfq.calibration.alphaq import alphaq_weight_statistics
    weight = torch.randn(300, 384, generator=torch.Generator().manual_seed(4))
    class Rows:
        def read_rows(self, start, end, *, device): return weight[start:end].to(device)
    class Gate(BatchGate):
        def __call__(self, requested=1024): return min(requested, 128)
    alpha, variance = matrix_statistics(Rows(), 300, 384, 'cpu', Gate())
    expected_alpha, expected_variance = alphaq_weight_statistics(weight[None], backend='svd')
    assert alpha == pytest.approx(expected_alpha[0], rel=1e-6)
    assert variance == pytest.approx(expected_variance[0], rel=1e-6)


def test_dense_model_preserves_norm_and_supports_full_precision_mfq_input(tmp_path):
    from mfq.formats.io import save
    from mfq.formats.header import FileHeader
    root = tmp_path / 'dense'; root.mkdir()
    config = {'model_type': 'qwen3', 'num_hidden_layers': 1, 'hidden_size': 128}
    (root / 'config.json').write_text(json.dumps(config))
    tensors = {'lm_head.weight': torch.randn(64, 128), 'model.norm.weight': torch.ones(128)}
    save_file(tensors, root / 'model.safetensors')
    recipe, output = tmp_path / 'dense.json', tmp_path / 'dense.mfq'
    generate_recipe(root, recipe, 12, ['NINT4', 'NINT8'], backend='cpu')
    fit_model(root, recipe, output, backend='cpu')
    store = open_mmap(output)
    try:
        assert store.records['model.norm.weight'].dtype == 'F32'
    finally:
        store.close()
    native = tmp_path / 'native.mfq'
    save(native, FileHeader(version=2, model_arch='qwen3', extra={'hf_config': config}), {
        'model.output.weight': np.random.default_rng(4).normal(size=(64,128)).astype(np.float32),
        'model.norm.weight': np.ones(128, dtype=np.float32)})
    recipe2 = tmp_path / 'mfq.json'
    generate_recipe(native, recipe2, 12, ['NINT4', 'NINT8'], backend='cpu')
    fit_model(native, recipe2, tmp_path / 'converted.mfq', backend='cpu')


def test_workspace_directories_persist_and_file_browser_lists_files(tmp_path):
    from mfq.server.protocol.quantization import RecipePayload
    tools = ToolJobHandlers(ModelCatalog([tmp_path]), ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
    workbench = tools.quantization_workbench
    output = tmp_path / 'exports'
    result = workbench.configure(QuantizationDirectories(import_directory=str(tmp_path), export_directory=str(output)))
    assert result.import_directory == str(tmp_path.resolve())
    assert result.export_directory == str(output.resolve())
    for family in ('MXFP8-SQ', 'FP8-SQ'):
        assert result.candidate_groups[family] == [f'{family}-{bits}' for bits in range(1, 9)]
    assert result.candidates == [name for group in result.candidate_groups.values() for name in group]
    assert len(result.candidates) == len(set(result.candidates)) == 32
    request = RecipePayload(input=str(tmp_path), output=str(output / 'recipe.json'), target_bpw=4, candidates=result.candidates)
    assert request.candidates == result.candidates
    fresh = ToolJobHandlers(ModelCatalog([tmp_path]), tools.paths).quantization_workbench
    assert fresh.workspace() == result
    (tmp_path / 'legal.json').write_text('{}')
    assert any(item.name == 'legal.json' and not item.directory for item in fresh.files(str(tmp_path)).data)
    with pytest.raises(Exception, match='not a directory'):
        fresh.configure(QuantizationDirectories(import_directory=str(tmp_path / 'legal.json'), export_directory=str(output)))


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('enabled', [None, False, True])
@pytest.mark.parametrize('layerwise', [None, False, True])
def test_workbench_imatrix_template_default_and_opt_out_reach_the_shared_collector(tmp_path, monkeypatch, prepared, enabled, layerwise):
    from mfq.server.protocol.quantization import WorkbenchImatrixPayload
    async def run():
        root = tmp_path / 'model'
        root.mkdir()
        corpus = tmp_path / ('corpus' if prepared else 'corpus.txt')
        if prepared:
            corpus.mkdir()
            (corpus / 'manifest.json').write_text('{}')
        else:
            corpus.write_text('hello')
        catalog = ModelCatalog([root])
        tools = ToolJobHandlers(catalog, ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        monkeypatch.setattr(tools.quantization_workbench, 'source', lambda _: SimpleNamespace(path=str(root), format='hf', imatrix_supported=True))
        async def collect(context, **kwargs):
            assert kwargs['apply_chat_template'] is (enabled is not False)
            assert kwargs['layerwise'] is (layerwise is not False)
            assert kwargs['corpus'] == corpus
            kwargs['output'].write_bytes(b'fixture')
            return {'chat_rendering': {'enabled': kwargs['apply_chat_template']}}
        monkeypatch.setattr(tools, '_collect_imatrix', collect)
        class Context:
            async def artifact(self, **kwargs): pass
        payload = {'model': str(root), 'corpus': str(corpus), 'output': str(tmp_path / 'output.imatrix')}
        if enabled is not None:
            payload['apply_chat_template'] = enabled
        if layerwise is not None:
            payload['layerwise'] = layerwise
        assert WorkbenchImatrixPayload.model_validate(payload).apply_chat_template is (enabled is not False)
        result = await tools.quantization_workbench.imatrix(Context(), payload)
        assert result['chat_rendering']['enabled'] is (enabled is not False)
    asyncio.run(run())


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('layerwise', [False, True])
def test_imatrix_template_choice_becomes_worker_render_mode(tmp_path, monkeypatch, enabled, layerwise):
    async def run():
        tools = ToolJobHandlers(ModelCatalog([tmp_path]), ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        output = tmp_path / 'out.imatrix'
        async def execute(context, argv, **kwargs):
            assert argv[argv.index('--render-mode') + 1] == ('auto' if enabled else 'plain')
            assert ('--resident' in argv) is (not layerwise)
            output.write_bytes(b'fixture')
            return [json.dumps({'event': 'calibration_corpus_prepared', 'chat_rendering': {'enabled': enabled}}),
                json.dumps({'event': 'imatrix_saved', 'entries': 1, 'tokens': 16})]
        monkeypatch.setattr(tools, '_run', execute)
        result = await tools._collect_imatrix(None, model=tmp_path, corpus=tmp_path, output=output, backend='metal', device='',
            attention='sdpa', objective='naq', window_length=16, batch_size=1, train_tokens=16, seed=7,
            accumulation_dtype='auto', apply_chat_template=enabled, layerwise=layerwise)
        assert result['chat_rendering']['enabled'] is enabled
    asyncio.run(run())


def test_imported_imatrix_creates_its_output_subdirectory(tmp_path):
    async def run():
        root, matrix, _ = fixture(tmp_path)
        tools = ToolJobHandlers(ModelCatalog([root]), ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        class Context:
            async def artifact(self, **kwargs): pass
        target = tmp_path / 'exports' / 'imatrix' / 'imported.imatrix'
        result = await tools.quantization_workbench.import_artifact(Context(), {'kind': 'imatrix', 'source': str(matrix), 'output': str(target)})
        assert result['output'] == str(target.resolve())
        assert target.read_bytes() == matrix.read_bytes()
    asyncio.run(run())


def test_preserve_scheduler_waits_for_service_then_reduces_batch(tmp_path, monkeypatch):
    async def run():
        tools = ToolJobHandlers(ModelCatalog([tmp_path]), ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        workbench = tools.quantization_workbench
        class Pool:
            busy = True
            async def quantization_pressure(self): return {'busy': self.busy}
        pool = Pool(); tools.runtime_manager = pool
        monkeypatch.setattr('mfq.server.services.quantization.psutil.virtual_memory', lambda: SimpleNamespace(total=32 << 30, available=3 << 30))
        class Context:
            cancel_event = asyncio.Event()
            progress_events = []
            def raise_if_cancelled(self):
                if self.cancel_event.is_set(): raise asyncio.CancelledError()
            async def progress(self, value, **kwargs): self.progress_events.append(kwargs)
        context = Context()
        assert await workbench.permit(context, 1024, 'allow', .3) == 1024
        waiting = asyncio.create_task(workbench.permit(context, 1024, 'preserve', .3))
        await asyncio.sleep(.05)
        assert not waiting.done()
        assert context.progress_events[-1]['data']['reason'] == 'service_busy'
        pool.busy = False
        rows = await asyncio.wait_for(waiting, 2)
        assert rows == 256
        pool.busy = True
        waiting = asyncio.create_task(workbench.permit(context, 1024, 'preserve', .3))
        await asyncio.sleep(.01); context.cancel_event.set()
        with pytest.raises(asyncio.CancelledError): await waiting
    asyncio.run(run())


@pytest.mark.parametrize('native_sq', [False, True])
def test_real_background_workers_register_recipe_and_complete_model(tmp_path, native_sq):
    async def run():
        root = native_source_fixture(tmp_path, 'mxfp8') if native_sq else fixture(tmp_path)[0]
        tools = ToolJobHandlers(ModelCatalog([root]), ToolJobPaths(tmp_path, sys.executable, None, None, None, None))
        store = SessionStore(tmp_path / 'jobs.sqlite3')
        manager = JobManager(store, tools.handlers())
        async def finish(job):
            for _ in range(600):
                result = store.get_job(job.id)
                if result.status.value not in {'queued', 'running', 'cancelling'}:
                    if result.status.value != 'succeeded':
                        events = store.list_job_events(job.id)
                        pytest.fail(str(result.error) + str(events)[-3000:])
                    return result
                await asyncio.sleep(.05)
            pytest.fail('worker timeout')
        try:
            recipe = tmp_path / 'exports' / 'recipe.json'
            candidates = ['MXFP8-SQ-6'] if native_sq else ['NINT4', 'NINT8']
            target = valid_target(root, candidates)
            generated = await finish(await manager.submit(CreateJobRequest(kind='quantization.recipe', payload={
                'input': str(root), 'output': str(recipe), 'target_bpw': target, 'candidates': candidates, 'backend': 'cpu', 'service_policy': 'allow'})))
            assert generated.result['artifact_kind'] == 'recipe'
            output = tmp_path / 'exports' / 'model.mfq'
            fitted = await finish(await manager.submit(CreateJobRequest(kind='quantization.fit', payload={
                'input': str(root), 'recipe': str(recipe), 'output': str(output), 'backend': 'cpu', 'service_policy': 'preserve'})))
            assert fitted.result['artifact_kind'] == 'model'
            assert (await tools.catalog.resolve_path(output)).path == output
            assert fitted.result['total_bytes'] == output.stat().st_size
        finally:
            await manager.close()
    asyncio.run(run())
