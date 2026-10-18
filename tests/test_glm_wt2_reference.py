"""Small HF source and native WT2/KL checks; CUDA execution is opt-in."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from mfq.evaluation.reference import ReferenceWriter
from mfq.evaluation.source import ReferenceSource
from tests.test_cuda_flash_next_runtime import glm_model_fixture
from tests.test_native_wt2_reference import write_weights


@pytest.fixture
def glm_hf(tmp_path):
    import torch
    config, weights = glm_model_fixture(['linear_attention', 'deepseek_sparse_attention'], True)
    text = config['text_config']
    text.update(hc_mult=4, max_position_embeddings=512, num_nextn_predict_layers=1)
    rng = np.random.default_rng(5304)
    for name in list(weights):
        if name.endswith('.mhc.pre.function'):
            weights[name] = rng.normal(0, .02, (24, 128)).astype(np.float32)
        elif name.endswith('.mhc.pre.base'):
            weights[name] = rng.normal(0, .02, 24).astype(np.float32)
    # Reassemble the actual HF K/V source; the importer must derive both heads.
    base = 'model.block.1.attention'
    query = weights.pop(base + '.latent.query_embedding.weight').transpose(0, 2, 1)
    value = weights.pop(base + '.latent.output_unembedding.weight')
    weights[base + '.key_value_b.source.weight'] = np.concatenate((query, value), axis=1).reshape(-1, 16)
    source_weights = {}
    for name, values in weights.items():
        if '.mlp.experts.' in name:
            for expert, matrix in enumerate(values):
                roles = zip(('gate', 'up'), np.split(matrix, 2)) if '.gate_up.' in name else [('down', matrix)]
                for role, matrix in roles:
                    source_weights[name.split('.experts.')[0] + f'.experts.{expert}.{role}.weight'] = matrix
        else:
            source_weights[name] = values
    encoded = {}
    for name, values in source_weights.items():
        packed = values.ndim == 2 and name.endswith('.weight') and '.linear_attention.' not in name and not any(
            word in name for word in ('embedding', 'norm', 'router'))
        if name in {'model.output.weight', 'model.token_embedding.weight'}:
            raw = torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16).view(torch.uint16).numpy()
            encoded[name] = ('BF16', raw)
        elif packed:
            scales = np.full(((values.shape[0] + 127) // 128, (values.shape[1] + 127) // 128), .125, np.float32)
            raw = torch.from_numpy(np.ascontiguousarray(values / .125)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            encoded[name] = ('F8_E4M3', raw)
            encoded[name.removesuffix('.weight') + '.weight_scale_inv'] = ('F32', scales)
        else:
            encoded[name] = ('F32', values)
    # Invalid optional weights make accidental vision/MTP loading observable.
    encoded['predictor.embedding_norm.weight'] = ('F32', np.zeros(1, np.float32))
    encoded['vision.patch_embedding.weight'] = ('F32', np.zeros(1, np.float32))
    root = tmp_path / 'hf'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(config))
    write_weights(root / 'model.safetensors', encoded)
    return root, config, weights


def test_glm_hf_fixture_contains_original_fp8_and_full_mhc(glm_hf):
    root, config, _ = glm_hf
    source = ReferenceSource(root)
    assert source.architecture == 'glm5_next_text'
    assert source.summary()['source_precisions'] == ['BF16', 'FP32', 'FP8']
    assert config['text_config']['hc_mult'] == 4
    assert source.records['model.output.weight'].dtype == 'BF16'
    assert source.records['model.token_embedding.weight'].dtype == 'BF16'
    assert source.records['model.block.0.attention.mhc.pre.function'].shape == (24, 128)
    assert source.records['model.block.1.attention.key_value_b.source.weight'].dtype == 'F8_E4M3'


@pytest.fixture
def cuda_runtime():
    runtime = os.environ.get('MFQ_CUDA_REFERENCE_RUNTIME') or shutil.which('mfq-runtime')
    if not runtime:
        pytest.skip('set MFQ_CUDA_REFERENCE_RUNTIME to the native mfq-runtime executable')
    candidates = [ctypes.util.find_library('cudart')]
    if os.name == 'nt':
        candidates += [str(path) for directory in (Path(runtime).parent, Path(os.environ.get('CUDA_PATH', '.')) / 'bin')
            for path in directory.glob('cudart64_*.dll')]
    else:
        candidates += [str(path) for path in Path('/usr/local').glob('cuda*/lib64/libcudart.so*')]
    for candidate in filter(None, candidates):
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        count = ctypes.c_int()
        if library.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value:
            return Path(runtime)
    pytest.skip('CUDA runtime reports no available device')


def test_native_glm_layerwise_resident_and_handoff(glm_hf, cuda_runtime, tmp_path):
    from mfq.evaluation.reference_cuda import generate
    root, _, _ = glm_hf
    source = ReferenceSource(root)
    ids = np.random.default_rng(5305).integers(0, 32, (1, 512), dtype='<i4')
    tokens = np.repeat(ids, 2, axis=0)
    token_path = tmp_path / 'tokens.bin'
    token_path.write_bytes(struct.pack('<II', 2, 512) + tokens.tobytes())
    outputs = []
    for resident in (False, True):
        output = tmp_path / f'{resident}.f32'
        command = [str(cuda_runtime), '--wt2-reference', '--model', str(root), '--ctx-size', '512',
            '--reference-input', str(token_path), '--reference-output', str(output)]
        result = subprocess.run(command + (['--resident'] if resident else []), capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        assert [row['layer'] for row in events if row['event'] == 'wt2_layer'] == [1, 2]
        assert not any(row.get('fallback_reason') for row in events)
        assert not Path(str(output) + '.hidden').exists()
        logits = np.fromfile(output, '<f4').reshape(2, 255, 32)
        assert np.isfinite(logits).all()
        # Repeated windows must start with fresh KDA and MLA state.
        np.testing.assert_array_equal(logits[0], logits[1])
        outputs.append(logits)
    # One FP16 epsilon bounds a storage-rounding difference; both paths run
    # the same operators with the same source weights and full contexts.
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=np.finfo(np.float16).eps, atol=np.finfo(np.float16).eps)
    scratch = tmp_path / 'handoff'
    scratch.mkdir()
    with ReferenceWriter(tmp_path / 'reference.bin', tokens, 32) as writer:
        original = writer.logits
        def check(chunk, start, values):
            np.testing.assert_array_equal(values, outputs[0][chunk, start:start + len(values)])
            return original(chunk, start, values)
        writer.logits = check
        nll, metadata = generate(source, tokens, writer, scratch, layerwise=True, runtime=cuda_runtime, device='cuda:0')
        assert np.isfinite(nll) and metadata['attention'] == 'native_kda_dsa'
        writer.commit()
    assert not (scratch / 'logits.f32').exists()


def write_mixed_glm_mfq(model, config, original_weights):
    from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
    from mfq.formats.header import FileHeader
    from mfq.formats.io import save
    from mfq.formats.mfe import MfePool, MfeTensor
    from mfq.formats.nint import NintSpec
    from mfq.quantize.nint_quant import quantize
    from tests.mixed_family_fixtures import make_flat_family
    weights = dict(original_weights)
    # MFQ uses absorbed MLA tensors, which were removed when forming HF K/V.
    matrix = weights.pop('model.block.1.attention.key_value_b.source.weight').reshape(2, 32, 16)
    weights['model.block.1.attention.latent.query_embedding.weight'] = matrix[:, :16].transpose(0, 2, 1).copy()
    weights['model.block.1.attention.latent.output_unembedding.weight'] = matrix[:, 16:].copy()
    for name, values in list(weights.items()):
        if '.experts.' not in name:
            continue
        _, output, width = values.shape
        pools = (
            MfePool(np.array([0, 2], np.int32), quantize(values[[0, 2]].reshape(2 * output, width), NintSpec(4, 8, 6))),
            MfePool(np.array([1], np.int32), make_flat_family('NVQ2', rows=output, neuron_len=width)),
        )
        weights[name] = MfeTensor(values.shape, pools)
    graph = dict(schema_version=1, architecture='glm5_next',
        canonical_naming=dict(namespace='mfq.tensor', version=1, component_roots=['model', 'vision', 'predictor']),
        topology=dict(text_layers=2, vision_layers=1, predictor_layers=1),
        graph=dict(kind='causal_lm', backbone='glm5_next'), components=[
            dict(kind='text', tensor_root='model', implementation='glm5_next', policy='decoder'),
            dict(kind='vision', tensor_root='vision', implementation='grid_vit', policy='optional',
                input_contract='grid_vision.v1', position_policy='grid_mrope'),
            dict(kind='predictor', tensor_root='predictor', implementation='next_token_prediction', policy='optional')],
        capabilities=['text', 'vision', 'mtp'])
    weights.update({'predictor.embedding_norm.weight': np.zeros(1, np.float32),
        'vision.patch_embedding.weight': np.zeros(1, np.float32),
        MODEL_CONFIG_ASSET: json.dumps(config).encode(), MODEL_GRAPH_ASSET: json.dumps(graph).encode()})
    save(model, FileHeader(version=2, model_arch='glm5_next'), weights)


def test_glm_mixed_mfq_fixture_serializes_on_cpu(glm_hf, tmp_path):
    from mfq.formats.io import load
    from mfq.formats.mfe import MfeTensor
    _, config, weights = glm_hf
    model = tmp_path / 'mixed.mfq'
    write_mixed_glm_mfq(model, config, weights)
    _, tensors = load(model)
    expert = tensors['model.block.0.mlp.experts.gate_up.weight']
    assert isinstance(expert, MfeTensor) and len(expert.pools) == 2
    assert tensors['predictor.embedding_norm.weight'].shape == (1,)
    assert tensors['vision.patch_embedding.weight'].shape == (1,)


@pytest.mark.parametrize('offload', [False, True])
def test_native_glm_kl_mixed_experts_and_optional_components(glm_hf, cuda_runtime, tmp_path, offload):
    from mfq.evaluation.reference_cuda import generate
    root, config, weights = glm_hf
    tokens = np.tile(np.arange(512, dtype='<i4') % 32, (1, 1))
    scratch = tmp_path / 'reference-scratch'
    scratch.mkdir()
    reference = tmp_path / 'reference.bin'
    with ReferenceWriter(reference, tokens, 32) as writer:
        generate(ReferenceSource(root), tokens, writer, scratch, layerwise=True, runtime=cuda_runtime, device='cuda:0')
        writer.commit()
    model = tmp_path / 'model.mfq'
    write_mixed_glm_mfq(model, config, weights)
    evaluator = cuda_runtime.with_name('mfq-eval' + cuda_runtime.suffix)
    environment = os.environ.copy()
    if offload:
        environment.update(MFQ_MOE_PRELOAD_ALL='1', MFQ_MOE_ASSERT_RESIDENT='1', MFQ_MOE_HOST_CACHE_BYTES='1048576')
    result = subprocess.run([str(evaluator), 'kl', '--model', str(model), '--kl-base', str(reference), '--ctx-size', '512',
        '--kl-reference-n-batch', '512', '--kl-reference-n-ubatch', '512'] +
        (['--moe-gpu-cache-gb', '0.000001'] if offload else []), text=True, capture_output=True, timeout=180, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith('cpp_kl_result '))
    fields = dict(field.split('=', 1) for field in line.split()[1:])
    assert fields['scored_tokens'] == '255'
    assert all(np.isfinite(float(fields[name])) for name in ('kld', 'bf16_ppl', 'mfq_ppl', 'same_top'))
    assert 0 <= float(fields['same_top']) <= 1
    if offload:
        plans = [dict(field.split('=', 1) for field in line.split()[1:]) for line in result.stderr.splitlines()
            if line.startswith('moe_residency_plan ')]
        assert plans and any(int(plan['cold_bytes']) > 0 for plan in plans)
