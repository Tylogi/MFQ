import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mfq.evaluation.reference import ReferenceWriter, tokenize_windows
from mfq.evaluation.source import ReferenceSource


def test_windows_package_includes_native_tokenizer_without_torch():
    script = (Path(__file__).resolve().parents[1] / 'packaging' / 'build_release_windows.ps1').read_text()
    build = script.split('function Build-PythonCli {', 1)[1]
    collected = re.search(r'foreach \(\$package in @\((.*?)\)\)', build, re.S)
    excluded = re.search(r'foreach \(\$module in @\((.*?)\)\)', build, re.S)
    assert collected is not None and excluded is not None
    packages = set(re.findall(r'"([^"]+)"', collected.group(1)))
    exclusions = set(re.findall(r'"([^"]+)"', excluded.group(1)))
    assert 'tokenizers' in packages and 'tokenizers' not in exclusions
    assert 'torch' in exclusions and 'torch' not in packages


def write_weights(path, values):
    payloads, header, offset = [], {}, 0
    for name, (dtype, value) in values.items():
        data = value.tobytes()
        header[name] = {'dtype': dtype, 'shape': list(value.shape), 'data_offsets': [offset, offset + len(data)]}
        payloads.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(payloads))


def test_native_trace_encoding_and_atomic_failure(tmp_path):
    tokens = np.arange(32, dtype='<i4').reshape(1, 32) % 19
    values = np.random.default_rng(123).normal(size=(15, 19)).astype(np.float32)
    values[0, 0] = -100
    output = tmp_path / 'reference'
    with ReferenceWriter(output, tokens, 19) as writer:
        nll = writer.logits(0, 0, values)
        writer.commit()
    raw = output.read_bytes()
    assert raw[:8] == b'_logit3_'
    target_offset = 16 + 12 + tokens.size * 4
    logp = values - np.log(np.exp(values).sum(-1, keepdims=True))
    targets = np.frombuffer(raw, '<f4', count=15, offset=target_offset)
    np.testing.assert_allclose(targets, logp[np.arange(15), tokens[0, 17:]], atol=1e-6)
    assert nll == pytest.approx(-targets.sum(dtype=np.float64))
    rows = np.frombuffer(raw, '<u2', offset=target_offset + 60).reshape(15, 24)
    headers = rows[:, :4].copy().view('<f4')
    decoded = rows[:, 4:23] * headers[:, :1] + headers[:, 1:]
    np.testing.assert_array_equal(decoded.argmax(-1), logp.argmax(-1))
    assert np.all(rows[:, -1] == 0)
    with pytest.raises(FileExistsError), ReferenceWriter(output, tokens, 19):
        pass
    failed = tmp_path / 'failed'
    with pytest.raises(ValueError), ReferenceWriter(failed, tokens, 19) as writer:
        writer.logits(0, 0, np.full((15, 19), np.nan))
    assert not failed.exists() and not failed.with_name('failed.partial').exists()


def test_native_tokenizer_keeps_wt2_bos_window_contract(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    tokenizer = Tokenizer(models.WordLevel({'<bos>': 0, '<eos>': 1, 'a': 2, '<unk>': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 0)])
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'bos_token': '<bos>', 'eos_token': '<eos>', 'add_bos_token': True}))
    tokens, metadata = tokenize_windows(tmp_path, ' '.join(['a'] * 80), 32, 2)
    np.testing.assert_array_equal(tokens[:, 0], [0, 0])
    assert np.all(tokens[:, 1:] == 2)
    assert metadata['add_bos'] and metadata['vocab_size'] == 4


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('experts', [False, True])
def test_native_microscale_source_reads_original_packed_rows(tmp_path, packed, experts):
    pytest.importorskip('mlx.core')
    from mfq.evaluation.reference_mlx import SourceWeights
    shape = (2, 3, 16 if packed else 32) if experts else (3, 16 if packed else 32)
    raw = np.full(shape, 0x22 if packed else 0x38, dtype=np.uint8)
    scales = np.full((*shape[:-1], 1), 128, dtype=np.uint8)
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'num_experts': 2}))
    write_weights(tmp_path / 'model.safetensors', {
        'model.embed_tokens.weight': ('F32', np.ones((5, 32), dtype=np.float32)),
        'test.weight': ('U8' if packed else 'F8_E4M3', raw),
        'test.weight_scale': ('F8_E8M0', scales),
    })
    source = ReferenceSource(tmp_path)
    weights = SourceWeights(source)
    values = np.asarray(weights.read('test.weight', 1, 2))
    np.testing.assert_array_equal(values, np.full((1, 3, 32) if experts else (1, 32), 2.))
    assert ('MXFP4' if packed else 'MXFP8') in source.summary()['source_precisions']


@pytest.mark.parametrize('dtype,one', [('F8_E4M3', 0x38), ('F8_E5M2', 0x3c)])
def test_native_fp8_block_scales_and_row_offsets(tmp_path, dtype, one):
    pytest.importorskip('mlx.core')
    from mfq.evaluation.reference_mlx import SourceWeights
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5',
        'quantization_config': {'weight_block_size': [2, 2]}}))
    write_weights(tmp_path / 'model.safetensors', {
        'model.embed_tokens.weight': ('F32', np.ones((5, 4), dtype=np.float32)),
        'test.weight': (dtype, np.full((4, 4), one, dtype=np.uint8)),
        'test.weight_scale_inv': ('F32', np.asarray([[1, 2], [3, 4]], dtype=np.float32)),
    })
    values = np.asarray(SourceWeights(ReferenceSource(tmp_path)).read('test.weight', 1, 3))
    np.testing.assert_array_equal(values, [[1, 1, 2, 2], [3, 3, 4, 4]])


@pytest.mark.parametrize('family', ['qwen3_5', 'qwen3_5_moe', 'gemma4_text', 'gemma4_moe', 'gemma4_shared_ple'])
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_mlx_layerwise_and_resident_match_original_precision_graph(tmp_path, monkeypatch, family, dtype):
    mx = pytest.importorskip('mlx.core')
    pytest.importorskip('mlx_lm')
    from mlx.utils import tree_flatten
    from mlx_lm.models import gemma4_text, qwen3_5

    from mfq.evaluation.reference_mlx import generate
    qwen = family.startswith('qwen')
    config = {'model_type': family if qwen else 'gemma4_text', 'hidden_size': 32, 'vocab_size': 67, 'intermediate_size': 64,
        'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 16,
        'max_position_embeddings': 64, 'tie_word_embeddings': False, 'rms_norm_eps': 1e-6}
    if qwen:
        config.update(full_attention_interval=2, linear_num_value_heads=2, linear_num_key_heads=1,
            linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            rope_parameters={'rope_theta': 10000., 'partial_rotary_factor': .5, 'mrope_section': [1, 1, 2]})
        if family.endswith('_moe'):
            config.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=32)
        model = qwen3_5.TextModel(qwen3_5.TextModelArgs.from_dict(config))
    else:
        config.update(global_head_dim=16, num_global_key_value_heads=1, num_kv_shared_layers=0,
            hidden_size_per_layer_input=0, sliding_window=16, layer_types=['sliding_attention', 'full_attention'],
            enable_moe_block=False, use_double_wide_mlp=False, final_logit_softcapping=30.)
        if family.endswith('_moe'):
            config.update(enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=32)
        if family.endswith('_shared_ple'):
            config.update(num_hidden_layers=4, num_kv_shared_layers=2, hidden_size_per_layer_input=8,
                vocab_size_per_layer_input=67, layer_types=['sliding_attention', 'full_attention'] * 2)
        model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config))
    mx.random.seed(123)
    values = {}
    chosen = getattr(mx, dtype)
    for name, value in tree_flatten(model.parameters()):
        if value.ndim >= 2:
            value = mx.random.normal(value.shape) * .05
        value = value.astype(chosen)
        model.load_weights([(name, value)], strict=False)
        exported = value
        if qwen:
            if name.endswith(('input_layernorm.weight', 'post_attention_layernorm.weight', 'q_norm.weight', 'k_norm.weight', 'model.norm.weight')):
                exported = exported - 1
            if name.endswith('conv1d.weight'):
                exported = exported.transpose(0, 2, 1)
        values[name] = ('BF16', np.asarray(exported.view(mx.uint16))) if dtype == 'bfloat16' else ('F32', np.asarray(exported))
    for index in range(config['num_hidden_layers']):
        base = f'model.layers.{index}.' + ('mlp.switch_mlp' if qwen else 'experts.switch_glu')
        if base + '.gate_proj.weight' in values:
            gate_dtype, gate = values.pop(base + '.gate_proj.weight')
            _, up = values.pop(base + '.up_proj.weight')
            head = base.replace('switch_mlp', 'experts').removesuffix('.switch_glu')
            values[head + '.gate_up_proj'] = (gate_dtype, np.concatenate((gate, up), axis=-2))
            values[head + '.down_proj'] = values.pop(base + '.down_proj.weight')
    model.eval()
    mx.eval(model.parameters())
    (tmp_path / 'config.json').write_text(json.dumps(config))
    write_weights(tmp_path / 'model.safetensors', values)
    source = ReferenceSource(tmp_path)
    tokens = np.random.default_rng(42).integers(0, 67, (2, 32), dtype=np.int32)
    expected = np.concatenate([np.asarray(model(mx.array(ids[None]))[:, 16:31].astype(mx.float32)) for ids in tokens])
    del model
    outputs = []
    for layerwise in (True, False):
        scratch = tmp_path / str(layerwise)
        scratch.mkdir()
        output = scratch / 'reference'
        with ReferenceWriter(output, tokens, 67) as writer:
            nll, metadata = generate(source, tokens, writer, scratch, layerwise=layerwise)
            writer.commit()
        assert np.isfinite(nll) and metadata['runtime'] == 'mlx'
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    target_offset = 16 + 24 + tokens.size * 4
    actual = np.frombuffer(outputs[0], '<f4', count=30, offset=target_offset).reshape(2, 15)
    logp = expected - np.log(np.exp(expected).sum(-1, keepdims=True))
    target = np.take_along_axis(logp, tokens[:, 17:, None], -1).squeeze(-1)
    np.testing.assert_allclose(actual, target, atol=.005 if dtype == 'bfloat16' else 5e-5)
    if family == 'qwen3_5' and dtype == 'float32':
        from mfq.evaluation.reference_mlx import SourceWeights
        original = SourceWeights.layer_weights
        failures = []
        def simulate_pressure(self, index, gemma):
            if index == 1 and not failures:
                failures.append(index)
                raise RuntimeError('allocation failed: out of memory')
            return original(self, index, gemma)
        monkeypatch.setattr(SourceWeights, 'layer_weights', simulate_pressure)
        scratch = tmp_path / 'fallback'
        scratch.mkdir()
        with ReferenceWriter(scratch / 'reference', tokens, 67) as writer:
            _, metadata = generate(source, tokens, writer, scratch, layerwise=False)
            writer.commit()
        assert metadata['layerwise'] is True and failures == [1]
        assert (scratch / 'reference').read_bytes() == outputs[0]


def test_native_cuda_memory_plan_respects_visible_device(monkeypatch):
    from types import SimpleNamespace

    from mfq.evaluation import source
    from mfq.server.services import hardware
    monkeypatch.setattr(source, 'ReferenceSource', lambda _: SimpleNamespace(
        text={'hidden_size': 16}, records={}, summary=lambda: {'parameters': 1000}))
    monkeypatch.setattr(source.psutil, 'virtual_memory', lambda: SimpleNamespace(total=64 << 30, available=32 << 30))
    monkeypatch.setattr(hardware, 'gpu_memory_status', lambda: [
        {'id': '0', 'memory_total_bytes': 8 << 30, 'memory_available_bytes': 256 << 20},
        {'id': '1', 'memory_total_bytes': 24 << 30, 'memory_available_bytes': 16 << 30}])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    assert source.loading_plan('/source', backend='cuda', device='cuda:0')['resident_allowed']
    assert not source.loading_plan('/source', backend='cuda', device='cuda:1')['resident_allowed']
    with pytest.raises(ValueError, match='not visible'):
        source.loading_plan('/source', backend='cuda', device='cuda:2')


def test_glm_tokenizer_uses_plain_text_without_bos_or_chat(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    tokenizer = Tokenizer(models.WordLevel({'a': 0, '<|endoftext|>': 1, '[gMASK]': 2, '<sop>': 3}, unk_token='a'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True, trim_offsets=False)
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': 'TokenizersBackend', 'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
        'extra_special_tokens': ['[gMASK]', '<sop>'], 'chat_template': '[gMASK]<sop>{{ messages }}'}))
    tokens, metadata = tokenize_windows(tmp_path, ' '.join(['a'] * 1024), 512, 2)
    assert tokens.shape == (2, 512) and np.all(tokens == 0)
    assert metadata['bos_token'] is None and metadata['eos_token'] == metadata['pad_token'] == 1
    assert not metadata['add_bos'] and not metadata['add_eos']


@pytest.mark.parametrize('device', ['cuda:0', 'mps'])
def test_glm_reference_gate_is_independent_of_imatrix_support(tmp_path, monkeypatch, device):
    from mfq.evaluation import reference, reference_cuda
    from mfq.server.services import evaluation_datasets
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'glm5_next', 'text_config': {
        'model_type': 'glm5_next_text', 'hidden_size': 4, 'max_position_embeddings': 512, 'vocab_size': 19}}))
    write_weights(tmp_path / 'model.safetensors', {'model.language_model.embed_tokens.weight': ('F32', np.ones((19, 4), np.float32))})
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('fixture corpus')
    monkeypatch.setattr(evaluation_datasets, 'WT2_TEXT_SHA256', reference.sha256(corpus))
    monkeypatch.setattr(reference, 'loading_plan', lambda *a, **k: {'resident_allowed': True})
    monkeypatch.setattr(reference, 'tokenize_windows', lambda *a: (np.zeros((1, 512), dtype='<i4'), {'add_bos': False}))
    def generate(source, tokens, writer, directory, **kwargs):
        assert source.architecture == 'glm5_next_text' and not source.summary()['imatrix_supported']
        return writer.logits(0, 0, np.zeros((255, 19), np.float32)), {
            'runtime': 'test', 'dtype': 'float16', 'attention': 'native_kda_dsa'}
    monkeypatch.setattr(reference_cuda, 'generate', generate)
    output = tmp_path / 'reference.bin'
    if device == 'mps':
        with pytest.raises(ValueError, match='requires native CUDA'):
            reference.generate_reference(tmp_path, corpus, output, chunks=1, device=device)
        assert not output.exists()
    else:
        result = reference.generate_reference(tmp_path, corpus, output, chunks=1, device=device)
        manifest = json.loads(Path(result['manifest']).read_text())
        assert manifest['evaluation']['logit_positions'] == [256, 510]
        assert manifest['evaluation']['target_positions'] == [257, 511]
        assert result['scored_tokens'] == 255


@pytest.mark.parametrize('failure', ['', 'size', 'order', 'missing', 'exit', 'nonfinite'])
def test_cuda_chunk_handoff_preserves_f32_and_detects_failure(tmp_path, monkeypatch, failure):
    from mfq.evaluation import reference_cuda
    helper = tmp_path / 'runtime.py'
    helper.write_text('''import json, struct, sys
from pathlib import Path
args = sys.argv
assert '--reference-chunk-handoff' in args
output = Path(args[args.index('--reference-output') + 1])
failure = args[-1]
print(json.dumps({'event': 'source_loading', 'layerwise': True}), flush=True)
if failure == 'exit': sys.exit(7)
if failure == 'missing': sys.exit(0)
for chunk in range(2):
    values = [i * .001234567 + chunk for i in range(15 * 19)]
    if failure == 'nonfinite': values[0] = float('nan')
    raw = struct.pack('<' + 'f' * len(values), *values)
    output.write_bytes(raw[:-4] if failure == 'size' else raw)
    print(json.dumps({'event': 'wt2_logits_ready', 'chunk': 1 if failure == 'order' else chunk}), flush=True)
    if sys.stdin.readline() != 'continue\\n': sys.exit(9)
    output.unlink()
''')
    popen = subprocess.Popen
    processes = []
    def launch(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '5'
        process = popen([sys.executable, str(helper), *command[1:], failure], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(reference_cuda.subprocess, 'Popen', launch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5')
    tokens = np.zeros((2, 32), dtype='<i4')
    output = tmp_path / 'reference.bin'
    with ReferenceWriter(output, tokens, 19) as writer:
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                reference_cuda.generate(SimpleNamespace(architecture='glm5_next_text', path=tmp_path), tokens, writer, tmp_path,
                    layerwise=False, runtime=sys.executable, device='cuda:1')
        else:
            expected_rows = []
            original = writer.logits
            def record(chunk, start, values):
                expected = np.arange(15 * 19, dtype=np.float64).reshape(15, 19) * .001234567 + chunk
                np.testing.assert_array_equal(values, expected.astype('<f4'))
                expected_rows.append(chunk)
                return original(chunk, start, values)
            monkeypatch.setattr(writer, 'logits', record)
            nll, metadata = reference_cuda.generate(SimpleNamespace(architecture='glm5_next_text', path=tmp_path), tokens, writer,
                tmp_path, layerwise=False, runtime=sys.executable, device='cuda:1')
            assert np.isfinite(nll) and expected_rows == [0, 1]
            assert metadata['layerwise'] and metadata['logits_handoff'] == 'float32_per_chunk'
            assert not (tmp_path / 'logits.f32').exists()
            writer.commit()
    assert processes[0].poll() is not None
    assert output.exists() == (not failure)
    assert not (tmp_path / 'reference.bin.partial').exists()
