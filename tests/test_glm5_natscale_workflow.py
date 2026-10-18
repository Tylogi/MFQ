"""GLM FP8, recipe, multimodal, measurement and resumable export integration."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mfq.cli import main
from mfq.calibration import alphaq_cache
from mfq.calibration.alphaq_model import model_scope
from mfq.calibration.alphaq_workflow import planned
from mfq.calibration.artifact import CalibrationScheme
from mfq.calibration.ud_recipes import UD_MODELS
from mfq.formats.assets import MODEL_GRAPH_ASSET
from mfq.formats.io import open_mmap
from mfq.quantize.imatrix import ImportanceEntry, load_importance_matrix, save_importance_matrix
from tests.test_flash_next_architecture import _glm_config


def fixture(tmp_path, monkeypatch, *, expert_rows=128):
    root = tmp_path/'model'; root.mkdir()
    config = _glm_config()
    text = config['text_config']
    text.update(hidden_size=128, intermediate_size=128, num_hidden_layers=2,
        vocab_size=32, layer_types=['linear_attention', 'deepseek_sparse_attention'],
        mlp_layer_types=['dense', 'sparse'], first_k_dense_replace=1,
        n_routed_experts=3, num_experts_per_tok=1, moe_intermediate_size=128,
        num_attention_heads=2, q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16,
        v_head_dim=16, indexer_types=['full', 'full'])
    text['linear_attn_config'].update(num_heads=2, head_dim=64, kda_layers=[0], full_attn_layers=[1])
    config['vision_config'].update(hidden_size=16, out_hidden_size=128, depth=1, num_heads=2)
    config['quantization_config'] = dict(quant_method='fp8', weight_block_size=[128, 128],
                                        activation_scheme='dynamic', fmt='e4m3')
    (root/'config.json').write_text(json.dumps(config))
    gen = torch.Generator().manual_seed(178)
    tensors, recipe = {}, {}
    def add(name, shape, alias=None, dtype='Q8_0', fp8=False):
        weight = torch.randn(shape, generator=gen)*.3
        tensors[name] = weight.to(torch.float8_e4m3fn) if fp8 else weight.bfloat16()
        if fp8:
            tensors[name.removesuffix('.weight')+'.weight_scale_inv'] = torch.full(
                tuple(math.ceil(d/128) for d in shape), .25)
        if alias is not None:
            recipe[alias] = dtype
    add('model.language_model.embed_tokens.weight', (32, 128), 'token_embd.weight')
    add('lm_head.weight', (32, 128), 'output.weight')
    add('model.language_model.norm.weight', (128,), 'output_norm.weight', 'F32')
    add('model.visual.patch_embed.proj.weight', (16, 3, 2, 2))
    add('model.visual.merger.proj.weight', (128, 16))
    for layer in range(3):
        prefix = f'model.language_model.layers.{layer}.'
        alias = f'blk.{layer}.'
        add(prefix+'input_layernorm.weight', (128,), alias+'attn_norm.weight', 'F32')
        add(prefix+'post_attention_layernorm.weight', (128,), alias+'ffn_norm.weight', 'F32')
        if layer == 0:
            for p in ('q', 'k', 'v'):
                add(prefix+f'self_attn.{p}_proj.weight', (128, 128), alias+f'attn_{p}.weight')
            for p in ('gate', 'up', 'down'):
                add(prefix+f'mlp.{p}_proj.weight', (128, 128), alias+f'ffn_{p}.weight', fp8=True)
        else:
            add(prefix+'self_attn.q_a_proj.weight', (16, 128), alias+'attn_q_a.weight', fp8=True)
            add(prefix+'self_attn.kv_b_proj.weight', (64, 16), fp8=True)
            recipe[alias+'attn_k_b.weight'] = 'Q8_0'
            recipe[alias+'attn_v_b.weight'] = 'Q8_0'
            for p in ('gate', 'up', 'down'):
                add(prefix+f'mlp.shared_experts.{p}_proj.weight', (128, 128),
                    alias+f'ffn_{p}_shexp.weight', fp8=True)
                recipe[alias+f'ffn_{p}_exps.weight'] = 'Q4_K' if layer == 2 else 'IQ1_S'
                for expert in range(3):
                    shape = (128, expert_rows) if p == 'down' else (expert_rows, 128)
                    add(prefix+f'mlp.experts.{expert}.{p}_proj.weight', shape, fp8=True)
            add(prefix+'mlp.gate.weight', (3, 128), alias+'ffn_gate_inp.weight')
        add(prefix+'self_attn.o_proj.weight', (128, 128), alias+'attn_output.weight', fp8=layer > 0)
    for suffix, alias in (('enorm', 'enorm'), ('hnorm', 'hnorm'), ('shared_head.norm', 'shared_head_norm')):
        add(f'model.language_model.layers.2.{suffix}.weight', (128,), f'blk.2.nextn.{alias}.weight', 'F32')
    add('model.language_model.layers.2.eh_proj.weight', (128, 256), 'blk.2.nextn.eh_proj.weight')
    save_file(tensors, root/'model.safetensors')
    entries = {f'blk.1.ffn_{p}_exps.weight': ImportanceEntry(
        np.tile(np.linspace(.2, 2., expert_rows if p == 'down' else 128), (3, 1)),
        np.array([7, 3, 0])) for p in ('gate', 'up', 'down')}
    entries['blk.1.ffn_gate_inp.weight'] = ImportanceEntry(np.ones((1, 128)), np.array([10]))
    for layer in (0, 1):
        for alias, dtype in recipe.items():
            if not alias.startswith(f'blk.{layer}.') or not alias.endswith('.weight') or '_exps' in alias:
                continue
            if dtype != 'F32' and 'attn_k_b' not in alias and 'attn_v_b' not in alias:
                width = 128
                entries[alias] = ImportanceEntry(np.ones((1, width)), np.array([10]))
    imatrix = tmp_path/'imatrix.npz'; save_importance_matrix(imatrix, entries)
    recipe_path = tmp_path/'tiny-recipe.json'
    recipe_path.write_text(json.dumps(dict(format='mfq.gguf-recipe.v1', tensor_types=recipe, ud_file_bytes=600000)))
    registry = dict(config={k: text[k] for k in ('hidden_size', 'num_hidden_layers', 'n_routed_experts',
        'num_experts_per_tok', 'vocab_size')}, parameters=sum(t.numel() for n,t in tensors.items()
            if 'scale_inv' not in n and not n.startswith(('model.visual.', 'model.language_model.layers.2.'))),
        directory='', prefix='', caps={'IQ1_S': 600000})
    monkeypatch.setitem(UD_MODELS, 'glm5_next_text', registry)
    monkeypatch.setattr('mfq.calibration.alphaq_workflow.automatic_recipe', lambda *a, **k: recipe_path)
    return root, imatrix, recipe_path


def command(root, imatrix, cache, output):
    return ['quantize', str(root), str(output), '--imatrix', str(imatrix), '--ud-tier', 'IQ1_S',
        '--alphaq-allocator', 'natscale', '--alphaq-profiles', 'NINT4,NINT8',
        '--alphaq-cache', str(cache), '--backend', 'cpu', '--device', 'cpu']


@pytest.mark.parametrize('mode', ['analytic', 'measured'])
def test_cpu_glm_fp8_vision_predictor_export_and_cache_reuse(tmp_path, monkeypatch, mode):
    root, imatrix, _ = fixture(tmp_path, monkeypatch)
    cache, output = tmp_path/'cache', tmp_path/'out.mfq'
    if mode == 'measured':
        assert main(['calibrate', 'natscale-errors', '--model', str(root), '--imatrix', str(imatrix),
            '--cache', str(cache), '--profiles', 'NINT4,NINT8', '--backend', 'cpu', '--device', 'cpu']) == 0
        measured = json.loads((cache/'natscale-errors.json').read_text())
        assert measured['complete']
        assert measured['errors']['1']['down']['NINT8'] < measured['errors']['1']['down']['NINT4']
    argv = [*command(root, imatrix, cache, output), '--natscale-error', mode]
    assert main(argv) == 0
    store = open_mmap(output)
    try:
        policy = store.header.extra['alphaq']
        assert output.stat().st_size <= 600000+policy['vision_bytes']+policy['mtp_bytes']
        assert policy['vision_bytes'] > 0 and policy['mtp_bytes'] > 0
        assert store.records['vision.patch_embedding.weight'].dtype == 'BF16'
        assert store.records['predictor.block.0.mlp.experts.gate.weight'].dtype == 'MFE'
        assert store['model.block.1.mlp.experts.down.weight'].shape == (3, 128, 128)
        assert store['predictor.block.0.mlp.experts.gate.weight'].pools[0].tensor.spec.bits == 4
        for name in store.records:
            store[name]  # Parse every tensor/asset through the Python reader.
        graph = json.loads(store.read_blob(MODEL_GRAPH_ASSET))
        assert {'vision', 'predictor'} <= {c['tensor_root'] for c in graph['components']}
        assert 'unweighted' in policy['mtp_imatrix']
    finally:
        store.close()
    def forbidden(*args, **kwargs):
        raise AssertionError('completed statistics must be reused')
    monkeypatch.setattr(alphaq_cache, '_collect', forbidden)
    assert main([*argv, '--overwrite']) == 0
    report = json.loads(next((cache/'allocations').glob('*/export-report.json')).read_text())
    assert report['main_text_file_bytes'] <= report['maximum_file_bytes']
    assert report['file_bytes'] == report['main_text_file_bytes']+report['vision_bytes']+report['mtp_bytes']


def test_statistics_energy_matches_decoded_fp8_and_old_cache_is_invalidated(tmp_path, monkeypatch):
    from mfq.calibration.alphaq_cache import collect_statistics, rows_for_plan
    from mfq.tools import quantize_hf_to_mfq as q
    root, imatrix, _ = fixture(tmp_path, monkeypatch)
    matrix = load_importance_matrix(imatrix)
    plans = planned(root, None, CalibrationScheme(None, 'test', 0, {}, {}, {}), {}, 'native')
    scope = model_scope(plans, matrix, exclude_scaled_dense=True)
    cache = tmp_path/'statistics'
    stat = collect_statistics(root, scope.experts[:1], cache, 'cpu')[0]
    with rows_for_plan(root, scope.experts[0]) as source:
        weight = source.read_rows(0, 384, device='cpu').reshape(3, 128, 128)
    np.testing.assert_allclose(stat.column_energy, weight.square().sum(dim=1).numpy())
    from mfq.calibration.natscale import projection_energies
    expected = (stat.column_energy*matrix.entries[scope.imatrix_entries[stat.name]].values).sum(axis=1)
    expected[-1] = 0
    np.testing.assert_allclose(projection_energies([stat], scope, matrix)[stat.name], expected)
    path = cache/'statistics.json'; doc = json.loads(path.read_text())
    del doc['tensors'][stat.name]['column_energy']; path.write_text(json.dumps(doc))
    original, calls = alphaq_cache._collect, []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(alphaq_cache, '_collect', counted)
    collect_statistics(root, scope.experts[:1], cache, 'cpu')
    assert calls


def test_measured_mode_covers_every_profile_and_reuses_tables(tmp_path, monkeypatch):
    from mfq.calibration.alphaq import ALPHAQ_PROFILES
    from mfq.calibration.natscale_errors import measure_errors
    from mfq.quantize.backend import resolve_quant_backend
    root, imatrix, _ = fixture(tmp_path, monkeypatch, expert_rows=8)
    matrix = load_importance_matrix(imatrix)
    plans = planned(root, None, CalibrationScheme(None, 'test', 0, {}, {}, {}), {}, 'native')
    scope = model_scope(plans, matrix, exclude_scaled_dense=True)
    cache = tmp_path/'measurement'
    errors = measure_errors(root, scope, matrix, cache, resolve_quant_backend('cpu', 'cpu'),
                            sample_layers=1, sample_experts=1)
    assert all(set(v) == set(ALPHAQ_PROFILES) for v in errors['1'].values())
    def forbidden(*args, **kwargs):
        raise AssertionError('complete error measurements must be reused')
    monkeypatch.setattr('mfq.calibration.natscale_errors.trained_tables', forbidden)
    assert measure_errors(root, scope, matrix, cache, resolve_quant_backend('cpu', 'cpu'),
                          sample_layers=1, sample_experts=1) == errors


def test_vision_nint8_and_predictor_exclusion_are_explicit_and_resumable(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path, monkeypatch)
    output, cache = tmp_path/'vision.mfq', tmp_path/'cache'
    argv = [*command(root, imatrix, cache, output), '--natscale-error', 'analytic',
            '--alphaq-vision', 'NINT8', '--alphaq-mtp', 'exclude']
    assert main(argv) == main([*argv, '--overwrite']) == 0
    with open_mmap(output) as store:
        assert store.records['vision.merger.projection.weight'].dtype == 'NINT'
        assert not any(n.startswith('predictor.') for n in store.records)
        assert store.header.extra['alphaq']['mtp_bytes'] == 0


def test_alphaq_joint_explicitly_keeps_scaled_dense_outside_allocation(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path/'joint.mfq'
    assert main(['quantize', str(root), str(output), '--imatrix', str(imatrix), '--target-bpw', '40',
        '--alphaq-dense', 'joint', '--alphaq-profiles', 'NINT4,NINT8', '--backend', 'cpu',
        '--alphaq-mtp', 'exclude', '--alphaq-vision', 'exclude']) == 0
    with open_mmap(output) as store:
        assert store.records['model.block.1.mlp.shared_expert.down.weight'].dtype == 'BF16'
        assert 'decoded BF16' in store.header.extra['alphaq']['joint_scaled_dense']


def test_invalid_natscale_modes_and_mutually_exclusive_caps_fail_early(tmp_path, monkeypatch):
    root, imatrix, _ = fixture(tmp_path, monkeypatch)
    argv = command(root, imatrix, tmp_path/'cache', tmp_path/'bad.mfq')
    with pytest.raises(ValueError, match='recipe dense mode'):
        main([*argv, '--alphaq-dense', 'joint'])
    with pytest.raises(ValueError, match='tau'):
        main([*argv, '--natscale-tau', '0'])
    with pytest.raises(SystemExit) as error:
        main([*argv, '--target-bpw', '3'])
    assert error.value.code == 2
    assert not (tmp_path/'cache/statistics.json').exists()
