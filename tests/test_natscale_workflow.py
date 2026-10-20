"""Synthetic end-to-end coverage for the architecture-neutral NatScale workflow."""

import json

import numpy as np
import torch
from safetensors.torch import save_file

from mfq.cli import main
from mfq.formats.io import open_mmap
from mfq.quantize.imatrix import ImportanceEntry, save_importance_matrix


def synthetic_routed_model(tmp_path):
    root = tmp_path/'model'
    root.mkdir()
    (root/'config.json').write_text(json.dumps({
        'model_type': 'qwen4_exp',
        'text_config': {
            'model_type': 'qwen4_exp_text',
            'num_hidden_layers': 1,
            'num_experts': 3,
            'num_experts_per_tok': 1,
        },
    }))
    generator = torch.Generator().manual_seed(739)
    tensors = {
        'model.language_model.layers.0.mlp.experts.gate_up_proj': torch.randn(
            3, 16, 128, generator=generator
        ).bfloat16(),
        'model.language_model.layers.0.mlp.experts.down_proj': torch.randn(
            3, 128, 8, generator=generator
        ).bfloat16(),
        'model.language_model.layers.0.mlp.gate.weight': torch.randn(
            3, 128, generator=generator
        ).float(),
        'model.language_model.layers.0.mlp.shared_expert.down_proj.weight': torch.randn(
            128, 8, generator=generator
        ).bfloat16(),
        'lm_head.weight': torch.randn(8, 128, generator=generator).bfloat16(),
    }
    save_file(tensors, root/'model.safetensors')
    entries = {
        **{f'blk.0.ffn_{projection}_exps.weight': ImportanceEntry(
            np.ones((3, 8 if projection == 'down' else 128)), np.array([7, 2, 1])
        ) for projection in ('gate', 'up', 'down')},
        'blk.0.ffn_gate_inp.weight': ImportanceEntry(np.ones((1, 128)), np.array([10])),
        'blk.0.ffn_down_shexp.weight': ImportanceEntry(np.ones((1, 8)), np.array([10])),
    }
    imatrix = tmp_path/'imat.npz'
    save_importance_matrix(imatrix, entries)
    return root, imatrix, sum(tensor.numel() for tensor in tensors.values())


def _recipe(path):
    path.write_text(json.dumps({
        'format': 'mfq.gguf-recipe.v1',
        'ud_file_bytes': 100_000,
        'tensor_types': {
            **{f'blk.0.ffn_{projection}_exps.weight': 'IQ2_XXS'
               for projection in ('gate', 'up', 'down')},
            'blk.0.ffn_gate_inp.weight': 'F32',
            'blk.0.ffn_down_shexp.weight': 'Q8_0',
            'output.weight': 'Q8_0',
        },
    }))


def test_separate_expert_source_selects_indexed_rows(tmp_path):
    from mfq.tools.quantize_hf_to_mfq import _SeparateExpertRowSource

    root = tmp_path/'separate-experts'
    root.mkdir()
    first = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    second = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    save_file({'expert-0': first, 'expert-1': second}, root/'model.safetensors')
    source = _SeparateExpertRowSource(
        root, (2, 3, 2), (('expert-0',), ('expert-1',)),
        (('model.safetensors',), ('model.safetensors',)),
    )
    try:
        rows = source.read_rows(np.array([0, 4, 2]), device='cpu')
    finally:
        source.close()
    torch.testing.assert_close(rows, torch.stack((first[0], second[1], first[2])))


def test_natscale_ud_tier_workflow_uses_synthetic_registered_recipe(tmp_path, monkeypatch):
    root, imatrix, parameters = synthetic_routed_model(tmp_path)
    recipe = tmp_path/'synthetic-recipe.json'
    _recipe(recipe)
    model = {'parameters': parameters, 'caps': {'IQ1_S': 100_000}}
    monkeypatch.setattr(
        'mfq.calibration.alphaq_workflow.automatic_recipe',
        lambda *_args, **_kwargs: recipe,
    )
    monkeypatch.setattr(
        'mfq.calibration.alphaq_workflow.registered_model', lambda _config: model
    )
    cache, output = tmp_path/'cache', tmp_path/'natscale.mfq'
    common = [
        '--model', str(root), '--imatrix', str(imatrix), '--cache', str(cache),
        '--backend', 'cpu', '--device', 'cpu',
    ]

    assert main(['calibrate', 'natscale-statistics', *common]) == 0
    assert main(['calibrate', 'natscale-errors', *common, '--profiles', 'NINT4,NINT8',
                 '--sample-layers', '1', '--sample-experts', '1']) == 0
    assert main([
        'quantize', str(root), str(output), '--imatrix', str(imatrix),
        '--ud-tier', 'IQ1_S', '--alphaq-allocator', 'natscale',
        '--alphaq-profiles', 'NINT4,NINT8', '--alphaq-cache', str(cache),
        '--natscale-sample-layers', '1', '--natscale-sample-experts', '1',
        '--backend', 'cpu', '--device', 'cpu',
    ]) == 0

    with open_mmap(output) as store:
        policy = store.header.extra['alphaq']
        assert policy['allocator'] == 'natscale'
        assert policy['ud_tier'] == 'IQ1_S'
        assert output.stat().st_size <= 100_000
        assert all(store.records[f'model.block.0.mlp.experts.{projection}.weight'].dtype == 'MFE'
                   for projection in ('gate', 'up', 'down'))
