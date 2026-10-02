import json
from decimal import Decimal

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mfq.cli import main
from mfq.calibration import alphaq_cache
from mfq.formats.io import open_mmap
from mfq.quantize.imatrix import ImportanceEntry, save_importance_matrix


def fixture(tmp_path):
    root=tmp_path/'model'; root.mkdir()
    (root/'config.json').write_text(json.dumps(dict(model_type='qwen4_exp',text_config=dict(
        model_type='qwen4_exp_text',num_hidden_layers=1,num_experts=3,num_experts_per_tok=1))))
    generator=torch.Generator().manual_seed(73)
    tensors={
        'model.language_model.layers.0.mlp.experts.gate_up_proj':torch.randn(3,16,128,generator=generator).bfloat16(),
        'model.language_model.layers.0.mlp.experts.down_proj':torch.randn(3,8,128,generator=generator).bfloat16(),
        'model.language_model.layers.0.mlp.gate.weight':torch.randn(3,128,generator=generator).float(),
        'model.language_model.layers.0.mlp.shared_expert.down_proj.weight':torch.randn(8,128,generator=generator).bfloat16(),
        'lm_head.weight':torch.randn(8,128,generator=generator).bfloat16(),
    }
    save_file(tensors,root/'model.safetensors')
    entries={f'blk.0.ffn_{p}_exps.weight':ImportanceEntry(np.ones((3,128)),np.array([7,2,1]))
             for p in ('gate','up','down')}
    entries['blk.0.ffn_gate_inp.weight']=ImportanceEntry(np.ones((1,128)),np.array([10]))
    entries['blk.0.ffn_down_shexp.weight']=ImportanceEntry(np.ones((1,128)),np.array([10]))
    imatrix=tmp_path/'imat.npz';save_importance_matrix(imatrix,entries)
    return root,imatrix,sum(t.numel() for t in tensors.values())


def test_normal_command_outputs_capped_model_and_reuses_all_statistics(tmp_path,monkeypatch):
    root,imatrix,count=fixture(tmp_path)
    output=tmp_path/'output.mfq';cache=tmp_path/'cache'
    argv=['quantize',str(root),str(output),'--target-bpw','12','--imatrix',str(imatrix),
          '--alphaq-cache',str(cache),'--alphaq-profiles','NINT4,NINT8','--backend','cpu','--device','cpu']
    assert main(argv)==0
    assert output.stat().st_size <= int(Decimal(12)*count/8)
    store=open_mmap(output)
    try:
        assert store.header.extra['alphaq']['model_weight_count']==count
        assert store.header.extra['alphaq']['dense_mode']=='joint'
        assert store.records['model.output.weight'].dtype=='BF16'
        assert all(store.records[f'model.block.0.mlp.experts.{p}.weight'].dtype=='MFE'
                   for p in ('gate','up','down'))
    finally:
        store.close()
    def forbidden(*args,**kwargs):
        raise AssertionError('completed statistics must not be recomputed')
    monkeypatch.setattr(alphaq_cache,'_collect',forbidden)
    assert main([*argv,'--overwrite'])==0
    stats=json.loads((cache/'statistics.json').read_text())
    stats['identity']['method']['gamma']=2
    (cache/'statistics.json').write_text(json.dumps(stats))
    with pytest.raises(ValueError,match='another model or method'):
        main([*argv,'--overwrite'])


def test_recipe_mode_keeps_missing_imatrix_recipe_precision(tmp_path):
    root,imatrix,count=fixture(tmp_path)
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps({'format':'mfq.gguf-recipe.v1','tensor_types':{
        **{f'blk.0.ffn_{p}_exps.weight':'IQ2_XXS' for p in ('gate','up','down')},
        'blk.0.ffn_gate_inp.weight':'F32','blk.0.ffn_down_shexp.weight':'Q8_0','output.weight':'Q8_0'}}))
    output=tmp_path/'recipe.mfq'
    assert main(['quantize',str(root),str(output),'--target-bpw','12','--imatrix',str(imatrix),
        '--recipe',str(recipe),'--alphaq-profiles','NINT4,NINT8','--backend','cpu'])==0
    store=open_mmap(output)
    try:
        assert store.header.extra['alphaq']['dense_mode']=='recipe'
        assert store.records['model.output.weight'].dtype=='NINT'
        assert output.stat().st_size <= int(Decimal(12)*count/8)
    finally:
        store.close()


def test_auto_recipe_is_restricted_to_evaluated_model():
    from mfq.calibration.alphaq_workflow import automatic_recipe,recipe_types
    config=dict(model_type='qwen4_exp_text',hidden_size=2560,num_hidden_layers=48,
        num_experts=512,num_experts_per_tok=10,vocab_size=248320)
    path=automatic_recipe({'text_config':config},78869128864*8/176943899520)
    assert path.stem=='Q2_K_XL'
    assert all(v=='Q8_0' for k,v in recipe_types(path).items() if k.endswith('_exps.weight'))
    assert automatic_recipe({**config,'num_hidden_layers':49},3.56) is None


def test_unavailable_console_does_not_stop_normal_output(tmp_path,monkeypatch):
    root,imatrix,_=fixture(tmp_path)
    def denied(*args,**kwargs):
        raise PermissionError('optional console denied')
    monkeypatch.setattr('builtins.print',denied)
    output=tmp_path/'quiet.mfq'
    assert main(['quantize',str(root),str(output),'--target-bpw','12','--imatrix',str(imatrix),
        '--alphaq-profiles','NINT4,NINT8','--backend','cpu'])==0
    assert output.stat().st_size > 0


def test_required_statistics_write_failure_propagates(tmp_path,monkeypatch):
    root,imatrix,_=fixture(tmp_path)
    def denied(*args,**kwargs):
        raise PermissionError('required checkpoint denied')
    monkeypatch.setattr(alphaq_cache,'_atomic_json',denied)
    with pytest.raises(PermissionError,match='required checkpoint'):
        main(['quantize',str(root),str(tmp_path/'failed.mfq'),'--target-bpw','12',
              '--imatrix',str(imatrix),'--alphaq-profiles','NINT4,NINT8','--backend','cpu'])
    assert not (tmp_path/'failed.mfq').exists()
