"""Pinned recipes and imatrix headers bind the complete GLM text/predictor plan."""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from mfq.architectures.tensor_schema import glm5_recipe_name
from mfq.calibration.alphaq_workflow import recipe_types
from mfq.calibration.artifact import CalibrationScheme
from mfq.calibration.ud_recipes import automatic_recipe, registered_model
from mfq.tools import quantize_hf_to_mfq as q


def header_fixture():
    doc = json.loads((Path(__file__).parent/'fixtures/glm5_next_headers.json').read_text())
    config, inventory = doc['config'], {}
    for item in doc['templates']:
        for layer in item['layers'] or [0]:
            experts = range(288) if '{expert}' in item['name'] else [0]
            for expert in experts:
                name = item['name'].format(layer=layer, expert=expert)
                inventory[name] = q.SourceTensorMetadata(name, 'header-only.safetensors',
                                                       tuple(item['shape']), item['dtype'])
    return config, inventory, doc['imatrix_names']


@pytest.mark.parametrize('tier', ['IQ1_S', 'IQ1_M', 'IQ4_XS'])
def test_every_pinned_recipe_tensor_and_imatrix_entry_binds_once(tier):
    config, inventory, names = header_fixture()
    path = automatic_recipe(config, tier=tier)
    raw = json.loads(path.read_text())
    plans = q._plan(Path('.'), True, recipe_types(path, text_layers=45), 'F32',
        CalibrationScheme(None, 'test', 0, {}, {}, {}), quantize_mtp=True, quantize_ple=True,
        source_inventory=inventory, source_config=config)
    aliases, imatrix = defaultdict(list), defaultdict(list)
    for item in plans:
        alias = glm5_recipe_name(item.name)
        if alias:
            aliases[alias].append(item.name)
        for alias in q._hf_imatrix_names(item):
            imatrix[alias].append(item.name)
    assert {k: v for k,v in aliases.items() if len(v) != 1} == {}
    assert set(raw['tensor_types']) == set(aliases)
    assert {name: imatrix.get(name, []) for name in names if len(imatrix.get(name, [])) != 1} == {}
    # This integer runtime constant is preserved but has no GGUF float policy.
    assert {p.name for p in plans if glm5_recipe_name(p.name) is None} <= {'model.runtime.hash_metadata'}
    count = sum(__import__('math').prod(p.shape) for p in plans
                if p.name.startswith('model.') and p.source_dtype not in {'I32', 'I64'})
    assert count == registered_model(config)['parameters']
    assert raw['ud_file_bytes'] == registered_model(config)['caps'][tier]


def test_model_registry_rejects_other_shapes_and_preserves_qwen_selection():
    config, _, _ = header_fixture()
    assert automatic_recipe(config, tier='IQ1_S').name == 'GLM-5.3-Flash-UD-IQ1_S.json'
    config['text_config']['n_routed_experts'] = 287
    assert automatic_recipe(config, 3) is None
    with pytest.raises(ValueError, match='registered'):
        automatic_recipe(config, tier='IQ1_S')
    qwen = dict(model_type='qwen4_exp_text', hidden_size=2560, num_hidden_layers=48,
        num_experts=512, num_experts_per_tok=10, vocab_size=248320)
    assert automatic_recipe(qwen, 78869128864*8/176943899520).stem == 'Q2_K_XL'
