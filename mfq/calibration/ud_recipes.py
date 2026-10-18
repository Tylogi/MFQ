"""Pinned whole-file UD ceilings and nonexpert policies, keyed by model config."""
from decimal import Decimal
from pathlib import Path


UD_MODELS = {
    'qwen4_exp_text': dict(
        config=dict(hidden_size=2560, num_hidden_layers=48, num_experts=512,
                    num_experts_per_tok=10, vocab_size=248320),
        parameters=176943899520, directory='', prefix='',
        caps=dict(IQ1_S=72546461344, IQ1_M=74538755776, Q2_K_XL=78869128864,
            IQ3_XXS=81961823936, Q3_K_XL=89986353824, IQ4_XS=93682584224,
            Q4_K_XL=111334654784, Q5_K_XL=158286406650, Q6_K_XL=169165382688)),
    'glm5_next_text': dict(
        config=dict(hidden_size=4096, num_hidden_layers=45, n_routed_experts=288,
                    num_experts_per_tok=8, vocab_size=154880),
        parameters=313326811966, directory='glm5_next', prefix='GLM-5.3-Flash-UD-',
        caps=dict(IQ1_S=93087245088, IQ1_M=97579344672, IQ4_XS=156822111200)),
}
UD_TIERS = tuple(dict.fromkeys(tier for model in UD_MODELS.values() for tier in model['caps']))


def registered_model(config):
    text = config.get('text_config', config)
    model = UD_MODELS.get(text.get('model_type'))
    if model is None or any(text.get(k) != v for k, v in model['config'].items()):
        return None
    return model


def automatic_recipe(config, target_bpw=None, *, tier=None):
    model = registered_model(config)
    if model is None:
        if tier:
            raise ValueError('--ud-tier requires a registered model configuration')
        return None
    if tier is None:
        target = Decimal(str(target_bpw))*model['parameters']/8
        tier = min(model['caps'], key=lambda name: abs(Decimal(model['caps'][name])-target))
    if tier not in model['caps']:
        raise ValueError(f'UD tier {tier} is unavailable for this model')
    return Path(__file__).with_name('recipes')/model['directory']/(model['prefix']+tier+'.json')
