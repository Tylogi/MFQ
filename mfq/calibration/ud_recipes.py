"""Pinned whole-file UD ceilings and nonexpert policies, keyed by model config."""
from decimal import Decimal
from pathlib import Path

from mfq._alphaq_profiles import UD_MODELS, UD_TIERS

__all__ = ['UD_MODELS', 'UD_TIERS', 'registered_model', 'automatic_recipe']


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
