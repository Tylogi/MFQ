from __future__ import annotations

from typing import Any

from mfq.server.protocol.models import ModelAttentionProfile


def attention_profile(config: dict[str, Any]) -> ModelAttentionProfile | None:
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        return None
    kind = text.get('model_type', config.get('model_type'))
    types = text.get('layer_types')
    qwen_hybrid = kind in {'qwen4_exp', 'qwen4_exp_text', 'qwen3_5', 'qwen3_5_text',
        'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next'}
    if not isinstance(types, list) and qwen_hybrid:
        layers, interval = text.get('num_hidden_layers'), text.get('full_attention_interval', 4)
        if type(layers) is int and type(interval) is int and layers > 0 and interval > 0:
            types = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(layers)]
    linear = {'linear_attention', 'gdn', 'kda', 'mamba', 'recurrent_attention'}
    known_types = {'full_attention', 'sliding_attention', 'sparse_attention', 'deepseek_sparse_attention', 'gqa', 'mha', 'mla', 'qsa', 'dsa', 'csa', 'csa2'}
    if isinstance(types, list) and types:
        if any(not isinstance(value, str) or value not in linear | known_types for value in types):
            return None
        hybrid = any(value in linear for value in types) and any(value not in linear for value in types)
    else:
        if qwen_hybrid or kind in {'glm5_next', 'glm5_next_text'}:
            return None
        hybrid = False
    sparse_kinds = {'qwen4_exp', 'qwen4_exp_text', 'glm5_next', 'glm5_next_text',
        'glm_dsa', 'glm_moe_dsa', 'deepseek_v32', 'deepseek_v4', 'deepseek_v41', 'deepseek_v41_text'}
    dense_kinds = {'qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe', 'qwen3_5', 'qwen3_5_text',
        'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next', 'llama', 'mistral', 'mixtral',
        'minicpm', 'minicpmo', 'gemma', 'gemma2', 'gemma3', 'gemma3_text', 'deepseek_v2', 'deepseek_v3'}
    if kind not in sparse_kinds | dense_kinds and not types:
        return None
    sparse = kind in sparse_kinds or any(value in {'sparse_attention', 'qsa', 'dsa', 'csa', 'csa2'} for value in types or [])
    ratios = text.get('compress_ratios', [])
    compressed = kind in {'deepseek_v4', 'deepseek_v41', 'deepseek_v41_text'} or (
        isinstance(ratios, list) and any(type(value) is int and value > 1 for value in ratios))
    return ModelAttentionProfile(sparse=sparse, compressed=compressed, hybrid=hybrid)
