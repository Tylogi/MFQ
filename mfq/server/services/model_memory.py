from __future__ import annotations

from typing import Any

from mfq.server.protocol.models import ModelCacheComponent, ModelCacheProfile


def cache_profile(config: dict[str, Any]) -> ModelCacheProfile | None:
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        return None
    kind = text.get('model_type', config.get('model_type'))

    def integer(key: str, default: int | None = None) -> int:
        value = text.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(key)
        return value

    try:
        maximum, layers = integer('max_position_embeddings'), integer('num_hidden_layers')
        if min(maximum, layers) <= 0:
            return None
        fixed = 0
        components = []
        if kind == 'deepseek_v4':
            dim, index, window = integer('head_dim'), integer('index_head_dim'), integer('sliding_window')
            ratios = text.get('compress_ratios')
            if min(dim, index, window) <= 0 or not isinstance(ratios, list) or len(ratios) < layers:
                return None
            fixed = layers * window * dim * 2
            for ratio in (4, 128):
                count = ratios[:layers].count(ratio)
                width = dim + index if ratio == 4 else dim
                fixed += count * ratio * width * (12 if ratio == 4 else 4)
                if count:
                    components.append(ModelCacheComponent(bytes_per_row=count * width * 2, tokens_per_row=ratio))
            if any(value not in (0, 4, 128) for value in ratios[:layers]):
                return None
        elif kind in {'qwen4_exp', 'qwen4_exp_text', 'qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next'}:
            types = text.get('layer_types')
            if not isinstance(types, list):
                interval = integer('full_attention_interval', 4)
                if interval <= 0:
                    return None
                types = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(layers)]
            if len(types) != layers or any(value not in ('linear_attention', 'full_attention') for value in types):
                return None
            linear = types.count('linear_attention')
            if linear:
                keys, key_dim = integer('linear_num_key_heads'), integer('linear_key_head_dim')
                heads, value_dim = integer('linear_num_value_heads'), integer('linear_value_head_dim')
                kernel = integer('linear_conv_kernel_dim')
                if min(keys, key_dim, heads, value_dim, kernel) <= 0:
                    return None
                channels = 2 * keys * key_dim + heads * value_dim
                if key_dim != value_dim:
                    return None
                state_copies = 1 if kind in {'qwen4_exp', 'qwen4_exp_text'} else 2
                fixed += state_copies * linear * 4 * (heads * value_dim * value_dim + (kernel - 1) * channels)
            full = types.count('full_attention')
            if full:
                heads, dim = integer('num_key_value_heads'), integer('head_dim')
                if min(heads, dim) <= 0:
                    return None
                components.append(ModelCacheComponent(bytes_per_row=full * heads * dim * 4, allocation='power_of_two', minimum_rows=16))
            if kind in {'qwen4_exp', 'qwen4_exp_text'}:
                index, ratio = integer('indexer_head_dim'), integer('indexer_compress_ratio')
                if min(index, ratio) <= 0:
                    return None
                if full:
                    components.extend([
                        ModelCacheComponent(bytes_per_row=full * index * 2, allocation='power_of_two', minimum_rows=16),
                        ModelCacheComponent(bytes_per_row=full * index * 4, tokens_per_row=ratio, allocation='power_of_two', minimum_rows=16),
                    ])
                ple_layers = text.get('ple_layer_ids', [])
                if not isinstance(ple_layers, list):
                    return None
                if ple_layers:
                    kernel, ngram, hc, hidden = integer('ple_conv_kernel_size'), integer('ngram_size'), integer('hc_count'), integer('hidden_size')
                    if min(kernel, ngram, hc, hidden) <= 0:
                        return None
                    fixed += len(ple_layers) * 4 * (kernel - 1) * ngram * hc * hidden
        elif kind in {'qwen2', 'qwen3', 'qwen3_moe', 'llama', 'mistral', 'minicpm', 'minicpmo'}:
            heads = integer('num_key_value_heads', integer('num_attention_heads'))
            dim = integer('head_dim', integer('hidden_size') // integer('num_attention_heads'))
            if min(heads, dim) <= 0 or text.get('sliding_window'):
                return None
            components.append(ModelCacheComponent(bytes_per_row=layers * heads * dim * 4, allocation='power_of_two', minimum_rows=16))
        else:
            return None
        if fixed < 0:
            return None
        return ModelCacheProfile(max_context=maximum, fixed_bytes=fixed, components=components)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
