from __future__ import annotations

from typing import Any

from mfq.server.protocol.models import ModelCacheComponent, ModelCacheFixedComponent, ModelCacheProfile


def cache_profile(config: dict[str, Any], *, include_recurrent: bool = True) -> ModelCacheProfile | None:
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
        fixed_components = []
        components = []

        def state(group: str, name: str, count: int, size: int) -> None:
            if count and size:
                fixed_components.append(ModelCacheFixedComponent(group=group, name=name, layers=count, bytes=size))

        if kind == 'deepseek_v4':
            dim, index, window = integer('head_dim'), integer('index_head_dim'), integer('sliding_window')
            ratios = text.get('compress_ratios')
            if min(dim, index, window) <= 0 or not isinstance(ratios, list) or len(ratios) < layers:
                return None
            state('MLA', 'sliding_window', layers, layers * window * dim * 2)
            for ratio in (4, 128):
                count = ratios[:layers].count(ratio)
                width = dim + index if ratio == 4 else dim
                state('MLA', f'compression_state_{ratio}', count, count * ratio * width * (12 if ratio == 4 else 4))
                if count:
                    components.append(ModelCacheComponent(group='MLA', name='compressed_kv', layers=count,
                        bytes_per_row=count * dim * 2, tokens_per_row=ratio))
                    if ratio == 4:
                        components.append(ModelCacheComponent(group='MLA', name='indexer_key', subgroup='indexer', layers=count,
                            bytes_per_row=count * index * 2, tokens_per_row=ratio))
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
            if linear and include_recurrent:
                keys, key_dim = integer('linear_num_key_heads'), integer('linear_key_head_dim')
                heads, value_dim = integer('linear_num_value_heads'), integer('linear_value_head_dim')
                kernel = integer('linear_conv_kernel_dim')
                if min(keys, key_dim, heads, value_dim, kernel) <= 0:
                    return None
                channels = 2 * keys * key_dim + heads * value_dim
                if key_dim != value_dim:
                    return None
                state_copies = 1 if kind in {'qwen4_exp', 'qwen4_exp_text'} else 2
                recurrent = linear * 4 * heads * value_dim * value_dim
                convolution = linear * 4 * (kernel - 1) * channels
                state('GDN', 'recurrent_state', linear, recurrent)
                state('GDN', 'convolution_state', linear, convolution)
                state('GDN', 'reset_template', linear, (state_copies - 1) * (recurrent + convolution))
            full = types.count('full_attention')
            if full:
                heads, dim = integer('num_key_value_heads'), integer('head_dim')
                if min(heads, dim) <= 0:
                    return None
                group = 'QSA' if kind in {'qwen4_exp', 'qwen4_exp_text'} else 'GQA'
                components.append(ModelCacheComponent(group=group, name='raw_kv', layers=full,
                    bytes_per_row=full * heads * dim * 4, allocation='power_of_two', minimum_rows=16))
            if kind in {'qwen4_exp', 'qwen4_exp_text'}:
                index, ratio = integer('indexer_head_dim'), integer('indexer_compress_ratio')
                if min(index, ratio) <= 0:
                    return None
                if full:
                    components.extend([
                        ModelCacheComponent(group='QSA', name='indexer_key', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 2, allocation='power_of_two', minimum_rows=16),
                        ModelCacheComponent(group='QSA', name='indexer_pooled', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 4, tokens_per_row=ratio, allocation='power_of_two', minimum_rows=16,
                            row_rounding='floor', active_after=integer('indexer_budget', 2048)),
                    ])
                ple_layers = text.get('ple_layer_ids', [])
                if not isinstance(ple_layers, list):
                    return None
                if ple_layers and include_recurrent:
                    kernel, ngram, hc, hidden = integer('ple_conv_kernel_size'), integer('ngram_size'), integer('hc_count'), integer('hidden_size')
                    if min(kernel, ngram, hc, hidden) <= 0:
                        return None
                    state('PLE', 'convolution_state', len(ple_layers), len(ple_layers) * 4 * (kernel - 1) * ngram * hc * hidden)
        elif kind in {'qwen2', 'qwen3', 'qwen3_moe', 'llama', 'mistral', 'minicpm', 'minicpmo'}:
            heads = integer('num_key_value_heads', integer('num_attention_heads'))
            dim = integer('head_dim', integer('hidden_size') // integer('num_attention_heads'))
            if min(heads, dim) <= 0 or text.get('sliding_window'):
                return None
            query_heads = integer('num_attention_heads')
            group = 'GQA' if query_heads > heads else 'MHA'
            components.append(ModelCacheComponent(group=group, name='raw_kv', layers=layers,
                bytes_per_row=layers * heads * dim * 4, allocation='power_of_two', minimum_rows=16))
        else:
            return None
        return ModelCacheProfile(max_context=maximum, fixed_bytes=sum(item.bytes for item in fixed_components),
            fixed_components=fixed_components, components=components)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
