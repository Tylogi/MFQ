from __future__ import annotations

from typing import Any

from mfq.server.protocol.models import ModelParameterBreakdown


def parameter_breakdown(config: dict[str, Any]) -> ModelParameterBreakdown | None:
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        return None
    kind = text.get('model_type', config.get('model_type'))
    if kind not in {'qwen4_exp', 'qwen4_exp_text', 'deepseek_v4'}:
        return None

    def integer(key: str, default: int | None = None) -> int:
        value = text.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(key)
        return value

    try:
        h, layers = integer('hidden_size'), integer('num_hidden_layers')
        vocab, intermediate = integer('vocab_size'), integer('moe_intermediate_size')
        experts = integer('num_experts' if str(kind).startswith('qwen') else 'n_routed_experts')
        topk = integer('num_experts_per_tok')
        if min(h, layers, vocab, intermediate, experts, topk) <= 0 or topk > experts:
            return None
        embeddings = h * vocab * (1 if text.get('tie_word_embeddings', config.get('tie_word_embeddings', False)) else 2)
        routed = layers * experts * 3 * h * intermediate
        dense, ple, vision = embeddings, 0, 0
        if str(kind).startswith('qwen'):
            hc, rank = integer('hc_count'), integer('hc_lowrank')
            width = hc * h
            mixer = 2 * width * rank + width
            branch = mixer + hc * width
            common = 2 * branch + h * experts + 3 * h * integer('shared_expert_intermediate_size') + h
            dense += mixer
            types = text.get('layer_types')
            if not isinstance(types, list) or len(types) != layers or text.get('attention_bias', False):
                return None
            for layer in types:
                if layer == 'linear_attention':
                    heads = integer('linear_num_value_heads')
                    v = heads * integer('linear_value_head_dim')
                    channels = 2 * integer('linear_num_key_heads') * integer('linear_key_head_dim') + v
                    attention = h * (channels + 2 * v + 2 * heads)
                    attention += channels * integer('linear_conv_kernel_dim') + 2 * heads + integer('linear_value_head_dim')
                elif layer == 'full_attention':
                    dim = integer('head_dim')
                    q, kv = integer('num_attention_heads') * dim, integer('num_key_value_heads') * dim
                    index_dim = integer('indexer_head_dim')
                    index = (integer('indexer_n_heads') + integer('indexer_kv_heads')) * index_dim
                    attention = h * (3 * q + 2 * kv + index) + 2 * dim + 2 * index_dim
                else:
                    return None
                dense += common + attention
            ple_layers = text.get('ple_layer_ids', [])
            if not isinstance(ple_layers, list):
                return None
            if ple_layers:
                embedding = integer('ple_embed_dim')
                ple = len(ple_layers) * integer('ngram_vocab_size_base') * embedding
                dense += len(ple_layers) * (embedding * (width + h) + 3 * width + width * integer('ple_conv_kernel_size'))
            visual = config.get('vision_config')
            if isinstance(visual, dict) and not config.get('language_model_only', False):
                def vget(key: str) -> int:
                    value = visual.get(key)
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        raise ValueError(key)
                    return value
                vh, vi = vget('hidden_size'), vget('intermediate_size')
                merged = vh * vget('spatial_merge_size') ** 2
                output = vget('out_hidden_size')
                vision = vh * vget('in_channels') * vget('temporal_patch_size') * vget('patch_size') ** 2 + vh
                vision += vget('num_position_embeddings') * vh
                vision += vget('depth') * (4 * vh * vh + 2 * vh * vi + 9 * vh + vi)
                merger = merged * merged + merged + output * merged + output + 2 * vh
                vision += merger
                deepstack = visual.get('deepstack_visual_indexes', [])
                if not isinstance(deepstack, list):
                    return None
                vision += len(deepstack) * (merger - 2 * vh + 2 * merged)
                dense += vision
        else:
            dim, heads = integer('head_dim'), integer('num_attention_heads')
            qr, rank, groups = integer('q_lora_rank'), integer('o_lora_rank'), integer('o_groups')
            hc = integer('hc_mult')
            metadata = hc * hc + 2 * hc
            dense += h + hc * hc * h + hc + 1
            common = 2 * h + 2 * (metadata * hc * h + metadata + 3)
            common += h * experts + 3 * h * intermediate * integer('n_shared_experts')
            attention = h * (qr + dim + groups * rank) + heads * dim * (qr + rank) + qr + dim + heads
            ratios = text.get('compress_ratios')
            if not isinstance(ratios, list) or len(ratios) < layers:
                return None
            for ratio in ratios[:layers]:
                if ratio not in {0, 4, 128}:
                    return None
                dense += common + attention
                if ratio:
                    width = dim * (2 if ratio == 4 else 1)
                    dense += 2 * width * h + ratio * width + dim
                if ratio == 4:
                    index, index_heads = integer('index_head_dim'), integer('index_n_heads')
                    dense += qr * index_heads * index + 4 * index * h + index_heads * h + 8 * index + index
        active = dense - embeddings - vision + layers * topk * 3 * h * intermediate
        return ModelParameterBreakdown(
            total=dense + routed + ple, dense=dense, routed_experts=routed, ple=ple, active=active,
        )
    except (ValueError, TypeError, OverflowError):
        return None
