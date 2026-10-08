from copy import deepcopy

import pytest

from mfq.server.services.model_parameters import parameter_breakdown


def flash_config():
    return {
        'model_type': 'qwen4_exp', 'language_model_only': False,
        'text_config': {
            'model_type': 'qwen4_exp_text', 'hidden_size': 2560, 'num_hidden_layers': 48,
            'vocab_size': 248320, 'moe_intermediate_size': 640, 'num_experts': 512,
            'num_experts_per_tok': 10, 'shared_expert_intermediate_size': 640,
            'hc_count': 4, 'hc_lowrank': 320, 'layer_types': ['linear_attention'] * 3 + ['full_attention'],
            'linear_num_value_heads': 48, 'linear_value_head_dim': 128,
            'linear_num_key_heads': 16, 'linear_key_head_dim': 128, 'linear_conv_kernel_dim': 4,
            'head_dim': 256, 'num_attention_heads': 24, 'num_key_value_heads': 2,
            'indexer_head_dim': 128, 'indexer_n_heads': 4, 'indexer_kv_heads': 1,
            'ple_layer_ids': [2], 'ple_embed_dim': 2560, 'ngram_vocab_size_base': 20000000,
            'ple_conv_kernel_size': 4,
        },
        'vision_config': {
            'hidden_size': 1152, 'intermediate_size': 4304, 'depth': 27,
            'in_channels': 3, 'temporal_patch_size': 2, 'patch_size': 16,
            'spatial_merge_size': 2, 'out_hidden_size': 2560,
            'num_position_embeddings': 2304, 'deepstack_visual_indexes': [],
        },
    }


def test_flash_next_counts_match_original_tensor_shapes_without_quantization_sidecars():
    config = flash_config()
    config['text_config']['layer_types'] *= 12
    result = parameter_breakdown(config)
    assert result.routed_experts == 120795955200
    assert result.dense == 5396629616
    assert result.ple == 51200000000
    assert result.total == result.dense + result.routed_experts + result.ple
    assert result.active == 6035596160
    config['text_config']['mtp_num_hidden_layers'] = 3
    config['text_config']['quantization_config'] = {'bits': 2}
    assert parameter_breakdown(config) == result
    config['language_model_only'] = True
    text_only = parameter_breakdown(config)
    assert result.dense - text_only.dense == 448931056
    assert text_only.active == result.active


def test_deepseek_v4_main_model_excludes_dspark_extra_layers_and_packed_fp4_storage():
    config = {
        'model_type': 'deepseek_v4', 'hidden_size': 4096, 'num_hidden_layers': 43,
        'vocab_size': 129280, 'moe_intermediate_size': 2048, 'n_routed_experts': 256,
        'num_experts_per_tok': 6, 'n_shared_experts': 1, 'head_dim': 512,
        'num_attention_heads': 64, 'q_lora_rank': 1024, 'o_lora_rank': 1024, 'o_groups': 8,
        'hc_mult': 4, 'index_head_dim': 128, 'index_n_heads': 64,
        'compress_ratios': [0, 0] + [r for _ in range(20) for r in (4, 128)] + [4, 0, 0, 0],
    }
    result = parameter_breakdown(config)
    assert result.routed_experts == 277025390592
    assert 284e9 < result.total < 285e9
    assert 12e9 < result.active < 14e9
    assert result.ple == 0
    assert result.total == result.dense + result.routed_experts
    config['expert_dtype'] = 'fp4'
    config['num_nextn_predict_layers'] = 1
    config['dspark_target_layer_ids'] = [40, 41, 42]
    assert parameter_breakdown(config) == result


@pytest.mark.parametrize('change', [
    {'hidden_size': -1}, {'hidden_size': True}, {'num_experts_per_tok': 513},
    {'layer_types': []}, {'layer_types': ['unsupported'] * 48}, {'head_dim': None},
])
def test_incomplete_or_invalid_metadata_never_fabricates_counts(change):
    config = deepcopy(flash_config())
    config['text_config']['layer_types'] *= 12
    config['text_config'].update(change)
    assert parameter_breakdown(config) is None
    assert parameter_breakdown({'model_type': 'unknown', 'parameter_count': 100}) is None
