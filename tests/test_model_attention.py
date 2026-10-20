import pytest

from mfq.server.services.model_attention import attention_profile


@pytest.mark.parametrize('config,expected', [
    ({'model_type': 'minicpmo'}, (False, False, False)),
    ({'model_type': 'qwen3_5_text', 'num_hidden_layers': 4}, (False, False, True)),
    ({'model_type': 'qwen3_5_text', 'layer_types': ['full_attention'] * 4}, (False, False, False)),
    ({'model_type': 'qwen4_exp_text', 'num_hidden_layers': 4, 'indexer_compress_ratio': 4}, (True, False, True)),
    ({'model_type': 'glm5_next_text', 'layer_types': ['linear_attention', 'deepseek_sparse_attention']}, (True, False, True)),
    ({'model_type': 'glm_moe_dsa'}, (True, False, False)),
    ({'model_type': 'deepseek_v4', 'compress_ratios': [4, 128]}, (True, True, False)),
    ({'model_type': 'deepseek_v41_text', 'compress_ratios': [0, 2, 1]}, (True, True, False)),
])
def test_attention_categories_follow_text_config(config, expected):
    result = attention_profile({'text_config': config, 'vision_config': {'model_type': 'vision'}})
    assert (result.sparse, result.compressed, result.hybrid) == expected


def test_unknown_architecture_is_not_guessed_from_a_model_name():
    assert attention_profile({'model_type': 'unknown'}) is None
