from mfq.server.services.model_memory import cache_profile


def test_flash_next_profile_includes_gdn_ple_and_qsa_index_caches():
    config = {'model_type': 'qwen4_exp_text', 'max_position_embeddings': 1048576,
              'num_hidden_layers': 48, 'layer_types': ['linear_attention'] * 3 + ['full_attention'],
              'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
              'linear_num_value_heads': 48, 'linear_value_head_dim': 128, 'linear_conv_kernel_dim': 4,
              'num_key_value_heads': 2, 'head_dim': 256, 'indexer_head_dim': 128,
              'indexer_compress_ratio': 4, 'ple_layer_ids': [2], 'ple_conv_kernel_size': 4,
              'ngram_size': 3, 'hc_count': 4, 'hidden_size': 2560}
    config['layer_types'] *= 12
    profile = cache_profile(config)
    assert profile.fixed_bytes == 36 * (48 * 128 * 128 + 3 * 10240) * 4 + 9 * 10240 * 4
    assert [component.bytes_per_row for component in profile.components] == [24576, 3072, 6144]
    assert [component.tokens_per_row for component in profile.components] == [1, 1, 4]
    assert all(component.allocation == 'power_of_two' and component.minimum_rows == 16 for component in profile.components)
    assert cache_profile({'text_config': config}) == profile


def test_deepseek_compressed_latents_are_not_double_counted_as_full_k_and_v():
    config = {'model_type': 'deepseek_v4', 'max_position_embeddings': 1048576, 'num_hidden_layers': 43,
              'head_dim': 512, 'index_head_dim': 128, 'sliding_window': 128,
              'compress_ratios': [0, 0] + [r for _ in range(20) for r in (4, 128)] + [4, 0, 0, 0]}
    profile = cache_profile(config)
    assert profile.fixed_bytes == 43 * 128 * 512 * 2 + 21 * 4 * 640 * 12 + 20 * 128 * 512 * 4
    assert [(component.bytes_per_row, component.tokens_per_row, component.allocation) for component in profile.components] == [(26880, 4, 'exact'), (20480, 128, 'exact')]


def test_regular_gqa_uses_kv_heads_and_two_byte_storage():
    profile = cache_profile({'model_type': 'llama', 'num_hidden_layers': 32, 'num_attention_heads': 32,
                             'num_key_value_heads': 8, 'hidden_size': 4096, 'max_position_embeddings': 8192})
    assert profile.fixed_bytes == 0
    assert profile.components[0].bytes_per_row == 32 * 8 * 128 * 2 * 2


def test_incomplete_unknown_and_invalid_configs_do_not_fabricate_cache_estimates():
    for config in ({}, {'model_type': 'llama'}, {'model_type': 'unknown', 'max_position_embeddings': 4096, 'num_hidden_layers': 32},
                   {'model_type': 'llama', 'max_position_embeddings': -1, 'num_hidden_layers': 32},
                   {'model_type': 'llama', 'max_position_embeddings': 4096, 'num_hidden_layers': True}):
        assert cache_profile(config) is None
