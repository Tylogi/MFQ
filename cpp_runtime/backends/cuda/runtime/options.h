#pragma once

#include <cstdint>
#include <string>

namespace mfq::cuda {

struct CudaLoadOptions {
    std::string model_path, config_path, tokenizer_model;
    std::string cpu_offload_layers_arg, moe_cache_profile_path;
    std::string tensor_parallel_arg, tensor_split_arg;
    std::string expert_parallel_arg, expert_split_arg;
    std::string layer_parallel_arg, layer_split_arg;
    double moe_gpu_cache_gb = 0.0;
    int cpu_threads = 0;
    int64_t context_size = 0;
    bool parallel_test_duplicates = false;
    bool n_gpu_layers_set = false;
    bool cpu_threads_set = false;
    int n_gpu_layers = -1;
};

struct TokenInputOptions {
    std::string ids_arg, ids_file;
    int gen = 16;
};

struct RuntimeOptions : CudaLoadOptions, TokenInputOptions {
    std::string minicpmo_input_prefix, minicpmo_output_prefix;
    std::string minicpmo_duplex_input_prefix, minicpmo_duplex_output_prefix;
    std::string transport_host = "127.0.0.1";
    std::string runtime_model_name = "mfq-model", transport_api_key;
    std::string runtime_sampling_profile;
    int transport_port = 8080;
    int continuous_batching = 0;
    int64_t prefill_chunk_size = 2048;
    int64_t minicpmo_tts_steps = 0;
    int64_t minicpmo_duplex_steps = 0;
    int64_t minicpmo_duplex_max_speak_tokens = 20;
    int64_t minicpmo_duplex_seed = 0;
    bool transport_mode = false;
    bool stdio_mode = false;
    bool minicpmo_duplex_greedy = false;
};

} // namespace mfq::cuda
