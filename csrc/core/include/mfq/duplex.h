#pragma once

#include <cstdint>
#include <vector>

namespace mfq {

struct DuplexSessionParams {
    std::vector<int64_t> system_prefix;
    std::vector<int64_t> system_suffix;
    std::vector<float> reference_audio_features;
    int32_t reference_audio_frames = 0;
    std::vector<int64_t> special_ids;
    std::vector<int64_t> forbidden_ids;
    bool greedy = false;
    double temperature = 0.7;
    int32_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int32_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;
    uint64_t seed = 0;
};

struct DuplexStepInput {
    std::vector<float> audio_features;
    std::vector<int64_t> text_tokens;
    int32_t audio_frames = 0;
    int64_t audio_prefix_extra_frames = 0;
    int64_t audio_suffix_extra_frames = 0;
    int32_t max_new_speak_tokens = 20;
    bool force_listen = false;
    bool force_speak = false;
};

struct DuplexStepResult {
    std::vector<int64_t> generated_tokens;
    std::vector<int32_t> audio_tokens;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
    int64_t audio_chunk_index = 0;
    int64_t language_cache_position = 0;
    int64_t audio_cache_position = 0;
    int64_t tts_cache_position = 0;
    double inference_ms = 0.0;
};

} // namespace mfq
