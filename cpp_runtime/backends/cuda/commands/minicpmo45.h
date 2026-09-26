#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace mfq::cuda {
struct CudaLoadOptions;
namespace commands { class ArgCursor; }

namespace minicpmo45 {

struct CommandOptions {
    std::string input_prefix, output_prefix;
    std::string duplex_input_prefix, duplex_output_prefix;
    std::int64_t tts_steps = 0;
    std::int64_t duplex_steps = 0;
    std::int64_t duplex_max_speak_tokens = 20;
    std::int64_t duplex_seed = 0;
    bool duplex_greedy = false;
};

bool parse_command_option(
    std::string_view option,
    commands::ArgCursor& args,
    CommandOptions& result);
void validate_command_options(const CommandOptions& options);

int run_composite(
    const CudaLoadOptions& load,
    const CommandOptions& options);
int run_duplex(
    const CudaLoadOptions& load,
    const CommandOptions& options);
int run_eval_batch(
    const std::string& model_path,
    const std::string& config_path,
    std::int64_t context_size,
    std::int64_t vision_batch_size);

} // namespace minicpmo45
} // namespace mfq::cuda
