#include "minicpmo45.h"

#include "models/minicpmo45/ops.h"
#include "cli.h"
#include "engine/options.h"

namespace mfq::cuda::minicpmo45 {

bool parse_command_option(
        std::string_view option,
        commands::ArgCursor& args,
        CommandOptions& result) {
    using commands::integer;
    using commands::usage_error;
    if (option == "--minicpmo-input-prefix") {
        result.input_prefix = args.value(option);
    } else if (option == "--minicpmo-output-prefix") {
        result.output_prefix = args.value(option);
    } else if (option == "--minicpmo-tts-steps") {
        result.tts_steps = integer<int64_t>(args.value(option), option);
        if (result.tts_steps < 0) {
            usage_error("--minicpmo-tts-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-input-prefix") {
        result.duplex_input_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-output-prefix") {
        result.duplex_output_prefix = args.value(option);
    } else if (option == "--minicpmo-duplex-steps") {
        result.duplex_steps = integer<int64_t>(args.value(option), option);
        if (result.duplex_steps < 0) {
            usage_error("--minicpmo-duplex-steps must be non-negative");
        }
    } else if (option == "--minicpmo-duplex-max-speak-tokens") {
        result.duplex_max_speak_tokens =
            integer<int64_t>(args.value(option), option);
        if (result.duplex_max_speak_tokens <= 0) {
            usage_error(
                "--minicpmo-duplex-max-speak-tokens must be positive");
        }
    } else if (option == "--minicpmo-duplex-seed") {
        result.duplex_seed = integer<int64_t>(args.value(option), option);
    } else if (option == "--minicpmo-duplex-greedy") {
        result.duplex_greedy = true;
    } else {
        return false;
    }
    return true;
}

void validate_command_options(const CommandOptions& options) {
    if (!options.output_prefix.empty() && options.input_prefix.empty()) {
        commands::usage_error(
            "--minicpmo-output-prefix requires --minicpmo-input-prefix");
    }
    if (!options.duplex_output_prefix.empty() &&
            options.duplex_input_prefix.empty()) {
        commands::usage_error(
            "--minicpmo-duplex-output-prefix requires "
            "--minicpmo-duplex-input-prefix");
    }
}

int run_composite(
        CudaExecutionContext& execution,
        const CudaLoadOptions& load,
        const CommandOptions& options) {
    return run_minicpmo45_composite(
        execution, load.model_path, load.config_path,
        options.input_prefix, options.output_prefix,
        load.context_size, options.tts_steps);
}

int run_duplex(
        CudaExecutionContext& execution,
        const CudaLoadOptions& load,
        const CommandOptions& options) {
    return run_minicpmo45_duplex(
        execution, load.model_path, load.config_path,
        options.duplex_input_prefix,
        options.duplex_output_prefix,
        load.context_size, options.duplex_steps,
        options.duplex_max_speak_tokens,
        options.duplex_greedy,
        options.duplex_seed);
}

int run_eval_batch(
        CudaExecutionContext& execution,
        const std::string& model_path,
        const std::string& config_path,
        int64_t context_size,
        int64_t vision_batch_size) {
    return run_minicpmo45_eval_batch(
        execution, model_path, config_path, context_size,
        vision_batch_size);
}

} // namespace mfq::cuda::minicpmo45
