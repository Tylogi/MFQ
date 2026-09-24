#include "cli.h"
#include "runtime/cuda_runtime.h"
#include "runtime/setup.h"
#include "transport.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace mfq::cuda::commands {
namespace {

void print_help() {
    std::cout
        << "MFQ native CUDA runtime\n\n"
        << "Usage:\n"
        << "  mfq-runtime --model MODEL --transport stdio|http [OPTIONS]\n"
        << "  mfq-runtime --model MODEL --ids IDS [--gen N] [OPTIONS]\n"
        << "  mfq-runtime --model MODEL --ids-file FILE [--gen N] [OPTIONS]\n"
        << "  mfq-runtime --model MODEL --minicpmo-input-prefix PREFIX "
           "--minicpmo-output-prefix PREFIX [OPTIONS]\n"
        << "  mfq-runtime --model MODEL --minicpmo-duplex-input-prefix PREFIX "
           "--minicpmo-duplex-output-prefix PREFIX [OPTIONS]\n\n"
        << "Runtime options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
        << "  --ids LIST                      token IDs\n"
        << "  --ids-file PATH                 raw int32 token IDs\n"
        << "  --gen N                         generated tokens (default 16)\n"
        << "  --transport TYPE                stdio or http\n"
        << "  --host HOST                     HTTP bind host (default 127.0.0.1)\n"
        << "  --port N                        HTTP port (default 8080)\n"
        << "  --model-name NAME               API model name\n"
        << "  --api-key KEY                   API key\n"
        << "  --sampling-profile PATH         sampling profile override\n"
        << "  --continuous-batching N         maximum concurrent sequences\n"
        << "  --prefill-chunk-size N          prefill chunk size (default 2048)\n"
        << "  --tokenizer PATH                external tokenizer GGUF\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
        << "  --moe-cache-profile PATH        MoE cache profile\n"
        << "  --tensor-parallel DEVICES       device count or comma-separated IDs\n"
        << "  --tensor-split WEIGHTS          tensor-parallel weights\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --expert-split WEIGHTS          expert-parallel weights\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  --layer-split WEIGHTS           layer-placement weights\n"
        << "  -h, --help                      show this help\n";
}

RuntimeOptions parse_runtime(ArgCursor& args) {
    RuntimeOptions result;
    bool transport_option = false;
    bool gen_option = false;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result)) continue;
        if (option == "--ids") {
            result.ids_arg = args.value(option);
            validate_integer_list(result.ids_arg, option);
        }
        else if (option == "--ids-file") result.ids_file = args.value(option);
        else if (option == "--gen") {
            result.gen = integer<int>(args.value(option), option);
            if (result.gen < 0) usage_error("--gen must be non-negative");
            gen_option = true;
        }
        else if (option == "--minicpmo-input-prefix") {
            result.minicpmo_input_prefix = args.value(option);
        }
        else if (option == "--minicpmo-output-prefix") {
            result.minicpmo_output_prefix = args.value(option);
        }
        else if (option == "--minicpmo-tts-steps") {
            result.minicpmo_tts_steps = integer<int64_t>(args.value(option), option);
            if (result.minicpmo_tts_steps < 0) {
                usage_error("--minicpmo-tts-steps must be non-negative");
            }
        }
        else if (option == "--minicpmo-duplex-input-prefix") {
            result.minicpmo_duplex_input_prefix = args.value(option);
        }
        else if (option == "--minicpmo-duplex-output-prefix") {
            result.minicpmo_duplex_output_prefix = args.value(option);
        }
        else if (option == "--minicpmo-duplex-steps") {
            result.minicpmo_duplex_steps = integer<int64_t>(args.value(option), option);
            if (result.minicpmo_duplex_steps < 0) {
                usage_error("--minicpmo-duplex-steps must be non-negative");
            }
        }
        else if (option == "--minicpmo-duplex-max-speak-tokens") {
            result.minicpmo_duplex_max_speak_tokens =
                integer<int64_t>(args.value(option), option);
            if (result.minicpmo_duplex_max_speak_tokens <= 0) {
                usage_error("--minicpmo-duplex-max-speak-tokens must be positive");
            }
        }
        else if (option == "--minicpmo-duplex-seed") {
            result.minicpmo_duplex_seed = integer<int64_t>(args.value(option), option);
        }
        else if (option == "--minicpmo-duplex-greedy") {
            result.minicpmo_duplex_greedy = true;
        }
        else if (option == "--transport") {
            if (result.transport_mode) {
                usage_error("runtime transport was specified more than once");
            }
            const std::string transport = args.value(option);
            if (transport != "stdio" && transport != "http") {
                usage_error("--transport must be stdio or http");
            }
            result.transport_mode = true;
            result.stdio_mode = transport == "stdio";
        }
        else if (option == "--host") {
            result.transport_host = args.value(option);
            transport_option = true;
        }
        else if (option == "--port") {
            result.transport_port = integer<int>(args.value(option), option);
            if (result.transport_port < 1 || result.transport_port > 65535) {
                usage_error("--port must be in [1, 65535]");
            }
            transport_option = true;
        }
        else if (option == "--continuous-batching") {
            result.continuous_batching = integer<int>(args.value(option), option);
            if (result.continuous_batching < 0) {
                usage_error("--continuous-batching must be non-negative");
            }
            transport_option = true;
        }
        else if (option == "--prefill-chunk-size") {
            result.prefill_chunk_size = integer<int64_t>(args.value(option), option);
            if (result.prefill_chunk_size <= 0) {
                usage_error("--prefill-chunk-size must be positive");
            }
            transport_option = true;
        }
        else if (option == "--model-name") {
            result.runtime_model_name = args.value(option);
            transport_option = true;
        }
        else if (option == "--api-key") {
            result.transport_api_key = args.value(option);
            transport_option = true;
        }
        else if (option == "--sampling-profile") {
            result.runtime_sampling_profile = args.value(option);
            transport_option = true;
        }
        else usage_error("unknown option: " + std::string(option));
    }

    if (result.model_path.empty()) usage_error("--model is required");
    if (!result.ids_arg.empty() && !result.ids_file.empty()) {
        usage_error("--ids and --ids-file are mutually exclusive");
    }
    if (!result.minicpmo_output_prefix.empty() &&
            result.minicpmo_input_prefix.empty()) {
        usage_error("--minicpmo-output-prefix requires --minicpmo-input-prefix");
    }
    if (!result.minicpmo_duplex_output_prefix.empty() &&
            result.minicpmo_duplex_input_prefix.empty()) {
        usage_error(
            "--minicpmo-duplex-output-prefix requires "
            "--minicpmo-duplex-input-prefix");
    }
    const bool token_mode = !result.ids_arg.empty() || !result.ids_file.empty();
    const int modes = static_cast<int>(result.transport_mode) +
        static_cast<int>(token_mode) +
        static_cast<int>(!result.minicpmo_input_prefix.empty()) +
        static_cast<int>(!result.minicpmo_duplex_input_prefix.empty());
    if (modes != 1) {
        usage_error("select exactly one execution mode: --transport, token input, "
                    "MiniCPM-o composite, or MiniCPM-o duplex");
    }
    if (gen_option && !token_mode) usage_error("--gen requires token input");
    if (transport_option && !result.transport_mode) {
        usage_error("transport options require --transport");
    }
    if (result.transport_mode && !result.config_path.empty()) {
        usage_error("--config cannot be used with --transport");
    }
    return result;
}

int execute_runtime(RuntimeOptions options) {
    return mfq::cuda::internal::with_command_errors([&]() -> int {
        if (options.stdio_mode) prepare_mfq_stdio_transport();
        mfq::cuda::internal::setup_cuda_load(options);
        if (!options.minicpmo_duplex_input_prefix.empty()) {
            return run_cuda_minicpmo_duplex(options);
        }
        if (!options.minicpmo_input_prefix.empty()) {
            return run_cuda_minicpmo_composite(options);
        }
        return run_cuda_inference(std::move(options));
    });
}

} // namespace

int run_runtime(int argc, char** argv) {
    try {
        ArgCursor args(argc, argv);
        auto options = parse_runtime(args);
        return execute_runtime(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    }
}

} // namespace mfq::cuda::commands
