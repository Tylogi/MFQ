#include "cli.h"
#include "runtime/cuda_engine.h"
#include "runtime/moe_expert_cache.h"
#include "runtime/setup.h"
#include "runtime/token_generation.h"
#include "minicpmo45.h"
#include "models/registry.h"
#include "mfq/model_source.h"
#include "transport.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

struct RuntimeCommandOptions
    : RuntimeOptions,
      mfq::cuda::minicpmo45::CommandOptions {};

RuntimeCommandOptions parse_runtime(ArgCursor& args) {
    RuntimeCommandOptions result;
    bool transport_option = false;
    bool gen_option = false;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result)) continue;
        if (mfq::cuda::minicpmo45::parse_command_option(
                option, args, result)) {
            continue;
        }
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
    mfq::cuda::minicpmo45::validate_command_options(result);
    const bool token_mode = !result.ids_arg.empty() || !result.ids_file.empty();
    const int modes = static_cast<int>(result.transport_mode) +
        static_cast<int>(token_mode) +
        static_cast<int>(!result.input_prefix.empty()) +
        static_cast<int>(!result.duplex_input_prefix.empty());
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

std::vector<uint8_t> read_runtime_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_runtime_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    if (bytes.empty()) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

int run_transport_runtime(RuntimeOptions& options) {
    auto source = mfq::open_model_source(options.model_path);
    if (!options.config_path.empty()) {
        throw std::runtime_error(
            "model runtime does not accept an external model config");
    }
    if (!source->has_asset(mfq::kModelConfigAsset) ||
            (!source->has_asset(mfq::cuda::kTokenizerGgufAsset) &&
             options.tokenizer_model.empty())) {
        throw std::runtime_error(
            "model runtime requires model config and tokenizer GGUF");
    }

    CudaEngineOptions engine_options = options;
    auto loaded = load_cuda_engine(std::move(engine_options));
    if (options.transport_api_key.empty()) {
        const char* env_key = std::getenv("MFQ_API_KEY");
        if (env_key != nullptr) options.transport_api_key = env_key;
    }

    MfqHttpRuntimeTransportConfig transport_config;
    transport_config.host = options.transport_host;
    transport_config.port = options.transport_port;
    transport_config.model_name = options.runtime_model_name;
    transport_config.model_type = loaded.metadata.model_type;
    transport_config.api_key = options.transport_api_key;
    transport_config.max_context = loaded.metadata.max_context;
    transport_config.vocab_size = loaded.metadata.vocab_size;
    transport_config.model_capabilities = MfqModelCapabilities{
        loaded.metadata.architecture,
        loaded.metadata.capabilities.text,
        loaded.metadata.capabilities.image_input,
        loaded.metadata.capabilities.video_input,
        loaded.metadata.capabilities.audio_input,
        loaded.metadata.capabilities.audio_output,
        loaded.metadata.capabilities.full_duplex,
        loaded.metadata.capabilities.mtp,
        "model-graph+cuda-adapters",
    };
    const auto& runtime_assets = *loaded.metadata.source;
    if (runtime_assets.has_asset(mfq::cuda::kTokenizerGgufAsset)) {
        transport_config.tokenizer_gguf = read_runtime_asset(
            runtime_assets, mfq::cuda::kTokenizerGgufAsset);
    } else {
        transport_config.tokenizer_model = options.tokenizer_model;
    }
    const auto embedded_profile = runtime_assets.metadata().find(
        "runtime.sampling.v1");
    transport_config.runtime_profile = resolve_mfq_runtime_profile(
        options.model_path,
        loaded.metadata.architecture,
        transport_config.model_type,
        transport_config.model_name,
        embedded_profile == runtime_assets.metadata().end()
            ? std::string()
            : embedded_profile->second,
        read_runtime_asset_text(runtime_assets, mfq::kModelConfigAsset),
        options.runtime_sampling_profile);

    auto transport = options.stdio_mode
        ? make_mfq_stdio_transport(transport_config)
        : make_mfq_http_transport(transport_config);
    MfqRuntime runtime(
        std::move(loaded.inference), std::move(transport));
    const int status = runtime.run();
    if (g_moe_expert_cache) {
        print_moe_expert_cache_stats(std::cout);
    }
    return status;
}

int execute_runtime(RuntimeCommandOptions options) {
    return mfq::cuda::internal::with_command_errors([&]() -> int {
        if (options.stdio_mode) prepare_mfq_stdio_transport();
        mfq::cuda::internal::setup_cuda_load(options);
        if (!options.duplex_input_prefix.empty()) {
            return mfq::cuda::minicpmo45::run_duplex(options, options);
        }
        if (!options.input_prefix.empty()) {
            return mfq::cuda::minicpmo45::run_composite(options, options);
        }
        if (options.transport_mode) {
            return run_transport_runtime(options);
        }
        return mfq::cuda::internal::run_cuda_token_generation(
            options, options);
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
