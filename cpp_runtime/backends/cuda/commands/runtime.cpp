#include "cli.h"
#include "engine/cuda_engine.h"
#include "minicpmo45.h"
#include "models/registry.h"
#include "mfq/model_source.h"
#include "transport.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
        << "  mfq-runtime --model MODEL --minicpmo-input-prefix PREFIX "
           "--minicpmo-output-prefix PREFIX [OPTIONS]\n"
        << "  mfq-runtime --model MODEL --minicpmo-duplex-input-prefix PREFIX "
           "--minicpmo-duplex-output-prefix PREFIX [OPTIONS]\n\n"
        << "Runtime options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
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

struct RuntimeOptions : CudaEngineOptions {
    std::string transport_host = "127.0.0.1";
    std::string runtime_model_name = "mfq-model", transport_api_key;
    std::string runtime_sampling_profile;
    int transport_port = 8080;
    bool transport_mode = false;
    bool stdio_mode = false;
};

struct RuntimeCommandOptions
    : RuntimeOptions,
      mfq::cuda::minicpmo45::CommandOptions {};

RuntimeCommandOptions parse_runtime(ArgCursor& args) {
    RuntimeCommandOptions result;
    bool transport_option = false;
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
        if (option == "--transport") {
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
    mfq::cuda::minicpmo45::validate_command_options(result);
    const int modes = static_cast<int>(result.transport_mode) +
        static_cast<int>(!result.input_prefix.empty()) +
        static_cast<int>(!result.duplex_input_prefix.empty());
    if (modes != 1) {
        usage_error("select exactly one execution mode: --transport, "
                    "MiniCPM-o composite, or MiniCPM-o duplex");
    }
    if (transport_option && !result.transport_mode) {
        usage_error("transport options require --transport");
    }
    if (result.transport_mode && !result.config_path.empty()) {
        usage_error("--config cannot be used with --transport");
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
    auto engine = load_cuda_engine(std::move(engine_options));
    if (options.transport_api_key.empty()) {
        const char* env_key = std::getenv("MFQ_API_KEY");
        if (env_key != nullptr) options.transport_api_key = env_key;
    }

    MfqHttpRuntimeTransportConfig transport_config;
    transport_config.host = options.transport_host;
    transport_config.port = options.transport_port;
    transport_config.model_name = options.runtime_model_name;
    transport_config.model_type = engine.metadata.model_type;
    transport_config.api_key = options.transport_api_key;
    transport_config.max_context = engine.metadata.max_context;
    transport_config.vocab_size = engine.metadata.vocab_size;
    transport_config.model_capabilities = MfqModelCapabilities{
        engine.metadata.architecture,
        engine.metadata.capabilities.text,
        engine.metadata.capabilities.image_input,
        engine.metadata.capabilities.video_input,
        engine.metadata.capabilities.audio_input,
        engine.metadata.capabilities.audio_output,
        engine.metadata.capabilities.full_duplex,
        engine.metadata.capabilities.mtp,
        "model-graph+cuda-adapters",
    };
    const auto& runtime_assets = *engine.metadata.source;
    const auto embedded_profile = runtime_assets.metadata().find(
        "runtime.sampling.v1");
    transport_config.runtime_profile = resolve_mfq_runtime_profile(
        options.model_path,
        engine.metadata.architecture,
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
    MfqRuntime runtime(std::move(engine), std::move(transport));
    return runtime.run();
}

int execute_runtime(RuntimeCommandOptions options) {
    return mfq::cuda::internal::with_command_errors([&]() -> int {
        if (options.stdio_mode) prepare_mfq_stdio_transport();
        if (options.duplex_input_prefix.empty() &&
                options.input_prefix.empty()) {
            return run_transport_runtime(options);
        }
        return mfq::cuda::internal::with_cuda_load(
            options, [&](CudaExecutionContext& execution) {
                return !options.duplex_input_prefix.empty()
                    ? mfq::cuda::minicpmo45::run_duplex(
                        execution, options, options)
                    : mfq::cuda::minicpmo45::run_composite(
                        execution, options, options);
        });
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
