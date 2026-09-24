#include "cli.h"
#include "runtime/runner.h"
#include "eval/kl.h"
#include "cuda_execution.h"
#include "moe_expert_cache.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {

struct EvalCommandOptions : CudaLoadOptions {
    std::string kl_base, kl_save_logits_f16;
    std::string kl_chunks_sequence_arg, kl_mmq_sequence_arg;
    std::string kl_evaluator_arg = "optimized";
    std::string kl_mmq_arg = "default";
    int kl_chunks = -1;
    int kl_score_count = -1;
    int64_t kl_n_batch = 0;
    KlReferenceContract kl_reference_contract;
    int kl_stream_layers = 0;
    int kl_stream_batch = 1;
};

} // namespace mfq::cuda

using namespace mfq::cuda::internal;

namespace {
KlEvaluator parse_kl_evaluator(const std::string& value) {
    if (value == "legacy") return KlEvaluator::Legacy;
    if (value == "optimized") return KlEvaluator::Optimized;
    throw std::runtime_error(
        "--kl-evaluator must be legacy or optimized");
}

struct EvalCommand : mfq::cuda::EvalCommandOptions {
    explicit EvalCommand(mfq::cuda::EvalCommandOptions options)
        : mfq::cuda::EvalCommandOptions(std::move(options)) {}
    int run() { return with_command_errors([&]() -> int {
        setup_cuda_load(*this);
        if (model_path.empty()) {
            std::cerr << "missing model or execution mode (see --help)\n";
            return 2;
        }
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (!cpu_offload_layers_arg.empty()) {
            g_dsv4_cpu_offload_layers =
                parse_layer_ranges(cpu_offload_layers_arg);
            std::vector<int> ordered(
                g_dsv4_cpu_offload_layers.begin(),
                g_dsv4_cpu_offload_layers.end());
            std::sort(ordered.begin(), ordered.end());
            std::cerr << "cpu_offload_layers=";
            for (size_t index = 0; index < ordered.size(); ++index) {
                if (index) std::cerr << ',';
                std::cerr << ordered[index];
            }
            std::cerr << std::endl;
        }
        std::vector<int64_t> kl_chunks_sequence;
        const KlEvaluator kl_evaluator =
            parse_kl_evaluator(kl_evaluator_arg);
        const KlMmqMode kl_mmq_mode =
            parse_kl_mmq_mode(kl_mmq_arg);
        if (kl_chunks == 0 || kl_chunks < -1) {
            throw std::runtime_error(
                "--kl-chunks must be positive or -1 for all chunks");
        }
        if (kl_score_count == 0 || kl_score_count < -1) {
            throw std::runtime_error(
                "--kl-score-count must be positive or -1 for the stored count");
        }
        if (kl_n_batch < 0) {
            throw std::runtime_error(
                "--kl-n-batch must be non-negative; 0 uses n_ctx");
        }
        if (kl_reference_contract.n_batch < 0 ||
                kl_reference_contract.n_ubatch < 0) {
            throw std::runtime_error(
                "KL reference n_batch and n_ubatch must be non-negative");
        }
        if ((kl_reference_contract.n_batch == 0) !=
                (kl_reference_contract.n_ubatch == 0)) {
            throw std::runtime_error(
                "--kl-reference-n-batch and --kl-reference-n-ubatch must "
                "be supplied together");
        }
        if (kl_reference_contract.n_ubatch >
                kl_reference_contract.n_batch) {
            throw std::runtime_error(
                "--kl-reference-n-ubatch must not exceed "
                "--kl-reference-n-batch");
        }
        const auto active_env = [](const char * name) {
            const char * value = std::getenv(name);
            return value != nullptr && value[0] != '\0';
        };
        if (!kl_base.empty() && active_env("MFQ_KL_WINDOW_M")) {
            throw std::runtime_error(
                "MFQ_KL_WINDOW_M is disabled because it silently changes "
                "the metric; use --kl-score-count");
        }
        std::vector<KlMmqMode> kl_mmq_sequence;
        if (!kl_mmq_sequence_arg.empty()) {
            if (kl_mmq_arg != "default") {
                throw std::runtime_error(
                    "--kl-mmq and --kl-mmq-sequence are mutually exclusive");
            }
            kl_mmq_sequence =
                parse_kl_mmq_sequence(kl_mmq_sequence_arg);
        }
        if (kl_mmq_mode != KlMmqMode::Default &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq nint8_1/fp16 requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (!kl_mmq_sequence.empty() &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized)) {
            throw std::runtime_error(
                "--kl-mmq-sequence requires "
                "--kl-base and --kl-evaluator optimized");
        }
        if (kl_n_batch != 0 &&
                (kl_base.empty() ||
                 kl_evaluator != KlEvaluator::Optimized ||
                 kl_stream_layers > 0)) {
            throw std::runtime_error(
                "--kl-n-batch requires non-streamed --kl-base with "
                "--kl-evaluator optimized");
        }
        std::unique_ptr<KlMmqScope> kl_mmq_scope;
        const KlMmqMode load_mmq_mode =
            kl_mmq_sequence.empty()
            ? kl_mmq_mode : kl_mmq_sequence.front();
        if (load_mmq_mode != KlMmqMode::Default) {
            kl_mmq_scope =
                std::make_unique<KlMmqScope>(load_mmq_mode);
        }
        if (!kl_chunks_sequence_arg.empty()) {
            kl_chunks_sequence = parse_ids(kl_chunks_sequence_arg);
            if (kl_base.empty() ||
                std::any_of(
                    kl_chunks_sequence.begin(), kl_chunks_sequence.end(),
                    [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error(
                    "--kl-chunks-sequence requires --kl-base and "
                    "positive comma-separated chunk counts");
            }
        }
        g_profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        if (!kl_base.empty()) std::cout << std::unitbuf;
        if (!kl_base.empty()) {
            std::cout << "cpp_kl_contract"
                      << " evaluator=" << kl_evaluator_name(kl_evaluator)
                      << " mmq=" << kl_mmq_mode_name(kl_mmq_mode)
                      << " chunk_limit=" << kl_chunks
                      << " score_count_override=" << kl_score_count
                      << " requested_n_batch=" << kl_n_batch
                      << " reference_n_batch="
                      << kl_reference_contract.n_batch
                      << " reference_n_ubatch="
                      << kl_reference_contract.n_ubatch
                      << "\n";
        }
        if (!kl_base.empty() && kl_stream_layers > 0) {
            if (kl_evaluator != KlEvaluator::Legacy) {
                throw std::runtime_error(
                    "--kl-evaluator optimized is unavailable for streamed KL");
            }
            if (!kl_chunks_sequence.empty()) {
                throw std::runtime_error(
                    "--kl-chunks-sequence is unavailable for streamed KL");
            }
            if (g_moe_expert_cache) {
                throw std::runtime_error(
                    "--moe-gpu-cache-gb is unavailable for streamed KL");
            }
            return run_kl_eval_streamed(
                model_path, config_path, kl_base,
                kl_save_logits_f16, kl_chunks,
                kl_stream_layers, kl_stream_batch,
                kl_score_count, kl_reference_contract);
        }
        return with_loaded_cuda_model(*this, false,
            [&]<mfq::cuda::CudaBackbone Backbone>(auto& model,
                    auto& runtime_components, auto t0, auto t1) -> int {
        if (!kl_base.empty()) {
            if (!kl_mmq_sequence.empty()) {
                if (!kl_chunks_sequence.empty()) {
                    throw std::runtime_error(
                        "--kl-mmq-sequence cannot be combined with "
                        "--kl-chunks-sequence");
                }
                for (size_t index = 0;
                     index < kl_mmq_sequence.size(); ++index) {
                    std::cout << "cpp_kl_mmq_sequence_begin index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                    KlMmqScope run_scope(kl_mmq_sequence[index]);
                    const int status = run_kl_eval_batched(
                        model, kl_base, kl_chunks,
                        kl_n_batch, kl_score_count,
                        kl_reference_contract);
                    if (status != 0) return status;
                    std::cout << "cpp_kl_mmq_sequence_end index=" << index
                              << " mmq="
                              << kl_mmq_mode_name(kl_mmq_sequence[index])
                              << " chunks=" << kl_chunks << "\n";
                }
                return 0;
            }
            if (kl_chunks_sequence.empty()) {
                const int status = run_selected_kl_eval(
                    model, kl_base, kl_chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (g_moe_expert_cache) {
                    print_moe_expert_cache_stats(std::cout);
                }
                return status;
            }
            for (size_t index = 0; index < kl_chunks_sequence.size(); ++index) {
                const int chunks =
                    static_cast<int>(kl_chunks_sequence[index]);
                std::cout << "cpp_kl_sequence_begin index=" << index
                          << " chunks=" << chunks << "\n";
                const int status = run_selected_kl_eval(
                    model, kl_base, chunks,
                    kl_evaluator,
                    kl_n_batch, kl_score_count,
                    kl_reference_contract);
                if (status != 0) return status;
                std::cout << "cpp_kl_sequence_end index=" << index
                          << " chunks=" << chunks << "\n";
            }
            if (g_moe_expert_cache) {
                print_moe_expert_cache_stats(std::cout);
            }
            return 0;
        }
        throw std::runtime_error("KL evaluation requires --kl-base");
            });
    }); }
};

} // namespace
namespace mfq::cuda {
int execute_eval_command(EvalCommandOptions options) {
    return EvalCommand(std::move(options)).run();
}
} // namespace mfq::cuda

namespace mfq::cuda::commands {
namespace {

void print_eval_help() {
    std::cout
        << "MFQ CUDA evaluator\n\n"
        << "Usage:\n"
        << "  mfq-eval kl --model MODEL --kl-base REFERENCE [OPTIONS]\n\n"
        << "Evaluation options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
        << "  --kl-base PATH                  reference logits\n"
        << "  --kl-save-logits-f16 PATH       save F16 logits\n"
        << "  --kl-chunks N                   chunk limit; -1 means all\n"
        << "  --kl-score-count N              stored-row score limit\n"
        << "  --kl-n-batch N                  optimized evaluator batch size\n"
        << "  --kl-reference-n-batch N        reference logical batch size\n"
        << "  --kl-reference-n-ubatch N       reference physical batch size\n"
        << "  --kl-chunks-sequence LIST       run multiple chunk limits\n"
        << "  --kl-evaluator MODE             optimized or legacy\n"
        << "  --kl-mmq MODE                   default, nint8_1, or fp16\n"
        << "  --kl-mmq-sequence LIST          run multiple MMQ modes\n"
        << "  --kl-stream-layers N            streamed evaluator layer group\n"
        << "  --kl-stream-batch N             streamed evaluator chunk batch\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
        << "  --tensor-parallel DEVICES       tensor-parallel devices\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  -h, --help                      show this help\n";
}

EvalCommandOptions parse_eval(ArgCursor& args) {
    EvalCommandOptions result;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_eval_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result)) continue;
        if (option == "--kl-base") result.kl_base = args.value(option);
        else if (option == "--kl-save-logits-f16") {
            result.kl_save_logits_f16 = args.value(option);
        }
        else if (option == "--kl-chunks") {
            result.kl_chunks = integer<int>(args.value(option), option);
            if (result.kl_chunks == 0 || result.kl_chunks < -1) {
                usage_error("--kl-chunks must be positive or -1");
            }
        }
        else if (option == "--kl-score-count") {
            result.kl_score_count = integer<int>(args.value(option), option);
            if (result.kl_score_count == 0 || result.kl_score_count < -1) {
                usage_error("--kl-score-count must be positive or -1");
            }
        }
        else if (option == "--kl-n-batch") {
            result.kl_n_batch = integer<int64_t>(args.value(option), option);
            if (result.kl_n_batch < 0) {
                usage_error("--kl-n-batch must be non-negative");
            }
        }
        else if (option == "--kl-reference-n-batch") {
            result.kl_reference_contract.n_batch =
                integer<int64_t>(args.value(option), option);
            if (result.kl_reference_contract.n_batch < 0) {
                usage_error("--kl-reference-n-batch must be non-negative");
            }
        }
        else if (option == "--kl-reference-n-ubatch") {
            result.kl_reference_contract.n_ubatch =
                integer<int64_t>(args.value(option), option);
            if (result.kl_reference_contract.n_ubatch < 0) {
                usage_error("--kl-reference-n-ubatch must be non-negative");
            }
        }
        else if (option == "--kl-chunks-sequence") {
            result.kl_chunks_sequence_arg = args.value(option);
            validate_integer_list(result.kl_chunks_sequence_arg, option);
        }
        else if (option == "--kl-evaluator") {
            result.kl_evaluator_arg = args.value(option);
            try { (void)parse_kl_evaluator(result.kl_evaluator_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--kl-mmq") {
            result.kl_mmq_arg = args.value(option);
            try { (void)parse_kl_mmq_mode(result.kl_mmq_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--kl-mmq-sequence") {
            result.kl_mmq_sequence_arg = args.value(option);
            try { (void)parse_kl_mmq_sequence(result.kl_mmq_sequence_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--kl-stream-layers") {
            result.kl_stream_layers = integer<int>(args.value(option), option);
            if (result.kl_stream_layers < 0) {
                usage_error("--kl-stream-layers must be non-negative");
            }
        }
        else if (option == "--kl-stream-batch") {
            result.kl_stream_batch = integer<int>(args.value(option), option);
            if (result.kl_stream_batch <= 0) {
                usage_error("--kl-stream-batch must be positive");
            }
        }
        else usage_error("unknown option: " + std::string(option));
    }
    if (result.model_path.empty()) usage_error("--model is required");
    if (result.kl_base.empty()) usage_error("--kl-base is required");
    return result;
}

} // namespace

int run_eval(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help" ||
                std::string_view(argv[1]) == "-h")) {
            print_eval_help();
            return 0;
        }
        if (argc < 2 || std::string_view(argv[1]) != "kl") {
            usage_error("expected subcommand 'kl'");
        }
        ArgCursor args(argc, argv, 2);
        auto options = parse_eval(args);
        return mfq::cuda::execute_eval_command(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    }
}

} // namespace mfq::cuda::commands
