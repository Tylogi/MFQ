#include "engine/cuda_engine.h"
#include "diagnostics/generation_result.h"
#include "cuda_execution.h"
#include "cuda_runtime_config.h"
#include "engine/text_session_cache.h"
#include "models/minicpmo45/causal_lm.h"
#include "models/qwen35/causal_lm.h"
#include "models/qwen35/linear_attention.h"
#include "models/session_state.h"
#include "mfq_paged_prefix_cache.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace mfq::cuda;
using namespace mfq::cuda::internal;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_linear_execution() {
    Nint8ZeroCpu nint_cpu;
    nint_cpu.out = 2;
    nint_cpu.ng = 1;
    nint_cpu.neuron_len = 32;
    nint_cpu.shape = {2, 32};
    nint_cpu.q.assign(64, 1);
    nint_cpu.scale_h.assign(2, 0x3c00); // FP16 1
    QuantLinear nint;
    nint.kind = QuantLinearKind::Nint;
    nint.nint = to_gpu_nint8_zero(nint_cpu);

    NvqCpu nvq_cpu;
    nvq_cpu.format = 3; // D4, four values per codebook entry
    nvq_cpu.sub_bits = 4;
    nvq_cpu.gs = nvq_cpu.neuron_len = 24;
    nvq_cpu.out = 2;
    nvq_cpu.ng = 1;
    nvq_cpu.nvec = 6;
    nvq_cpu.nsign = 3;
    nvq_cpu.shape = {2, 24};
    nvq_cpu.indices_packed.assign(12, 0);
    nvq_cpu.aux_packed.assign(6, 0);
    nvq_cpu.sub_scale_packed.assign(1, 0xff);
    nvq_cpu.neuron_scale_h.assign(2, 0x3c00);
    nvq_cpu.codebook.assign(256 * 4, 1);
    QuantLinear nvq;
    nvq.kind = QuantLinearKind::Nvq;
    nvq.nvq = to_gpu_nvq(nvq_cpu);

    CudaExecutionContext owner, other;
    for (const auto* linear : {&nint, &nvq}) {
        const auto input = (mfq_tensor_backend::arange(
            16 * linear->neuron_len(),
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32)) * 0.013)
            .to(mfq_tensor_backend::kFloat16).reshape({2, 8, linear->neuron_len()});
        for (auto mode : {KlMmqMode::Default, KlMmqMode::Fp16, KlMmqMode::Nint8One}) {
            owner.kl_mmq.mode = mode;
            other.kl_mmq.mode = mode == KlMmqMode::Nint8One
                ? KlMmqMode::Fp16 : KlMmqMode::Nint8One;
            auto verify = [&](auto run) {
                auto expected = run(input.reshape({16, linear->neuron_len()}))
                    .reshape({2, 8, 2});
                const auto calls = owner.kl_mmq.activation_quantize_calls;
                const auto dense_calls = owner.kl_mmq.dense_calls;
                const auto actual = run(input);
                check(actual.scalar_type() == expected.scalar_type() &&
                          actual.equal(expected),
                      "linear output lost its shape/dtype");
                check(owner.kl_mmq.activation_quantize_calls == calls +
                          (mode == KlMmqMode::Nint8One ? 1 : 0) &&
                      owner.kl_mmq.dense_calls == dense_calls +
                          (mode != KlMmqMode::Default ? 1 : 0) &&
                      other.kl_mmq.activation_quantize_calls == 0 &&
                      other.kl_mmq.dense_calls == 0,
                      "linear diagnostics escaped the explicit execution");
                check(run(input).equal(expected),
                      "linear execution changed without hidden state");
            };
            verify([&](auto x) { return linear->forward(owner, x); });
            verify([&](auto x) { return linear->forward_bf16_output(owner, x); });
            for (int gate_mode : {1, 2}) {
                verify([&](auto x) { return linear->forward_input_mul(owner, x, x, gate_mode); });
                if (linear->is_nint() && mode == KlMmqMode::Fp16) {
                    verify([&](auto x) {
                        return linear->forward_input_mul_f32_kld(owner, x, x, gate_mode);
                    });
                }
            }
        }
    }
}

static void check_snapshots(CudaExecutionContext& execution, bool hybrid) {
    Qwen35CausalLm model;
    model.execution = &execution;
    auto full = std::make_unique<FullBlock>();
    full->cache = KVCache(1, 1, 8, 4);
    model.blocks.push_back(std::move(full));
    if (hybrid) {
        auto linear = std::make_unique<qwen35::LinearAttentionBlock>();
        auto& config = linear->qwen_config;
        config.linear_conv_kernel_dim = 2;
        config.linear_num_key_heads = config.linear_num_value_heads = 1;
        config.linear_key_head_dim = config.linear_value_head_dim = 2;
        const auto options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
        linear->conv_state = mfq_tensor_backend::zeros({1, 1, 6}, options);
        linear->gdn_state = mfq_tensor_backend::zeros({1, 1, 2, 2}, options);
        model.blocks.push_back(std::move(linear));
    }
    model.cache_pos = 2;
    const auto text = model.capture_text_session_state({1, 2});
    check(text.kind() == (hybrid ? TextSessionStateKind::HybridAttention
                                : TextSessionStateKind::FullAttention),
          "wrong session codec");
    model.decode_position_delta = -1;
    auto image = model.capture_text_session_state({1, 2});
    image.input_key = "image";
    check(image.decode_position_delta == -1, "image position not captured");

    TextSessionCache memory({}, {});
    memory.store("text", text);
    memory.store("image", image);
    check(memory.restore_best(model, nullptr, "text", {1, 2, 3}, 2).tokens == 2,
          "text snapshot missed");
    check(model.decode_position_delta == 0, "image position leaked into text");
    check(memory.restore_best(model, nullptr, "image", {1, 2, 3}, 2, "image").tokens == 2,
          "image snapshot missed");
    check(model.decode_position_delta == -1, "image position not restored");
    if (hybrid) return;

    bool rejected = false;
    try {
        (void)encode_cuda_paged_session(image, 2, 0);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "paged codec silently discarded media positions");
    const auto decoded = decode_cuda_paged_session(
        encode_cuda_paged_session(text, 2, 0), text.tokens, 2);
    model.restore_text_session_state(decoded);
    check(model.decode_position_delta == 0, "paged text retained media position");

    const auto directory = std::filesystem::temp_directory_path() /
        ("mfq-runtime-state-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        mfq::cache::PagedPrefixCacheConfig config;
        config.cache_dir = directory;
        config.compatibility_key = "runtime-state-test";
        config.block_size_tokens = 2;
        auto paged = std::make_shared<mfq::cache::PagedPrefixCache>(config);
        TextSessionCache mixed({}, {}, paged);
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.fork_session("source", "fork") == 2,
              "fork omitted a cache backend");
        check(mixed.close_session("source") == 2,
              "close omitted a cache backend");
        check(mixed.restore_best(model, nullptr, "fork", {1, 2, 3}, 2, "image").tokens == 2,
              "forked memory snapshot missing");
        check(mixed.close_session("fork") == 2, "fork ownership missing");
        // Prefix lookup is cross-session: close both owners before checking.
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "closed memory snapshot is still reusable");
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.clear_live_sessions() == 2, "live clear omitted a backend");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "live clear retained memory snapshot");
        mixed.store("source", text);
        mixed.store("source", image);
        check(mixed.clear() >= 1, "clear released nothing");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2, "image").tokens == 0,
              "clear retained memory snapshot");
        check(mixed.restore_best(model, nullptr, "source", {1, 2, 3}, 2).tokens == 0,
              "clear retained paged snapshot");
    }
    std::filesystem::remove_all(directory);
}

// Optional real-model check: pass a Qwen3.5 model directory and tokenizer GGUF.
static void check_batching(const char* model_path, const char* tokenizer) {
    CudaEngineOptions options;
    options.model_path = model_path;
    options.tokenizer_model = tokenizer;
    options.context_size = 256;
    options.continuous_batching = 2;
    options.prefill_chunk_size = 8;
    auto engine = load_cuda_engine(options);
    using namespace mfq::engine;
    MfqSamplingParams sampling;
    sampling.max_tokens = 32; sampling.temperature = 0; sampling.top_k = 1; sampling.enable_mtp = false;
    const auto reference = mfq::cuda::diagnostics::check_engine_steps(engine, {101, 202, 303}, sampling);
    check(reference.size() == 32, "batched generation length");
    check(mfq::cuda::diagnostics::check_engine_steps(engine, {101, 202, 303}, sampling) == reference,
          "batched repeat changed output");
    EngineRequest cancelled; cancelled.id = "cancel";
    cancelled.token_ids = {101, 202, 303}; cancelled.input.sampling = sampling;
    check(engine.admit(cancelled) == Admission::accepted, "cancel admission");
    engine.cancel("cancel");
    auto result = engine.step({});
    check(result.events.size() == 1 && std::holds_alternative<Cancelled>(result.events[0].data),
          "cancel before prefill did not release");
    cancelled.token_ids.assign(32, 101);
    check(engine.admit(cancelled) == Admission::accepted, "prefill cancel admission");
    result = engine.step({"cancel"});
    check(result.events.size() == 1 && std::holds_alternative<PrefillProgress>(result.events[0].data),
          "prefill did not yield after one chunk");
    engine.cancel("cancel");
    result = engine.step({});
    check(result.events.size() == 1 && std::holds_alternative<Cancelled>(result.events[0].data),
          "cancel during prefill did not release");
    check(mfq::cuda::diagnostics::check_engine_steps(engine, {101, 202, 303}, sampling) == reference,
          "cancel changed subsequent output");

}

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    check_linear_execution();
    {
        CudaExecutionContext execution;
        check_snapshots(execution, false);
        check_snapshots(execution, true);
    }
    if (argc == 3) check_batching(argv[1], argv[2]);
    else check(argc == 1, "usage: mfq-cuda-runtime-state-test [model tokenizer]");
    std::cout << "CUDA runtime state checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
