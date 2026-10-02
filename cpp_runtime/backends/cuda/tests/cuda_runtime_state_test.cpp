#include "cuda_execution.h"
#include "cuda_runtime_config.h"
#include "cuda_sampling.h"
#include "diagnostics/generation_result.h"
#include "storage/text_session_cache.h"
#include "mfq_paged_prefix_cache.h"
#include "models/gemma4/causal_lm.h"
#include "models/minicpmo45/ops.h"
#include "models/minicpmo45/tts.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "storage/session_state.h"
#include "moe.h"
#include "storage/mfe_expert_store.h"
#include "storage/moe_expert_cache.h"
#include "storage/weight_loader.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace mfq::cuda;
using namespace mfq::cuda::internal;

static void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void check_sampling_storage() {
    using namespace mfq_tensor_backend;
    auto ids = tensor(std::vector<int64_t>{1, 2, 1}, TensorOptions().device(kCUDA).dtype(kInt64));
    auto storage = full({6}, 99, ids.options().dtype(kInt32));
    auto counts = SamplingOps::token_counts(ids, 6, storage);
    auto owned = SamplingOps::token_counts(ids, 6);
    check(counts.data_ptr() == storage.data_ptr(), "sampling replaced graph count storage");
    const auto expected = tensor(std::vector<int32_t>{0, 2, 1, 0, 0, 0});
    for (const auto& value : {counts, owned}) {
        auto host = value.cpu().contiguous();
        check(host.scalar_type() == kInt32 && host.nbytes() == expected.nbytes() &&
              std::memcmp(host.data_ptr(), expected.data_ptr(), host.nbytes()) == 0,
              "sampling counts were not reset from the prompt");
    }
    MfqSamplingParams params;
    params.temperature = 0.8; params.top_k = 4; params.top_p = 0.9;
    params.seed = 73; params.repetition_penalty = 1.2;
    Sampler first(params), second(params);
    check(first.ops().random_host().data_ptr() != second.ops().random_host().data_ptr(),
          "samplers share their host random buffer");
    auto logits = tensor(std::vector<float>{1, 2, 3, 4, 5, 6},
                         TensorOptions().device(kCUDA).dtype(kFloat32)).reshape({1, 6});
    for (int i = 0; i < 16; ++i)
        check(sample_logits(first, logits.clone(), counts).item<int64_t>() ==
              sample_logits(second, logits.clone(), owned).item<int64_t>(),
              "owned/reused sampling storage changed seeded output");
    (void)SamplingOps::token_counts(ids, 6, storage);
    check(owned.sum().item<int64_t>() == 3, "resetting one sampler changed another");
}

static void check_graph_warmup_state() {
    using namespace mfq_tensor_backend;
    Qwen35CausalLm model;
    auto block = std::make_unique<qwen35::LinearAttentionBlock>();
    auto* recurrent = block.get();
    const auto options = TensorOptions().device(kCUDA).dtype(kFloat32);
    recurrent->conv_state = ones({2}, options);
    recurrent->gdn_state = ones({3}, options);
    const auto* conv_address = recurrent->conv_state.data_ptr();
    const auto* gdn_address = recurrent->gdn_state.data_ptr();
    model.blocks.push_back(std::move(block));
    model.cache_pos = 5;
    for (bool fail : {false, true}) {
        MfqCudaGraph graph;
        int calls = 0;
        bool threw = false;
        try {
            prepare_decode_graph_memory(model, graph, [&] {
                ++calls;
                recurrent->conv_state.zero_();
                recurrent->gdn_state.zero_();
                if (fail) throw std::runtime_error("warmup failed");
            });
        } catch (const std::runtime_error&) { threw = true; }
        check(threw == fail && calls == (fail ? 1 : 2), "graph warmup execution changed");
        check(model.cache_pos == 5 && recurrent->conv_state.data_ptr() == conv_address &&
                  recurrent->gdn_state.data_ptr() == gdn_address,
              "graph warmup changed persistent state addresses");
        check(recurrent->conv_state.sum().item<float>() == 2 &&
                  recurrent->gdn_state.sum().item<float>() == 3,
              "graph warmup did not restore recurrent state");
    }
    recurrent->speculative_pending = true;
    bool rejected = false;
    try { (void)recurrent->graph_warmup_state(); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "graph warmup accepted unconfirmed recurrent state");
}

static void check_dense_loading() {
    using namespace mfq_tensor_backend;
    struct Source final : mfq::ModelSource {
        std::vector<uint8_t> bytes;
        std::vector<mfq::TensorMetadata> records{{"weight", "F32", "F32", std::nullopt, 0}};
        std::vector<std::filesystem::path> paths;
        std::unordered_map<std::string, std::string> meta;
        std::vector<std::string> asset_names;
        const std::vector<std::filesystem::path>& source_paths() const noexcept override { return paths; }
        std::string_view architecture() const noexcept override { return {}; }
        const std::unordered_map<std::string, std::string>& metadata() const noexcept override { return meta; }
        const std::vector<mfq::TensorMetadata>& tensors() const noexcept override { return records; }
        const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
            return name == "weight" ? &records[0] : nullptr;
        }
        void read_range_into(std::string_view, uint64_t offset, std::byte* output, size_t size) const override {
            check(offset <= bytes.size() && size <= bytes.size() - offset, "dense fixture range");
            if (size) std::memcpy(output, bytes.data() + offset, size);
        }
        const std::vector<std::string>& assets() const noexcept override { return asset_names; }
        bool has_asset(std::string_view) const noexcept override { return false; }
        std::vector<std::byte> read_asset(std::string_view) const override { throw std::runtime_error("no assets"); }
        std::optional<mfq::ModelGraph> model_graph() const override { return std::nullopt; }
    } source;
    CudaExecutionContext execution;
    const uint32_t rank = 2;
    const int64_t shape[] = {2, 2};
    const size_t header = sizeof(rank) + sizeof(shape);
    auto same = [](Tensor value, const Tensor &expected) {
        value = value.cpu().contiguous();
        return value.scalar_type() == expected.scalar_type() && value.sizes() == expected.sizes() &&
               std::memcmp(value.data_ptr(), expected.data_ptr(), expected.nbytes()) == 0;
    };
    for (const auto &[name, dtype] : std::vector<std::pair<std::string, ScalarType>>{
             {"BF16", kBFloat16}, {"F16", kFloat16}, {"F32", kFloat32}, {"I64", kInt64}, {"I32", kInt32}}) {
        const auto expected = tensor(std::vector<float>{-3.25, 0, 1.125, 7}).reshape({2, 2}).to(dtype);
        source.records[0].dtype = name;
        source.bytes.resize(header + expected.nbytes());
        std::memcpy(source.bytes.data(), &rank, sizeof(rank));
        std::memcpy(source.bytes.data() + sizeof(rank), shape, sizeof(shape));
        std::memcpy(source.bytes.data() + header, expected.data_ptr(), expected.nbytes());
        source.records[0].nbytes = source.bytes.size();
        auto native = load_dense_native_gpu(execution, source, "weight");
        check(same(native, expected), "dense native dtype/value changed");
        const auto promoted = name == "BF16" || name == "F16" ? kFloat32 : dtype;
        for (bool cpu_layer : {false, true}) {
            execution.loading_cpu_layer = cpu_layer;
            auto value = load_dense_gpu(execution, source, "weight");
            check(value.is_cuda() != cpu_layer && value.scalar_type() == promoted &&
                      same(value, expected.to(promoted)), "dense load placement/promotion changed");
        }
    }
    const auto valid = source.bytes;
    auto rejects = [&] {
        source.records[0].nbytes = source.bytes.size();
        bool rejected = false;
        try { (void)load_dense_cpu(execution, source, "weight"); }
        catch (const std::exception&) { rejected = true; }
        check(rejected, "malformed dense tensor accepted");
    };
    for (size_t size : {size_t(0), size_t(3), header - 1, valid.size() - 1}) {
        source.bytes = valid;
        source.bytes.resize(size);
        rejects();
    }
    for (int64_t extent : {int64_t(-1), std::numeric_limits<int64_t>::max()}) {
        source.bytes = valid;
        std::memcpy(source.bytes.data() + sizeof(rank), &extent, sizeof(extent));
        rejects();
    }
}

static void check_cached_moe_binding() {
    using namespace mfq_tensor_backend;
    MfeCpu cpu;
    cpu.n_experts = 2;
    cpu.out_per_expert = 4;
    cpu.neuron_len = 32;
    MfeCpuPool pool;
    pool.dtype = "MXFP4";
    pool.expert_ids = {0, 1};
    pool.mxfp4 = {8, 32, std::vector<uint8_t>(128), std::vector<uint8_t>(8, 127)};
    for (size_t i = 0; i < pool.mxfp4.values.size(); ++i)
        pool.mxfp4.values[i] = static_cast<uint8_t>(i * 17);
    cpu.pools.push_back(pool);

    // The same weights exercise host-backed and exact-range cache registration.
    auto append = [](auto& bytes, uint64_t value, int width) {
        for (int i = 0; i < width; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    };
    std::vector<uint8_t> payload{'M', 'X', 'T', '1', 1, 4, 0, 0};
    for (uint64_t value : {8, 32, 8, 16, 8, 1}) append(payload, value, 8);
    payload.insert(payload.end(), pool.mxfp4.values.begin(), pool.mxfp4.values.end());
    payload.insert(payload.end(), pool.mxfp4.scales.begin(), pool.mxfp4.scales.end());
    std::vector<uint8_t> blob{'N', 'I', 'M', '2'};
    for (uint64_t value : {2, 4, 32, 1, 2, 5}) append(blob, value, 4);
    append(blob, payload.size(), 8);
    append(blob, 0, 8);
    append(blob, 0, 4);
    append(blob, 1, 4);
    blob.insert(blob.end(), pool.dtype.begin(), pool.dtype.end());
    blob.insert(blob.end(), payload.begin(), payload.end());
    auto store = std::make_shared<MfeMxfp4ExpertStore>(MfqRecordRange{
        "experts", "MFE", {}, 0, blob.size(),
        [&blob](uint64_t offset, std::span<uint8_t> output) {
            check(offset <= blob.size() && output.size() <= blob.size() - offset,
                  "expert range is out of bounds");
            std::copy_n(blob.data() + offset, output.size(), output.data());
        }});
    CudaExecutionContext execution;
    const auto resident = to_gpu_mixed_moe(cpu, execution.config);
    auto options = TensorOptions().device(kCUDA);
    auto input = (arange(32, options.dtype(kFloat32)) * 0.01 - 0.1)
                     .to(kFloat16).reshape({1, 32});
    for (bool ranges : {false, true}) {
        auto cache = make_moe_expert_cache(1 << 20, execution.config);
        auto runtime = ranges ? make_mxfp4_range_runtime(*store)
                              : make_mixed_moe_runtime(cpu, false);
        auto weight = cache_moe_weight(cache, "experts", runtime, 1, 0, "gate",
                                       ranges ? store : nullptr);
        check(moe_expert_cache_has_sources(cache), "MoE source was not registered");
        finalize_moe_expert_cache(cache);
        for (int32_t expert : {0, 1, 0}) {
            auto ids = tensor(std::vector<int32_t>{expert}, options.dtype(kInt32)).reshape({1, 1});
            auto route = build_moe_route_plan(ids, 2);
            auto expected = resident.forward(execution, input, route).to(kFloat32);
            auto actual = weight.forward(execution, input, route).to(kFloat32);
            check((actual - expected).abs().max().item<float>() == 0,
                  "cache registration changed expert output");
        }
    }
}

static void check_gemma_composition() {
    using namespace mfq_tensor_backend;
    using mfq::models::gemma4::FfnNorm;
    auto fp32 = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto values = arange(16, fp32).reshape({2, 8});
    auto dense = ((values - 8) * 0.09).to(kFloat16);
    auto routed = ((values + 2) * 0.13).to(kFloat16);
    auto residual = ((values - 3) * 0.05).to(kFloat16);
    auto weight = arange(8, fp32) * 0.02 + 1;
    auto dense_norm = weight, expert_norm = weight + 0.1, output_norm = weight + 0.2;
    auto scale = full({1}, 0.8, fp32).to(kFloat16);
    auto run = [&](bool fused) {
        return mfq::models::gemma4::feed_forward(
            true, true, fused, residual, [&] { return dense; }, [&] { return routed; },
            [&](Tensor x, FfnNorm role) {
                auto norm = role == FfnNorm::dense     ? dense_norm
                            : role == FfnNorm::experts ? expert_norm
                                                       : output_norm;
                return rms_norm_f16_cuda(x.contiguous(), norm, 1e-6, 0.0);
            },
            [](Tensor x, Tensor y) { return x + y; },
            [](Tensor x, Tensor y) { return acc_cuda(x.contiguous(), y.contiguous()); },
            [&](Tensor x) { return x * scale; },
            [&](Tensor d, Tensor e, Tensor r) {
                return gemma4_ffn_merge_f16_cuda(d, e, r, dense_norm, expert_norm, output_norm,
                                                 scale, 1e-6);
            });
    };
    auto expected = run(false).to(kFloat32);
    auto actual = run(true).to(kFloat32);
    check((actual - expected).abs().max().item<float>() < 3e-3F,
          "shared Gemma FFN composition differs from the fused kernel");
}

static void check_tts_sampling() {
    using namespace mfq_tensor_backend;
    MiniCPMO45TtsSamplingOps ops;
    auto float_options = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto scores = tensor(std::vector<float>{-2, 4, 3, 1, 0, 20}, float_options).reshape({1, 6});
    auto token = tensor(std::vector<int64_t>{1}, TensorOptions().dtype(kInt64).device(kCUDA));
    std::vector<Tensor> history(16, token);
    auto penalized = ops.penalties(scores.clone(), history, 1.1).cpu();
    check(std::abs(penalized.data_ptr<float>()[1] - 4.0F / std::pow(1.1F, 16)) < 2e-6F,
          "TTS repeated-token penalty changed");
    auto top = ops.top_k(scores.clone(), 3).cpu();
    check(std::isinf(top.data_ptr<float>()[0]) && top.data_ptr<float>()[2] == 3,
          "TTS top-k filtering changed");
    mfq::models::minicpmo45::TtsSampling options{5, 1, 3, 0, 0.8, 0.0001, 1.1, 0.0};
    std::mt19937 rng(42);
    for (bool reference : {false, true}) {
        ops.evaluator_rng = reference ? &rng : nullptr;
        auto sample = [&](int64_t step) {
            return mfq::models::minicpmo45::sample_tts(ops, scores.clone(), options,
                                                       std::span<const Tensor>(history), step,
                                                       reference)
                .item<int64_t>();
        };
        check(sample(0) == 2, "TTS did not mask EOS or penalize history before selection");
        check(sample(1) == 5, "TTS did not release EOS after minimum length");
    }
}

static void check_ffn_branches() {
    using namespace mfq_tensor_backend;
    const auto options = TensorOptions().dtype(kFloat32).device(kCUDA);
    auto linear = [&](int64_t out, int64_t in, float scale) {
        QuantLinear result;
        result.kind = QuantLinearKind::Dense;
        result.dense =
            ((arange(out * in, options).reshape({out, in}) - out * in / 2) * scale).to(kFloat16);
        return result;
    };
    auto dense_ffn = [&](float scale) {
        FFN result;
        result.gate_up.layers = {linear(16, 8, scale), linear(16, 8, scale / 2)};
        result.gate_up.outs = {16, 16};
        result.down = linear(8, 16, scale / 3);
        return result;
    };
    auto input = ((arange(24, options).reshape({3, 8}) - 12) * 0.05).to(kFloat16);
    CudaExecutionContext execution;
    auto low = dense_ffn(0.01F);
    auto high = std::make_unique<FFN>(dense_ffn(0.015F));
    execution.config.decode_branch_parallel = true;
    auto decode = input.narrow(0, 0, 1);
    auto expected = acc_cuda(low.forward(execution, decode), high->forward(execution, decode));
    low.important_neurons = std::move(high);
    for (bool parallel : {false, true}) {
        execution.config.important_neuron_branch_parallel = parallel;
        auto actual = low.forward(execution, decode);
        if (parallel)
            check(low.important_neuron_executor->streams.size() == 2,
                  "IN parallel fixture did not use two streams");
        check((actual.to(kFloat32) - expected.to(kFloat32)).abs().max().item<float>() < 1e-3,
              "important-neuron branch composition changed");
    }
    // One physical shard exercises the TP operation binding on single-GPU hosts.
    auto shard = [](QuantLinear &weight, TensorParallelAxis axis) {
        QuantLinearShard piece;
        piece.kind = QuantLinearKind::Dense;
        piece.dense = weight.dense;
        piece.input_end = weight.dense.size(1);
        piece.output_end = weight.dense.size(0);
        weight.logical_out = piece.output_end;
        weight.logical_neuron_len = piece.input_end;
        weight.tensor_parallel_axis = axis;
        weight.tensor_parallel_shards = {std::move(piece)};
    };
    for (int activation = 0; activation < 3; ++activation) {
        auto local = dense_ffn(0.01F);
        auto parallel = dense_ffn(0.01F);
        local.geglu = parallel.geglu = activation == 1;
        local.swiglu_limit = parallel.swiglu_limit = activation == 2 ? 0.1 : 0.0;
        auto expected = local.forward(execution, input);
        for (auto &projection : parallel.gate_up.layers)
            shard(projection, TensorParallelAxis::Output);
        shard(parallel.down, TensorParallelAxis::Input);
        check(parallel.tensor_parallel_dense_compatible(), "TP fixture is not eligible");
        auto actual = parallel.forward(execution, input);
        check((actual.to(kFloat32) - expected.to(kFloat32)).abs().max().item<float>() < 1e-3,
              "tensor-parallel gated MLP differs from local composition");
    }
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
    const auto reference = mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling);
    check(reference.size() == 32, "batched generation length");
    check(mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling) == reference,
          "batched repeat changed output");
    EngineRequest cancelled; cancelled.id = "cancel";
    cancelled.token_ids = {101, 202, 303}; cancelled.input.sampling = sampling;
    check(engine->admit(cancelled) == Admission::accepted, "cancel admission");
    engine->cancel("cancel");
    auto result = engine->step({});
    check(result.events.size() == 1 && std::holds_alternative<Cancelled>(result.events[0].data),
          "cancel before prefill did not release");
    cancelled.token_ids.assign(32, 101);
    check(engine->admit(cancelled) == Admission::accepted, "prefill cancel admission");
    result = engine->step({"cancel"});
    check(result.events.size() == 1 && std::holds_alternative<PrefillProgress>(result.events[0].data),
          "prefill did not yield after one chunk");
    engine->cancel("cancel");
    result = engine->step({});
    check(result.events.size() == 1 && std::holds_alternative<Cancelled>(result.events[0].data),
          "cancel during prefill did not release");
    check(mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling) == reference,
          "cancel changed subsequent output");

}

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    check_sampling_storage();
    check_linear_execution();
    check_graph_warmup_state();
    check_dense_loading();
    check_cached_moe_binding();
    check_gemma_composition();
    check_tts_sampling();
    check_ffn_branches();
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
