#include "../kernels/mfq_cuda_norm_ops.h"
#include "../ops/cuda_execution.h"
#include "../engine/cuda_runtime_config.h"
#include "../ops/cuda_sampling.h"
#include "diagnostics/generation_result.h"
#include "storage/text_session_cache.h"
#include "mfq_paged_prefix_cache.h"
#include "models/gemma4/causal_lm.h"
#include "models/minicpmo45/ops.h"
#include "models/minicpmo45/tts.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "models/qwen35/batch_state.h"
#include "storage/session_state.h"
#include "../ops/moe.h"
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
    recurrent->state->conv_state = ones({2}, options);
    recurrent->state->gdn_state = ones({3}, options);
    const auto* conv_address = recurrent->state->conv_state.data_ptr();
    const auto* gdn_address = recurrent->state->gdn_state.data_ptr();
    model.blocks.push_back(std::move(block));
    model.cache_pos = 5;
    for (bool fail : {false, true}) {
        MfqCudaGraph graph;
        int calls = 0;
        bool threw = false;
        try {
            prepare_decode_graph_memory(model, graph, [&] {
                ++calls;
                recurrent->state->conv_state.zero_();
                recurrent->state->gdn_state.zero_();
                if (fail) throw std::runtime_error("warmup failed");
            });
        } catch (const std::runtime_error&) { threw = true; }
        check(threw == fail && calls == (fail ? 1 : 2), "graph warmup execution changed");
        check(model.cache_pos == 5 && recurrent->state->conv_state.data_ptr() == conv_address &&
                  recurrent->state->gdn_state.data_ptr() == gdn_address,
              "graph warmup changed persistent state addresses");
        check(recurrent->state->conv_state.sum().item<float>() == 2 &&
                  recurrent->state->gdn_state.sum().item<float>() == 3,
              "graph warmup did not restore recurrent state");
    }
    recurrent->state->speculative_pending = true;
    bool rejected = false;
    try { (void)recurrent->graph_warmup_state(); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "graph warmup accepted unconfirmed recurrent state");
}

struct WeightSource final : mfq::ModelSource {
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
        return name == records[0].name ? &records[0] : nullptr;
    }
    void read_range_into(std::string_view, uint64_t offset, std::byte* output, size_t size) const override {
        check(offset <= bytes.size() && size <= bytes.size() - offset, "dense fixture range");
        if (size) std::memcpy(output, bytes.data() + offset, size);
    }
    const std::vector<std::string>& assets() const noexcept override { return asset_names; }
    bool has_asset(std::string_view) const noexcept override { return false; }
    std::vector<std::byte> read_asset(std::string_view) const override { throw std::runtime_error("no assets"); }
    std::optional<mfq::ModelGraph> model_graph() const override { return std::nullopt; }
};

static void check_dense_loading() {
    using namespace mfq_tensor_backend;
    WeightSource source;
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
            if (name == "BF16" || name == "F16" || name == "F32") {
                auto linear = load_quant_linear(execution, source, "weight");
                check(linear.is_dense() && linear.dense.is_cuda() != cpu_layer &&
                          same(linear.dense, expected), "linear load changed stored weights");
                auto group = load_paired_gate_up(execution, source, {"weight", "weight"}, linear);
                check(group.layers.size() == 2 && same(group.layers[0].dense, expected) &&
                          same(group.layers[1].dense, expected), "loaded group weights changed");
                if (!cpu_layer) {
                    auto input = tensor(std::vector<float>{1, 2}).reshape({1, 2}).to(kCUDA, dtype);
                    auto output = tensor(std::vector<float>{-3.25, 15.125}).reshape({1, 2}).to(dtype);
                    check(same(linear.forward(execution, input), output), "loaded linear output changed");
                    auto outputs = group.forward(execution, input);
                    check(outputs.size() == 2 && same(outputs[0], output) && same(outputs[1], output),
                          "loaded projection group output changed");
                }
            }
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
    WeightSource source;
    source.bytes = blob;
    source.records = {{"experts", "MFE", "MFE", std::nullopt, blob.size()}};
    auto loaded_cpu = load_mfe_cpu(source, "experts");
    auto loaded = load_mfe_gpu(execution, source, "experts");
    auto dense = materialize_mfe_dense(loaded_cpu);
    auto options = TensorOptions().device(kCUDA);
    auto input = (arange(32, options.dtype(kFloat32)) * 0.01 - 0.1)
                     .to(kFloat16).reshape({1, 32});
    for (bool ranges : {false, true}) {
        execution.moe_expert_cache = make_moe_expert_cache(1 << 20, execution.config);
        auto& cache = execution.moe_expert_cache;
        std::weak_ptr<MoeExpertCache> lifetime = cache;
        auto runtime = ranges ? make_mxfp4_range_runtime(*store)
                              : make_mixed_moe_runtime(cpu, false);
        auto weight = cache_moe_weight(cache, "experts", runtime, 1, 0, "gate",
                                       ranges ? store : nullptr);
        check(moe_expert_cache_has_sources(cache), "MoE source was not registered");
        finalize_moe_expert_cache(cache);
        CudaExecutionContext other;
        other.moe_expert_cache = make_moe_expert_cache(1 << 20, other.config);
        execution.reset();
        check(!lifetime.expired(), "execution reset destroyed a live model cache");
        check(!moe_expert_cache_has_sources(other.moe_expert_cache),
              "Engine instances share registered MoE sources");
        for (int32_t expert : {0, 1, 0}) {
            auto ids = tensor(std::vector<int32_t>{expert}, options.dtype(kInt32)).reshape({1, 1});
            auto route = build_moe_route_plan(ids, 2);
            auto expected = resident.forward(execution, input, route).to(kFloat32);
            auto loaded_output = loaded.forward(execution, input, route).to(kFloat32);
            check((loaded_output - expected).abs().max().item<float>() == 0,
                  "MFE loading changed expert output");
            auto reference = mfe_dense_reference(loaded_cpu, input, {expert}, 1, 1, false);
            auto dense_output = matmul(input, dense.index({expert}).transpose(0, 1));
            check((reference - dense_output).abs().max().item<float>() == 0,
                  "loaded MFE reference differs from materialized weights");
            auto actual = weight.forward(execution, input, route).to(kFloat32);
            check((actual - expected).abs().max().item<float>() == 0,
                  "cache registration changed expert output");
        }
        auto retained_forward = weight.mixed_forward;
        weight = {};
        check(!lifetime.expired(), "copied projection lost its cache");
        auto ids = tensor(std::vector<int32_t>{1}, options.dtype(kInt32)).reshape({1, 1});
        auto route = build_moe_route_plan(ids, 2);
        auto actual = retained_forward(execution, input, route).to(kFloat32);
        auto expected = resident.forward(execution, input, route).to(kFloat32);
        check((actual - expected).abs().max().item<float>() == 0,
              "retained projection cannot use its cache");
        retained_forward = {};
        check(lifetime.expired(), "registered sources form a cache ownership cycle");
        check(bool(other.moe_expert_cache), "releasing one cache changed another Engine");
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

static void check_batch_state_ownership() {
    CudaExecutionContext execution;
    CudaExecutionScope scope(execution);
    Qwen35CausalLm model; model.execution = &execution;
    auto block = std::make_unique<FullBlock>();
    auto* full = block.get(); model.blocks.push_back(std::move(block));
    qwen35::Qwen35BatchStateAdapter adapter(model, 2, false);
    qwen35::QwenBatchRequestState first, second;
    adapter.prepare_prefill(first, 0, 2);
    full->state->cache = KVCache(1, 1, 8, 4);
    adapter.activate(first); adapter.resume_decode(2);
    auto decode = full->state;
    decode->decode_partial_o = mfq_tensor_backend::ones({4}, decode->cache.k.options());
    const auto* workspace = decode->decode_partial_o.data_ptr();
    adapter.suspend_decode(); adapter.prepare_prefill(second, 0, 2);
    full->state->cache = KVCache(1, 1, 8, 4);
    check(full->state != decode, "prefill reused live decode state");
    adapter.pause_prefill(second); adapter.resume_decode(2);
    check(full->state == decode && full->state->decode_partial_o.data_ptr() == workspace,
          "state binding discarded decode workspace");
    adapter.suspend_decode(); adapter.discard_prefill(second); adapter.resume_decode(2);
    check(full->state == decode && decode->decode_partial_o.sum().item<float>() == 4,
          "cancelled prefill corrupted decode state");
    adapter.release(first); adapter.finish_retire(0);
    check(first.slot() == -1 && second.slot() == -1, "request retained a retired slot");
}

static void check_snapshots(CudaExecutionContext& execution, bool hybrid) {
    Qwen35CausalLm model;
    model.execution = &execution;
    auto full = std::make_unique<FullBlock>();
    full->state->cache = KVCache(1, 1, 8, 4);
    model.blocks.push_back(std::move(full));
    if (hybrid) {
        auto linear = std::make_unique<qwen35::LinearAttentionBlock>();
        auto& config = linear->qwen_config;
        config.linear_conv_kernel_dim = 2;
        config.linear_num_key_heads = config.linear_num_value_heads = 1;
        config.linear_key_head_dim = config.linear_value_head_dim = 2;
        const auto options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
        linear->state->conv_state = mfq_tensor_backend::zeros({1, 1, 6}, options);
        linear->state->gdn_state = mfq_tensor_backend::zeros({1, 1, 2, 2}, options);
        model.blocks.push_back(std::move(linear));
    }
    model.cache_pos = 2;
    const std::vector<int64_t> prefix{1, 2};
    auto capture = model.capture_text_session_steps(prefix);
    auto first = capture.next();
    check(first.state == mfq::StepState::advanced && !first.value, "snapshot did not yield after one layer");
    capture = {}; // Cancelling capture must leave the live request state untouched.
    check(model.cache_pos == 2, "cancelled snapshot changed cache position");
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
    const std::vector<int64_t> prompt{1, 2, 3};
    const std::string session = "text", key;
    auto restore = memory.restore_steps(model, nullptr, session, prompt, 2, key);
    check(!restore.next().value, "restore did not yield after one layer");
    memory.close_session("text"); // The suspended restore owns its buffers.
    auto restored = mfq::finish_steps(std::move(restore));
    check(restored.tokens == 2, "closing the cache invalidated an active restore");
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
        mixed.store("cold", text);
        paged->flush();
        paged->trim_hot(0);
        {
            auto cold = mixed.restore_steps(model, nullptr, session, prompt, 2, key);
            auto step = cold.next();
            check(step.state != mfq::StepState::complete && !step.value,
                  "cold restore completed synchronously");
            const auto cancel_start = std::chrono::steady_clock::now();
            cold = {};
            check(std::chrono::steady_clock::now() - cancel_start < std::chrono::milliseconds(100),
                  "cold restore cancellation joined a disk task");
        }
        check(mixed.restore_best(model, nullptr, "cold", prompt, 2).tokens == 2,
              "cold restore after cancellation missed");
        mixed.close_session("cold");
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
        // Offline fixture cleanup waits for durable removal explicitly.
        paged->clear();
    }
    std::filesystem::remove_all(directory);
}

static void check_media_steps(mfq::engine::Engine& engine, const char* model_path) {
    using namespace mfq::engine;
    if (!engine.info().capabilities.image_input) return;
    const auto source = mfq::open_model_source(model_path);
    const auto config = mfq::models::qwen35::Config::from_source(*source);
    const auto& vision = config.grid_vision.value();
    const auto side = static_cast<int32_t>(vision.spatial_merge_size);
    EngineRequest image;
    image.id = "image";
    image.input.chat_input.emplace().preformatted_prompt =
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image."
        "<|im_end|>\n<|im_start|>assistant\n";
    image.input.sampling.temperature = 0;
    image.input.sampling.top_k = 1;
    image.input.sampling.max_tokens = 8;
    image.input.sampling.enable_mtp = false;
    auto& media = image.input.media.emplace();
    media.processor = MfqMultimodalProcessor::grid_vision;
    media.processor_name = mfq::kMfqGridVisionInputContract;
    media.vision_grid = media.image_grid = {1, side, side};
    media.vision_grid_shape = media.image_grid_shape = {1, 3};
    media.vision_types = {1};
    media.pixel_shape = {side * side, vision.patch_width()};
    media.pixel_values.assign(side * side * vision.patch_width(), 0.1F);
    const auto run = [&] {
        check(engine.admit(EngineRequest(image)) == Admission::accepted, "image admission");
        std::vector<int64_t> tokens;
        int terminals = 0, preparation_steps = 0;
        bool preparing = true;
        for (int tick = 0; tick < 1000 && !terminals; ++tick) {
            auto result = engine.step({image.id});
            if (preparing && result.events.empty() && !result.advanced.empty()) ++preparation_steps;
            for (const auto& event : result.events) {
                if (auto* failure = std::get_if<Failed>(&event.data)) throw std::runtime_error(failure->message);
                if (std::holds_alternative<PrefillProgress>(event.data)) preparing = false;
                if (auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                terminals += terminal(event.data);
            }
            if (result.wake_at) std::this_thread::sleep_until(*result.wake_at);
        }
        check(terminals == 1 && !tokens.empty(), "image request did not finish");
        return std::pair{tokens, preparation_steps};
    };
    const auto [reference, steps] = run();
    check(steps >= vision.depth + 2, "image encoder did not yield between layers");
    for (const auto stop : {int64_t(0), int64_t(2), 2 + vision.depth / 2, 2 + vision.depth}) {
        auto interrupted = image;
        interrupted.input.media->pixel_values[0] += static_cast<float>(stop + 1); // Bypass the image cache.
        check(engine.admit(std::move(interrupted)) == Admission::accepted, "image cancellation admission");
        for (int tick = 0, advanced = 0; advanced < stop; ++tick) {
            check(tick < 1000, "image preparation stalled");
            auto result = engine.step({image.id});
            check(result.events.empty(), "image preparation crossed its quantum");
            advanced += result.advanced.size();
            if (result.wake_at) std::this_thread::sleep_until(*result.wake_at);
        }
        engine.cancel(image.id);
        auto result = engine.step({});
        check(result.events.size() == 2 && std::holds_alternative<Cancelled>(result.events.back().data),
              "image cancellation did not release before its terminal");
        check(engine.step({}).events.empty(), "image emitted after cancellation");
        check(run().first == reference, "image cancellation changed subsequent output");
    }
    std::cout << "CUDA media step checks passed layers=" << vision.depth << " cancellation=4\n";
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
    check(engine->admit(EngineRequest(cancelled)) == Admission::accepted, "cancel admission");
    engine->cancel("cancel");
    auto result = engine->step({});
    check(std::count_if(result.events.begin(), result.events.end(), [](const auto& event) {
              return terminal(event.data);
          }) == 1 && std::holds_alternative<Cancelled>(result.events.back().data),
          "cancel before prefill did not release");
    cancelled.token_ids.assign(32, 101);
    check(engine->admit(EngineRequest(cancelled)) == Admission::accepted, "prefill cancel admission");
    for (int tick = 0; tick < 8; ++tick) {
        result = engine->step({"cancel"});
        if (!result.events.empty()) break;
    }
    check(result.events.size() == 1 && std::holds_alternative<PrefillProgress>(result.events[0].data),
          "prefill did not yield after one chunk");
    engine->cancel("cancel");
    result = engine->step({});
    check(std::count_if(result.events.begin(), result.events.end(), [](const auto& event) {
              return terminal(event.data);
          }) == 1 && std::holds_alternative<Cancelled>(result.events.back().data),
          "cancel during prefill did not release");
    check(mfq::cuda::diagnostics::check_engine_steps(*engine, {101, 202, 303}, sampling) == reference,
          "cancel changed subsequent output");
    EngineRequest session_request;
    session_request.id = "session";
    session_request.token_ids.assign(65, 101);
    session_request.input.sampling = sampling;
    session_request.input.sampling.max_tokens = 8;
    session_request.input.cache_plan.session_id = "session";
    session_request.input.cache_plan.stable_prefix_tokens = 64;
    const auto session_run = [&](int cancel_tick) {
        check(engine->admit(EngineRequest(session_request)) == Admission::accepted, "session admission");
        std::vector<int64_t> tokens;
        unsigned terminals = 0;
        for (int tick = 0; tick < 1000 && !terminals; ++tick) {
            if (tick == cancel_tick) engine->cancel("session");
            const auto start = Clock::now();
            auto step = engine->step({"session"});
            if (tick == cancel_tick) check(Clock::now() - start < std::chrono::milliseconds(100),
                "session cancellation blocked on cache work");
            (void)engine->control(RuntimeMetrics{});
            for (const auto& event : step.events) {
                if (const auto* failure = std::get_if<Failed>(&event.data)) throw std::runtime_error(failure->message);
                if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                if (terminal(event.data)) {
                    ++terminals;
                    check(std::holds_alternative<Cancelled>(event.data) == (cancel_tick >= 0),
                          "session returned the wrong terminal");
                }
            }
            if (step.wake_at) std::this_thread::sleep_until(*step.wake_at);
        }
        check(terminals == 1 && !engine->step({}).has_work, "session terminal ownership");
        return tokens;
    };
    const auto session_reference = session_run(-1);
    const auto previous_prompt = session_request.token_ids;
    session_request.token_ids.insert(session_request.token_ids.end(), session_reference.begin(), session_reference.end());
    session_request.input.cache_plan.stable_prefix_tokens = session_request.token_ids.size() - 1;
    session_run(3); // Cancel inside a layered snapshot restore.
    const auto restored_tokens = session_run(-1);
    engine->session({SessionCommand::Kind::clear});
    check(session_run(-1) == restored_tokens, "restored session changed greedy tokens");
    session_request.token_ids = previous_prompt;
    session_request.input.cache_plan.stable_prefix_tokens = 64;
    engine->session({SessionCommand::Kind::clear});
    session_run(12); // Eight prefill chunks followed by layered capture.
    check(session_run(-1) == session_reference, "cancelled capture changed greedy tokens");
    check_media_steps(*engine, model_path);
}

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    check_sampling_storage();
    check_batch_state_ownership();
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
