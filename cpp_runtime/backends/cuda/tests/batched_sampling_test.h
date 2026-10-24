#pragma once

#include "runtime/batch_sampling.h"
#include "token_constraint.h"
#include "tokenizer.h"
#include "chat.h"

static void check_batched_sampling_empty_mask(mfq::engine::MfqTokenizer& tokenizer) {
    namespace tb = mfq_tensor_backend;
    for (const auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        for (const int top_k : {1, 20}) {
            MfqSamplingParams params;
            params.top_k = top_k;
            params.seed = 73;
            Sampler sampler(params), reference(params);
            common_chat_params chat;
            chat.grammar = "root ::= (\"a\" | \"b\")*";
            auto constraint = mfq::engine::make_chat_token_constraint(tokenizer, chat);
            auto oracle_constraint = constraint->clone();
            auto host = tb::full({1, tokenizer.vocab_size()}, -INFINITY,
                tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            host.data_ptr<float>()[7] = 20.f;
            auto logits = host.to(tb::kCUDA).to(dtype);
            std::string expected, actual;
            try {
                (void)sample_logits(reference, logits.clone(), {}, oracle_constraint);
            } catch (const std::runtime_error& error) {
                expected = error.what();
            }
            BatchSampling sampling;
            try {
                (void)sampling.sample(logits, {{&sampler, {}, constraint}});
            } catch (const std::runtime_error& error) {
                actual = error.what();
            }
            check(expected == "chat-template grammar rejected every token" && actual == expected,
                "batch sampling changed an empty grammar distribution error");
            check(sampler.next_uniform_float() == reference.next_uniform_float(),
                "empty masked distribution consumed a retry draw");
        }
    }
    std::cout << "Batched sampling nonfinite masked distribution/RNG PASS cases=6\n";
}

static void check_batched_sampling(const char* tokenizer_path = nullptr) {
    namespace tb = mfq_tensor_backend;
    const auto context = default_context();
    const auto stream = mfq_current_cuda_stream();
    std::unique_ptr<mfq::engine::MfqTokenizer> tokenizer;
    if (tokenizer_path) tokenizer = std::make_unique<mfq::engine::MfqTokenizer>(tokenizer_path);
    int checks = 0;
    for (const int vocabulary : (tokenizer ? std::vector<int>{tokenizer->vocab_size()} :
            std::vector<int>{129, 4096, 248320})) {
        for (const int batch : {1, 4, 8, 16, 32}) {
            for (const auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
                for (const bool captured : {false, true}) {
                    context->begin_graph_pool(stream);
                    {
                        auto host = tb::empty({batch, vocabulary},
                            tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
                        auto logits = tb::empty({batch, vocabulary}, host.options().device(tb::kCUDA).dtype(dtype));
                        std::vector<Sampler> samplers, reference;
                        std::vector<tb::Tensor> counts, reference_counts;
                        std::vector<MfqTokenConstraintPtr> constraints, reference_constraints;
                        for (int row = 0; row < batch; ++row) {
                            MfqSamplingParams params;
                            const int kinds[] = {20, 0, 1, 64, 65, 257, 1024, 2};
                            params.top_k = std::min(vocabulary, kinds[row % 8]);
                            params.temperature = row % 9 == 8 ? 0.0 : .7 + .13 * (row % 5);
                            params.top_p = row % 3 == 0 ? 0.0 : row % 3 == 1 ? .81 : 1.0;
                            params.presence_penalty = row % 2 ? .3 : 0.0;
                            params.frequency_penalty = row % 3 ? -.07 : .21;
                            params.repetition_penalty = row % 2 ? 1.3 : .8;
                            if (row % 4 == 0) {
                                params.presence_penalty = 0;
                                params.frequency_penalty = 0;
                                params.repetition_penalty = 1;
                            }
                            params.seed = 73 + row * 173;
                            samplers.emplace_back(params);
                            reference.emplace_back(params);
                            samplers.back().discard_random(5 + row);
                            reference.back().discard_random(5 + row);
                            auto cpu_counts = tb::empty({vocabulary}, host.options().dtype(tb::kInt32));
                            for (int token = 0; token < vocabulary; ++token) {
                                cpu_counts.data_ptr<int32_t>()[token] = (token * 17 + row) % 4;
                            }
                            counts.push_back(cpu_counts.to(tb::kCUDA));
                            reference_counts.push_back(cpu_counts.to(tb::kCUDA));
                            MfqTokenConstraintPtr constraint;
                            if (tokenizer && row % 2 == 0) {
                                common_chat_params chat;
                                chat.grammar = "root ::= (\"a\" | \"b\")*";
                                constraint = mfq::engine::make_chat_token_constraint(*tokenizer, chat);
                            }
                            constraints.push_back(constraint);
                            reference_constraints.push_back(constraint ? constraint->clone() : nullptr);
                        }
                        BatchSampling sampling;
                        sampling.reserve(batch, vocabulary, logits.device());
                        const auto* address = sampling.metadata();
                        Graph graph;
                        tb::Tensor output;
                        for (int step = 0; step < 7; ++step) {
                            for (int row = 0; row < batch; ++row) {
                                for (int token = 0; token < vocabulary; ++token) {
                                    float value = step == 0 ? 0.f : step == 1 ?
                                        float((token * 37 + row) % 113 - 56) * .125f :
                                        std::sin(float(token + row * 23 + step * 19) * .037f) * 5.f;
                                    // Alternate grammar acceptance and rejection; keep ties
                                    // and cutoff edges in the unconstrained test matrix.
                                    if (step == 6 && !tokenizer) {
                                        if (token == 0) value = std::numeric_limits<float>::quiet_NaN();
                                        if (token == 1) value = -std::numeric_limits<float>::infinity();
                                        if (token == 2) value = -0.f;
                                        if (token == vocabulary - 1) value = std::numeric_limits<float>::infinity();
                                    }
                                    if (tokenizer) {
                                        value = token == (step % 2 ? 0 : 7) ? 20.f : value;
                                        if (token == tokenizer->eos_token()) value = -30.f;
                                    }
                                    host.data_ptr<float>()[int64_t(row) * vocabulary + token] = value;
                                }
                            }
                            logits.copy_(host.to(tb::kCUDA).to(dtype));
                            auto reference_logits = logits.clone();
                            std::vector<BatchSamplingInput> inputs(batch);
                            for (int row = 0; row < batch; ++row) {
                                const bool paused = step > 0 && row % 5 == step % 5;
                                if (!paused) inputs[row] = {&samplers[row], counts[row], constraints[row]};
                            }
                            sampling.prepare(logits, inputs);
                            check(address == sampling.metadata(), "batch sampling moved captured metadata");
                            if (captured) {
                                if (step == 0) {
                                    {
                                        GraphWarmupScope warmup(context, stream);
                                        output = sampling.run(logits);
                                    }
                                    MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                                    output = {};
                                    logits.copy_(reference_logits);
                                    graph.capture_begin_shared();
                                    output = sampling.run(logits);
                                    graph.capture_end();
                                }
                                graph.replay();
                            } else {
                                output = sampling.run(logits);
                            }
                            auto actual = sampling.finish(output);
                            for (int row = 0; row < batch; ++row) {
                                if (!inputs[row].sampler) continue;
                                auto expected = sample_logits(
                                    reference[row], reference_logits.narrow(0, row, 1).clone(),
                                    reference_counts[row], reference_constraints[row]);
                                const auto token = expected.item<int64_t>();
                                if (actual.host.data_ptr<int64_t>()[row] != token) {
                                    std::cerr << "batch sampling mismatch B=" << batch << " V=" << vocabulary
                                        << " dtype=" << int(dtype) << " graph=" << captured
                                        << " row=" << row << " step=" << step << " expected=" << token
                                        << " actual=" << actual.host.data_ptr<int64_t>()[row] << '\n';
                                    check(false, "batch sampling changed a token");
                                }
                                sample_token_counts_add_cuda(counts[row], actual.next.narrow(0, row, 1));
                                sample_token_counts_add_cuda(reference_counts[row], expected);
                                ++checks;
                            }
                        }
                        for (int row = 0; row < batch; ++row) {
                            check(samplers[row].next_uniform_float() == reference[row].next_uniform_float(),
                                "paused or constrained sampling changed RNG position");
                            auto actual = counts[row].cpu();
                            auto expected = reference_counts[row].cpu();
                            check(!std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()),
                                "batch sampling changed token counts");
                        }
                    }
                    context->end_graph_pool(stream);
                }
            }
        }
    }
    if (tokenizer) check_batched_sampling_empty_mask(*tokenizer);
    std::cout << "Batched sampling mixed parameters, penalties, pause/RNG and eager/graph PASS tokens="
        << checks << " grammar=" << bool(tokenizer) << '\n';
}

// A/B/A uses the retained single-row sampler and the same synthetic model,
// tokens, counts and seeds. Model load, allocations and warmup are untimed.
static void benchmark_batched_sampling(const char* model_path) {
    namespace tb = mfq_tensor_backend;
    using Clock = std::chrono::steady_clock;
    CudaExecutionContext execution;
    auto model = load_causal_lm<Qwen35CausalLm>(execution, model_path, "", 256);
    const auto context = default_context();
    const auto stream = mfq_current_cuda_stream();
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    constexpr int warmup = 16;
    constexpr int repetitions = 128;
    for (const int batch : {1, 4, 8, 16, 32}) {
        std::vector<int64_t> reference_tokens;
        for (int arm = 0; arm < 3; ++arm) {
            const bool batched = arm == 1;
            model.reset(batch);
            auto ids = tb::ones({batch, 1}, options);
            std::vector<Sampler> samplers;
            std::vector<BatchSamplingInput> inputs;
            for (int row = 0; row < batch; ++row) {
                MfqSamplingParams params;
                params.temperature = .8;
                params.top_k = std::min(20, int(model.vocab_size()));
                params.top_p = .9;
                params.presence_penalty = .2;
                params.frequency_penalty = .01;
                params.repetition_penalty = 1.15;
                params.seed = 20261023 + row;
                samplers.emplace_back(params);
            }
            for (int row = 0; row < batch; ++row) {
                inputs.push_back({&samplers[row], tb::zeros({model.vocab_size()}, options.dtype(tb::kInt32)), {}});
            }
            BatchSampling sampling;
            sampling.reserve(batch, int(model.vocab_size()), ids.device());
            double host_us = 0.0;
            double step_us = 0.0;
            std::vector<double> sampling_times, step_times;
            int readbacks = 0;
            std::vector<int64_t> tokens;
            tb::Tensor logits;
            for (int step = 0; step < warmup + repetitions; ++step) {
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto start = Clock::now();
                logits = model.forward(ids).reshape({batch, -1}).contiguous();
                // Match the existing engine's dependency on completed logits.
                const auto sample_start = Clock::now();
                if (batched) {
                    auto result = sampling.sample(logits, inputs);
                    ids = result.next.reshape({batch, 1});
                    if (step >= warmup) {
                        ++readbacks;
                        tokens.insert(tokens.end(), result.host.data_ptr<int64_t>(),
                            result.host.data_ptr<int64_t>() + batch);
                    }
                } else {
                    std::vector<tb::Tensor> next;
                    for (int row = 0; row < batch; ++row) {
                        next.push_back(sample_logits(samplers[row], logits.narrow(0, row, 1), inputs[row].counts));
                        const auto token = next.back().item<int64_t>();
                        if (step >= warmup) {
                            ++readbacks;
                            tokens.push_back(token);
                        }
                    }
                    ids = tb::cat(next, 0).reshape({batch, 1});
                }
                const auto sample_end = Clock::now();
                for (int row = 0; row < batch; ++row) {
                    sample_token_counts_add_cuda(inputs[row].counts, ids.narrow(0, row, 1).reshape({1}));
                }
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto end = Clock::now();
                if (step >= warmup) {
                    const auto sampling_time =
                        std::chrono::duration<double, std::micro>(sample_end - sample_start).count();
                    const auto step_time = std::chrono::duration<double, std::micro>(end - start).count();
                    host_us += sampling_time;
                    step_us += step_time;
                    sampling_times.push_back(sampling_time);
                    step_times.push_back(step_time);
                }
            }
            if (arm == 0) reference_tokens = tokens;
            else check(tokens == reference_tokens, "sampling A/B/A synthetic decode changed tokens");

            // Capture just the penalties/sampling kernels to measure GPU work
            // separately from host dispatch gaps and synchronous readbacks.
            auto random_host = tb::empty({batch}, logits.options().device(tb::kCPU)
                .dtype(tb::kFloat32).pinned_memory(true));
            for (int row = 0; row < batch; ++row) {
                auto probe = samplers[row];
                random_host.data_ptr<float>()[row] = probe.next_uniform_float();
            }
            sampling.prepare(logits, inputs);
            auto random = random_host.to(tb::kCUDA);
            const auto kernels = [&] {
                if (batched) {
                    (void)sampling.run(logits);
                } else {
                    for (int row = 0; row < batch; ++row) {
                        auto values = samplers[row].apply_penalties(logits.narrow(0, row, 1), inputs[row].counts);
                        (void)samplers[row].ops().sample_stochastic(values, random.narrow(0, row, 1),
                            samplers[row].params());
                    }
                }
            };
            constexpr int gpu_repetitions = 128;
            const auto repeated_kernels = [&] {
                for (int i = 0; i < gpu_repetitions; ++i) kernels();
            };
            context->begin_graph_pool(stream);
            float elapsed = 0;
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin));
            MFQ_CUDA_CHECK(cudaEventCreate(&end));
            {
                {
                    GraphWarmupScope scope(context, stream);
                    repeated_kernels();
                }
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                Graph graph;
                graph.capture_begin_shared();
                // Device-side timing excludes CPU submission pauses between samples.
                MFQ_CUDA_CHECK(cudaEventRecordWithFlags(begin, stream, cudaEventRecordExternal));
                repeated_kernels();
                MFQ_CUDA_CHECK(cudaEventRecordWithFlags(end, stream, cudaEventRecordExternal));
                graph.capture_end();
                graph.replay();
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                graph.replay();
                MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
                MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
            }
            MFQ_CUDA_CHECK(cudaEventDestroy(begin));
            MFQ_CUDA_CHECK(cudaEventDestroy(end));
            context->end_graph_pool(stream);
            const auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                return (values[values.size() / 2 - 1] + values[values.size() / 2]) * .5;
            };
            std::cout << "batch_sampling_bench B=" << batch << " arm=" << (batched ? "B" : arm ? "A2" : "A1")
                << " vocab=" << model.vocab_size() << " k=20 p=0.9 temperature=0.8 penalties=1"
                << " gpu_sampling_us=" << elapsed * 1000 / gpu_repetitions
                << " sampling_host_us=" << host_us / repetitions
                << " sampling_host_p50_us=" << median(sampling_times) << " d2h_per_step=" << readbacks / repetitions
                << " step_host_us=" << step_us / repetitions
                << " step_host_p50_us=" << median(step_times) << " reps=" << repetitions << std::endl;
        }
    }
}

static void check_batched_sampling_cutoffs() {
    namespace tb = mfq_tensor_backend;
    constexpr int batch = 5;
    constexpr int vocabulary = 129;
    int checks = 0;
    for (const auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        for (const float top_p : {0.f, .8f, 1.f}) {
            for (const float uniform : {0.f, .137f, .731f, .99999994f}) {
                auto host = tb::empty({batch, vocabulary},
                    tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
                auto packet = tb::empty({int64_t(batch * sizeof(SamplingRow))},
                    host.options().dtype(tb::kUInt8).pinned_memory(true));
                auto* rows = static_cast<SamplingRow*>(packet.data_ptr());
                SamplingPlan plan;
                plan.rows = plan.candidates = batch;
                plan.radix_k = 64;
                plan.softmax = plan.fallback = true;
                const int ks[] = {2, 20, 64, 65, 0};
                std::vector<SamplingOps::Params> params(batch);
                for (int row = 0; row < batch; ++row) {
                    rows[row] = SamplingRow{};
                    rows[row].source = row;
                    rows[row].top_k = ks[row];
                    rows[row].top_p = top_p;
                    rows[row].temperature = .7f;
                    rows[row].random = uniform;
                    rows[row].kind = row < 3 ? SamplingKind::radix :
                        row == 3 ? SamplingKind::fallback : SamplingKind::softmax;
                    const auto* radix = std::getenv("MFQ_SAMPLE_RADIX_TOPK");
                    if (radix && radix[0] == '0' && row < 3) rows[row].kind = SamplingKind::fallback;
                    params[row].top_k = ks[row];
                    params[row].top_p = top_p;
                    params[row].temperature = .7f;
                    for (int token = 0; token < vocabulary; ++token) {
                        host.data_ptr<float>()[row * vocabulary + token] = uniform == 0.f ? 0.f :
                            token % 4 == 0 ? -0.f : float((token * 17) % 11 - 5) * .125f;
                    }
                }
                auto logits = host.to(tb::kCUDA).to(dtype);
                auto device = packet.to(tb::kCUDA);
                auto actual = sample_batch_cuda(logits, static_cast<const SamplingRow*>(device.data_ptr()), plan).cpu();
                SamplingOps ops;
                for (int row = 0; row < batch; ++row) {
                    auto expected = ops.sample_stochastic(logits.narrow(0, row, 1), uniform, params[row]);
                    check(actual.data_ptr<int64_t>()[row] == expected.item<int64_t>(),
                        "batch sampling changed cutoff, tie or uniform endpoint semantics");
                    ++checks;
                }
            }
        }
    }
    std::cout << "Batched sampling cutoff/tie/uniform endpoints PASS tokens=" << checks << '\n';
}
