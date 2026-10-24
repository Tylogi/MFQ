#pragma once

#include "gdn_prefill_chunk.cuh"

struct GdnChunkInputs {
    Tensor q, k, v, g, beta, state;
    GdnChunkInputs(int batch, int hq, int hv, int tokens, int width, tb::ScalarType dtype) {
        const float amplitude = 1.f / std::sqrt(float(width));
        q = data({batch, hq, tokens, width}, 451, amplitude).to(dtype);
        k = data({batch, hq, tokens, width}, 452, amplitude).to(dtype);
        v = data({batch, hv, tokens, width}, 453, .7f).to(dtype);
        g = data({batch, hv, tokens}, 454, .03f).to(tb::kFloat32) - .04f;
        beta = data({batch, hv, tokens}, 455, .2f).to(tb::kFloat32) + .5f;
        state = data({batch, hv, width, width}, 456, .17f).to(tb::kFloat32);
    }
    std::vector<Tensor> run(bool transposed, bool tiled, bool initial, bool inplace = false) const {
        if (tiled || inplace) {
            auto s = initial ? state.clone() : tb::zeros(state.sizes(), state.options());
            auto fn = tiled ? (transposed ? gdn_inplace_transposed_tiled_cuda : gdn_inplace_tiled_cuda)
                            : (transposed ? gdn_inplace_transposed_cuda : gdn_inplace_cuda);
            auto result = fn(q, k, v, g, beta, s);
            if (result[1].data_ptr() != s.data_ptr()) throw std::runtime_error("GDN chunk replaced inplace state");
            return result;
        }
        MfqOptional<Tensor> s;
        if (initial) s = state;
        return (transposed ? gdn_transposed_cuda : gdn_cuda)(q, k, v, g, beta, s);
    }
};

void gdn_chunk_compare(const std::vector<Tensor>& actual, const std::vector<Tensor>& expected) {
    equal(actual[0], expected[0], "GDN chunk output");
    equal(actual[1], expected[1], "GDN chunk final state");
}

void gdn_prefill_chunk_case() {
    GdnPipelineScope option(0, "MFQ_GDN_PREFILL_CHUNK");
    {
        GdnPipelineScope automatic(-1, "MFQ_GDN_PREFILL_CHUNK");
        GdnColumnsScope columns(-1);
        if (gdn_prefill_chunk_enabled(1024, true) || !gdn_prefill_chunk_enabled(4096, true) ||
            gdn_prefill_chunk_enabled(4096, false))
            throw std::runtime_error("GDN chunk automatic threshold/KDA dispatch differs");
        GdnColumnsScope legacy(4);
        if (gdn_prefill_chunk_enabled(4096, true))
            throw std::runtime_error("GDN chunk ignored explicit recurrent columns");
        GdnPipelineScope::set(1, "MFQ_GDN_PREFILL_CHUNK");
        if (!gdn_prefill_chunk_enabled(63, true) || gdn_prefill_chunk_enabled(8, true))
            throw std::runtime_error("GDN chunk force/verification dispatch differs");
    }
    int cases = 0;
    const auto check = [&](GdnChunkInputs& input, bool transposed, bool tiled, bool initial, bool inplace) {
        GdnPipelineScope::set(0, "MFQ_GDN_PREFILL_CHUNK");
        auto original_state = input.state.clone();
        auto expected = input.run(transposed, tiled, initial, inplace);
        GdnPipelineScope::set(1, "MFQ_GDN_PREFILL_CHUNK");
        gdn_chunk_compare(input.run(transposed, tiled, initial, inplace), expected);
        equal(input.state, original_state, "GDN chunk initial state was modified", 0, 0);
        ++cases;
    };
    for (auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16})
    for (int width : {32, 64, 128}) for (int tokens : {63, 64, 65}) {
        GdnChunkInputs input(2, 2, 6, tokens, width, dtype);
        for (bool transposed : {false, true}) for (bool tiled : {false, true}) for (bool initial : {false, true})
            check(input, transposed, tiled, initial, initial);
    }
    for (auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        for (auto heads : {std::pair<int, int>{1, 1}, {2, 4}, {16, 48}, {16, 64}}) {
            GdnChunkInputs input(1, heads.first, heads.second, 257, 128, dtype);
            for (bool transposed : {false, true}) check(input, transposed, false, true, false);
        }
        GdnChunkInputs long_input(1, 2, 6, 16384, 128, dtype);
        for (bool initial : {false, true}) check(long_input, true, true, initial, initial);
        for (float gate : {0.f, -1.f, -20.f}) {
            GdnChunkInputs input(1, 2, 6, 129, 128, dtype);
            input.g = tb::ones(input.g.sizes(), input.g.options()) * gate;
            check(input, false, false, true, false);
        }
        // Decode and MTP verification must stay on the recurrent path even
        // when the chunk option is enabled explicitly.
        for (int tokens : {1, 2, 8}) {
            GdnChunkInputs input(1, 2, 6, tokens, 128, dtype);
            check(input, true, false, true, true);
        }
    }
    GdnChunkInputs large_state(1, 2, 6, 129, 128, tb::kFloat32);
    large_state.state = large_state.state * 1e6f;
    large_state.q = tb::zeros(large_state.q.sizes(), large_state.q.options());
    large_state.k = tb::zeros(large_state.k.sizes(), large_state.k.options());
    check(large_state, false, false, true, false);
    GdnChunkInputs strided_state(1, 2, 6, 65, 128, tb::kFloat32);
    strided_state.state = strided_state.state.transpose(2, 3);
    check(strided_state, true, false, true, false);
    GdnChunkInputs window_boundary(1, 2, 6, 16385, 128, tb::kFloat32);
    check(window_boundary, true, false, true, false);
    {
        // A nearly full device shrinks the workspace window; many small windows
        // must give the same output and state as one window over the prompt.
        GdnPipelineScope window(3, "MFQ_GDN_PREFILL_CHUNK_WINDOW");
        for (auto dtype : {tb::kFloat32, tb::kBFloat16}) {
            GdnChunkInputs small_windows(2, 2, 6, 1000, 128, dtype);
            check(small_windows, true, false, true, false);
        }
    }
    {
        // No workspace budget must decline the chunk path before touching
        // in-place state, then produce the recurrent result.
        GdnPipelineScope window(0, "MFQ_GDN_PREFILL_CHUNK_WINDOW");
        for (auto dtype : {tb::kFloat32, tb::kBFloat16}) {
            GdnChunkInputs no_workspace(2, 2, 6, 129, 128, dtype);
            auto before = no_workspace.state.clone();
            auto result = gdn_prefill_chunk_cuda(no_workspace.q, no_workspace.k, no_workspace.v,
                no_workspace.g, no_workspace.beta, no_workspace.state, true, true, false);
            if (!result.empty()) throw std::runtime_error("GDN chunk exceeded zero workspace budget");
            equal(no_workspace.state, before, "GDN declined chunk modified state", 0, 0);
            check(no_workspace, true, false, true, true);
        }
    }
    std::cout << "GDN chunk output/state cases=" << cases << " FP32/FP16/BF16 PASS\n";
}

void gdn_prefill_chunk_varlen_case() {
    GdnPipelineScope option(0, "MFQ_GDN_PREFILL_CHUNK");
    const std::vector<int64_t> lengths = {0, 1, 8, 63, 64, 65, 129, 257};
    int cases = 0;
    for (int window_chunks : {-1, 0}) {
    GdnPipelineScope window(window_chunks, "MFQ_GDN_PREFILL_CHUNK_WINDOW");
    for (auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16})
    for (bool transposed : {false, true}) for (bool tiled : {false, true}) for (bool initial : {false, true}) {
        GdnChunkInputs input(lengths.size(), 2, 6, 257, 128, dtype);
        auto state = initial ? input.state.clone() : tb::zeros(input.state.sizes(), input.state.options());
        auto actual = gdn_prefill_varlen_cuda(input.q, input.k, input.v, input.g, input.beta,
            lengths, state, transposed, tiled, true);
        if (actual[1].data_ptr() != state.data_ptr()) throw std::runtime_error("GDN varlen replaced state");
        for (size_t b = 0; b < lengths.size(); ++b) {
            const int tokens = lengths[b];
            auto expected_state = initial ? input.state.narrow(0, b, 1).clone()
                : tb::zeros({1, 6, 128, 128}, input.state.options());
            if (tokens) {
                const auto slice = [&](const Tensor& x) {
                    return x.narrow(0, b, 1).narrow(2, 0, tokens).contiguous();
                };
                auto fn = tiled ? (transposed ? gdn_inplace_transposed_tiled_cuda : gdn_inplace_tiled_cuda)
                                : (transposed ? gdn_inplace_transposed_cuda : gdn_inplace_cuda);
                auto expected = fn(slice(input.q), slice(input.k), slice(input.v),
                    slice(input.g), slice(input.beta), expected_state);
                equal(actual[0].narrow(0, b, 1).narrow(2, 0, tokens), expected[0], "GDN varlen output");
                expected_state = expected[1];
            }
            equal(actual[1].narrow(0, b, 1), expected_state, "GDN varlen final state");
            if (tokens < 257) {
                auto padding = actual[0].narrow(0, b, 1).narrow(2, tokens, 257 - tokens);
                equal(padding, tb::zeros(padding.sizes(), padding.options()), "GDN varlen padding", 0, 0);
            }
            ++cases;
        }
    }
    }
    // Requests which finish in the first workspace window must retain their
    // final state while another request advances beyond the window boundary.
    GdnChunkInputs long_input(3, 1, 2, 16385, 128, tb::kBFloat16);
    auto actual = gdn_prefill_varlen_cuda(long_input.q, long_input.k, long_input.v,
        long_input.g, long_input.beta, {0, 65, 16385}, long_input.state, true);
    equal(actual[1].narrow(0, 0, 1), long_input.state.narrow(0, 0, 1), "GDN empty long request", 0, 0);
    for (int b : {1, 2}) {
        const int tokens = b == 1 ? 65 : 16385;
        const auto slice = [&](const Tensor& x) {
            return x.narrow(0, b, 1).narrow(2, 0, tokens).contiguous();
        };
        auto expected = gdn_transposed_cuda(slice(long_input.q), slice(long_input.k), slice(long_input.v),
            slice(long_input.g), slice(long_input.beta), long_input.state.narrow(0, b, 1).contiguous());
        equal(actual[0].narrow(0, b, 1).narrow(2, 0, tokens), expected[0], "GDN long varlen output");
        equal(actual[1].narrow(0, b, 1), expected[1], "GDN long varlen state");
    }
    cases += 3;
    std::cout << "GDN chunk independent varlen cases=" << cases << " PASS\n";
}

void gdn_prefill_chunk_graph_case() {
    GdnPipelineScope option(1, "MFQ_GDN_PREFILL_CHUNK");
    GdnChunkInputs input(1, 2, 6, 129, 128, tb::kFloat32);
    auto state = input.state.clone();
    std::vector<Tensor> actual;
    DecodeWindow window(mfq_current_cuda_stream());
    window.capture([&] {
        actual = gdn_inplace_transposed_cuda(input.q, input.k, input.v, input.g, input.beta, state);
    }, [&] { state.copy_(input.state); }, [&] { actual.clear(); });
    for (float amplitude : {.5f, 1.f, 1.5f}) {
        input.q.copy_(data(input.q.sizes().vec(), 461, .08f * amplitude).to(tb::kFloat32));
        input.g.copy_(data(input.g.sizes().vec(), 462, .03f * amplitude).to(tb::kFloat32) - .05f);
        state.copy_(input.state);
        GdnPipelineScope::set(0, "MFQ_GDN_PREFILL_CHUNK");
        auto expected = input.run(true, false, true);
        window.run();
        gdn_chunk_compare(actual, expected);
        if (actual[1].data_ptr() != state.data_ptr()) throw std::runtime_error("GDN chunk graph replaced state");
    }
    std::cout << "GDN chunk changing-input graph/state PASS\n";
}

// Alternate paired trials, omit warmup, report median CUDA-event latency.
template<class Call>
std::pair<float, float> gdn_chunk_timing(Call call, int iterations = 3) {
    std::vector<float> times[2];
    for (int sample = 0; sample < 6; ++sample) for (int order = 0; order < 2; ++order) {
        const int mode = (sample + order) % 2;
        GdnPipelineScope::set(mode, "MFQ_GDN_PREFILL_CHUNK");
        call();
        Event start(cudaEventDefault), end(cudaEventDefault);
        const auto stream = mfq_current_cuda_stream();
        start.record(stream);
        for (int i = 0; i < iterations; ++i) call();
        end.record(stream);
        end.synchronize();
        float elapsed = 0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), end.get()));
        if (sample) times[mode].push_back(elapsed / iterations);
    }
    for (auto& values : times) std::sort(values.begin(), values.end());
    return {times[0][2], times[1][2]};
}

void gdn_prefill_chunk_benchmark(bool shapes = false) {
    GdnColumnsScope columns(-1);
    GdnPipelineScope staged(-1, "MFQ_GDN_PREFILL_STAGED");
    GdnPipelineScope pipeline(-1);
    GdnPipelineScope option(0, "MFQ_GDN_PREFILL_CHUNK");
    const std::vector<std::array<int, 4>> geometries = shapes
        ? std::vector<std::array<int, 4>>{{1, 2, 6, 32}, {1, 2, 6, 64}, {1, 2, 6, 128},
            {1, 16, 48, 32}, {1, 16, 48, 64}, {1, 16, 64, 128}, {2, 2, 6, 128}}
        : std::vector<std::array<int, 4>>{{1, 16, 48, 128}};
    const std::vector<int> lengths = shapes ? std::vector<int>{128, 256, 1024, 4096}
        : std::vector<int>{32, 64, 128, 256, 512, 1024, 4096, 16384, 65536};
    for (const auto& geometry : geometries) for (int tokens : lengths) {
        const auto [batch, hq, hv, width] = geometry;
        GdnChunkInputs input(batch, hq, hv, tokens, width, tb::kFloat32);
        auto expected = input.run(true, false, true);
        GdnPipelineScope::set(1, "MFQ_GDN_PREFILL_CHUNK");
        gdn_chunk_compare(input.run(true, false, true), expected);
        auto times = gdn_chunk_timing([&] { input.run(true, false, true); });
        std::cout << "gdn_chunk_bench tokens=" << tokens << " batch=" << batch << " key_heads=" << hq
                  << " value_heads=" << hv << " width=" << width
                  << " baseline_ms=" << times.first << " chunk_ms=" << times.second
                  << " speedup=" << times.first / times.second << std::endl;
    }
}

