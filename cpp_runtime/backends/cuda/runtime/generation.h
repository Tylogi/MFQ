#pragma once
#include "cuda_execution.h"
#include "decode_graph.h"
#include "mfq_tensor_backend.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>

namespace mfq::cuda::internal {
template <typename Model>
int generate_cli_tokens(Model& model, mfq_tensor_backend::Tensor ids,
        int gen, bool profile,
        std::chrono::steady_clock::time_point t0,
        std::chrono::steady_clock::time_point t1) {
        g_profiler.reset();
        auto next = model.next_token(ids);
        mfq_cuda_synchronize();
        auto t2 = std::chrono::steady_clock::now();
        report_cuda_memory("prefill");
        const char * empty_cache_env = std::getenv("MFQ_EMPTY_CACHE_BEFORE_GRAPH");
        if (empty_cache_env != nullptr && std::atoi(empty_cache_env) != 0) {
            mfq_cuda_empty_cache();
            report_cuda_memory("prefill_empty_cache");
        }
        g_profiler.report("prefill");
        g_profiler.reset();
        if (gen == 0) return 0;
        auto generated_cuda = mfq_tensor_backend::empty({gen}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        cudaStream_t stream = mfq_get_current_cuda_stream().stream();
        MFQ_CUDA_CHECK(cudaMemcpyAsync(generated_cuda.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                   sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
        const char* graph_env = std::getenv("MFQ_CUDA_GRAPH");
        const char * profile_graph_env = std::getenv("MFQ_PROFILE_CUDA_GRAPH");
        const bool profile_cuda_graph = profile && profile_graph_env != nullptr &&
            std::atoi(profile_graph_env) != 0;
        bool use_cuda_graph =
            (graph_env == nullptr || graph_env[0] != '0') &&
            !Model::is_flash_next &&
            mfq_cuda_graph_capture_supported() &&
            g_dsv4_cpu_offload_layers.empty() &&
            g_dense_cpu_layer_count == 0 &&
            !g_moe_expert_cache &&
            model_parallel_cuda_graph_enabled() &&
            (!profile || profile_cuda_graph) && gen > 1;
        const char * cuda_profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
        const bool cuda_profiler_range = cuda_profiler_env != nullptr &&
            std::atoi(cuda_profiler_env) != 0;
        auto decode_replay_t0 = t2;
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStart());
        if (use_cuda_graph) {
            auto static_input = mfq_tensor_backend::empty({1, 1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_pos = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_len = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto static_step = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
            auto graph_stream = mfq_get_stream_from_pool(false);
            MfqCudaGuard graph_device_guard(
                graph_stream.device_index());
            auto graph_compute_streams =
                make_cuda_graph_compute_streams(graph_stream);
            auto graph_stream_guards =
                activate_cuda_graph_compute_streams(
                    graph_compute_streams);
            cudaStream_t graph_raw_stream = graph_stream.stream();

            int64_t pos_h = model.cache_pos;
            int64_t len_h = pos_h + 1;
            int64_t step_h = 1;
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_input.template data_ptr<int64_t>(), next.template data_ptr<int64_t>(),
                                       sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_pos.template data_ptr<int64_t>(), &pos_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_len.template data_ptr<int64_t>(), &len_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaMemcpyAsync(static_step.template data_ptr<int64_t>(), &step_h,
                                       sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

            MfqCudaGraph graph;
            mfq_tensor_backend::Tensor static_next;
            const int64_t planned_len = model.cache_pos + gen;
            const int64_t attention_parts = decode_graph_attention_parts(
                planned_len, FullBlock::kDecodeAttentionMaxParts);
            {
                DecodeGraphBranchScope branch_scope;
                prepare_decode_graph_memory(model, graph, [&]() {
                    (void)model.next_token_static(
                        static_input, static_pos, static_len,
                        planned_len, attention_parts);
                }, cuda_graph_participant_streams(
                    graph_compute_streams));
                g_profiler.reset();
                g_profiler.graph_events = profile_cuda_graph;
                graph.capture_begin();
                static_next = g_profiler.measure("decode.model_total", [&]() {
                    return model.next_token_static(
                        static_input, static_pos, static_len,
                        planned_len, attention_parts);
                });
                g_profiler.measure("decode.commit", [&]() {
                    decode_graph_commit_cuda(
                        static_next, generated_cuda, static_step,
                        static_input, static_pos, static_len);
                    return 0;
                });
                graph.capture_end();
            }
            mfq_debug_dump_cuda_graph(graph);
            report_cuda_memory("graph_captured");

            decode_replay_t0 = std::chrono::steady_clock::now();
            for (int i = 1; i < gen; ++i) {
                graph.replay();
            }
            MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));
        } else {
            for (int i = 1; i < gen; ++i) {
                next = g_profiler.measure("decode.eager_model", [&]() {
                    return model.next_token(next.view({1, 1}));
                });
                g_profiler.measure("decode.eager_commit", [&]() {
                    MFQ_CUDA_CHECK(cudaMemcpyAsync(
                        generated_cuda.template data_ptr<int64_t>() + i,
                        next.template data_ptr<int64_t>(), sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream));
                    return 0;
                });
            }
        }
        mfq_cuda_synchronize();
        if (cuda_profiler_range) MFQ_CUDA_CHECK(cudaProfilerStop());
        auto t3 = std::chrono::steady_clock::now();
        g_profiler.report("decode");
        auto generated_tensor = generated_cuda.to(mfq_tensor_backend::kCPU).contiguous();
        auto generated_ptr = generated_tensor.template data_ptr<int64_t>();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_s = std::chrono::duration<double>(t2 - t1).count();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();
        double decode_replay_s = std::chrono::duration<double>(t3 - decode_replay_t0).count();
        std::cout << "load_sec=" << load_s << "\n";
        std::cout << "prefill_sec=" << prefill_s << "\n";
        std::cout << "decode_tokens=" << std::max(0, gen - 1) << "\n";
        std::cout << "decode_setup_sec=" << (decode_s - decode_replay_s) << "\n";
        std::cout << "decode_replay_sec=" << decode_replay_s << "\n";
        std::cout << "decode_sec=" << decode_s << "\n";
        if (gen > 1) std::cout << "decode_tok_per_s=" << (double)(gen - 1) / decode_s << "\n";
        if (gen > 1) std::cout << "decode_steady_tok_per_s=" << (double)(gen - 1) / decode_replay_s << "\n";
        std::cout << "generated_ids=";
        for (int64_t i = 0; i < generated_tensor.numel(); ++i) {
            if (i) std::cout << ",";
            std::cout << generated_ptr[i];
        }
        std::cout << "\n";
        if (g_moe_expert_cache) {
            print_moe_expert_cache_stats(std::cout);
        }
        return 0;
 }

} // namespace mfq::cuda::internal
