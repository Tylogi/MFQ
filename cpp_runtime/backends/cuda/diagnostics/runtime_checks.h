#pragma once

#include "engine/generation.h"
#include "generation_result.h"
#include "models/deepseek_v4/ops.h"
#include "models/qwen4_exp/ops.h"
#include "diagnostics/flash_next_mtp.h"
#include "qwen35/mtp.h"
#include "quant_linear.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "storage/text_session_cache.h"
#include "cuda_execution.h"
#include "qwen35/linear_attention.h"
#include "mfq/kernels/cuda/deepseek_v41.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

namespace mfq::cuda {
struct CudaEngineOptions;
}

namespace mfq::cuda::diagnostics {

using namespace mfq::cuda::internal;

int run_qwen35_mtp_bench(
    mfq::cuda::Qwen35CausalLm& model,
    Qwen35Mtp& mtp,
    bool enable_mtp,
    int generated_tokens,
    int repetitions);
int run_cuda_continuous_batching_check(
    mfq::cuda::Qwen35CausalLm& model);
int run_cuda_engine_isolation_check(CudaEngineOptions options);

template <typename Model>
static int run_prefill_sweep(
    Model& model,
    const std::vector<int64_t> & sizes,
    int repeats,CudaExecutionContext& execution) {
    if (repeats < 1) throw std::runtime_error("--prefill-sweep-reps must be positive");
    const int64_t max_m = *std::max_element(sizes.begin(), sizes.end());
    if (max_m > model.max_position_embeddings()) {
        throw std::runtime_error("--prefill-sweep exceeds the configured context size");
    }

    std::vector<int64_t> token_ids((size_t)max_m);
    const int64_t token_span = std::max<int64_t>(
        1, std::min<int64_t>(1024, model.vocab_size() - 2));
    for (int64_t i = 0; i < max_m; ++i) token_ids[(size_t)i] = 1 + i % token_span;
    auto all_ids = mfq_tensor_backend::tensor(
        token_ids, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA)).unsqueeze(0);

    if(const auto* chunk_option=std::getenv("MFQ_PREFILL_SWEEP_CHUNKS");chunk_option && chunk_option[0]) {
        const bool profile=execution.profiler.enabled;
        const auto* flush_sweep_option=std::getenv("MFQ_PREFILL_LAYER_FLUSH_SWEEP");
        const bool flush_sweep=flush_sweep_option && std::atoi(flush_sweep_option)!=0;
        const auto* async_sweep_option=std::getenv("MFQ_PREFILL_LAYER_ASYNC_SWEEP");
        const bool async_sweep=async_sweep_option && std::atoi(async_sweep_option)!=0;
        const auto* phase_sweep_option=std::getenv("MFQ_PREFILL_LAYER_PHASE_SWEEP");
        const bool phase_sweep=phase_sweep_option && std::atoi(phase_sweep_option)!=0;
        const auto* layer_major_sweep_option=std::getenv("MFQ_PREFILL_LAYER_MAJOR_SWEEP");
        const bool layer_major_sweep=layer_major_sweep_option && std::atoi(layer_major_sweep_option)!=0;
        const auto* layer_major_option=std::getenv("MFQ_MOE_PREFILL_LAYER_MAJOR");
        const bool layer_major_requested=layer_major_option && std::atoi(layer_major_option)!=0;
        const auto* kernel_sweep_option=std::getenv("MFQ_PREFILL_KERNEL_SWEEP");
        const bool kernel_sweep=kernel_sweep_option && std::atoi(kernel_sweep_option)!=0;
        const bool gr_gdn_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==2;
        const bool exact_sweep=kernel_sweep && std::atoi(kernel_sweep_option)>=3;
        const bool exact_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)>=4;
        const bool default_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==5;
        const bool quant_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==6;
        const bool gr_tile_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==7;
        const bool gr_mix_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==8;
        const bool nvq_cohort_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==10;
        const bool nvq_k_pair_sweep=kernel_sweep && (std::atoi(kernel_sweep_option)==9 || nvq_cohort_pair_sweep);
        const char* nvq_trial_option=nvq_cohort_pair_sweep?"MFQ_NVQ_PREFILL_COHORTS":"MFQ_NVQ_PREFILL_NARROW_G2";
        const auto* chunk_oracle_option=std::getenv("MFQ_PREFILL_CHUNK_SWEEP_ORACLE");
        const bool chunk_sweep_oracle=chunk_oracle_option && std::atoi(chunk_oracle_option)!=0;
        std::vector<int64_t> chunks;
        std::stringstream options(chunk_option);std::string term;
        while(std::getline(options,term,',')) {
            std::size_t parsed=0;const auto value=std::stoll(term,&parsed);
            if(parsed!=term.size())throw std::invalid_argument("invalid prefill chunk size");
            chunks.push_back(value);
        }
        if(chunks.empty() || std::any_of(chunks.begin(),chunks.end(),[](auto n){return n<=0;}))
            throw std::invalid_argument("MFQ_PREFILL_SWEEP_CHUNKS must contain positive chunk sizes");
        int64_t first_prompt_reference_top=-1;
        mfq_tensor_backend::Tensor first_prompt_reference_logits;
        mfq_tensor_backend::Tensor chunk_reference_logits;
        bool numerical_failure=false;
        for(int64_t m:sizes)for(int64_t chunk:chunks) {
            std::vector<double> elapsed_ms;
            mfq_tensor_backend::Tensor reference_logits;
            for(int repeat=0;repeat<repeats;++repeat) {
                const bool layer_major=(kernel_sweep?(exact_sweep || repeat%4>0):
                    layer_major_sweep?bool(repeat&1):layer_major_requested) && m>1024;
                if(kernel_sweep) {
                    const auto* exact_gr=std::getenv("MFQ_PREFILL_EXACT_GR_KERNEL");
                    const auto* gr=gr_mix_pair_sweep || nvq_k_pair_sweep?"17":gr_tile_pair_sweep?(repeat%2?(exact_gr?exact_gr:"12"):"6"):quant_pair_sweep?"6":exact_sweep?(repeat%2?(exact_gr?exact_gr:"8"):"1"):(repeat%4==3?"3":"1");
                    const auto* batch=quant_pair_sweep || gr_tile_pair_sweep || gr_mix_pair_sweep || nvq_k_pair_sweep?"1":exact_pair_sweep?(repeat%2?"1":"0"):(!gr_gdn_sweep && repeat%4>=2?"1":"0");
                    const auto* columns=exact_sweep || gr_gdn_sweep && repeat%4>=2?"4":"0";
                    const bool use_defaults=default_pair_sweep && repeat%2;
#ifdef _WIN32
                    _putenv_s("MFQ_GR_PREFILL_MATMUL",use_defaults?"":gr);
                    _putenv_s("MFQ_MOE_PREFILL_EXPERT_BATCH",use_defaults?"":batch);
                    _putenv_s("MFQ_GDN_PREFILL_COLUMNS",columns);
#else
                    if(use_defaults) {
                        unsetenv("MFQ_GR_PREFILL_MATMUL");unsetenv("MFQ_MOE_PREFILL_EXPERT_BATCH");
                    } else {
                        setenv("MFQ_GR_PREFILL_MATMUL",gr,1);setenv("MFQ_MOE_PREFILL_EXPERT_BATCH",batch,1);
                    }
                    setenv("MFQ_GDN_PREFILL_COLUMNS",columns,1);
#endif
                    if(quant_pair_sweep) {
                        const auto* wide_rows=repeat%2?"1":"0";
#ifdef _WIN32
                        _putenv_s("MFQ_NVQ_PREFILL_M256",wide_rows);
#else
                        setenv("MFQ_NVQ_PREFILL_M256",wide_rows,1);
#endif
                        std::cout<<"prefill_quant_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<" m256="<<wide_rows<<std::endl;
                    }
                    if(nvq_k_pair_sweep) {
                        const auto* narrow_g2=repeat%2?"1":"0";
#ifdef _WIN32
                        _putenv_s(nvq_trial_option,narrow_g2);
                        if(nvq_cohort_pair_sweep)_putenv_s("MFQ_GDN_PREFILL_PIPELINED",narrow_g2);
#else
                        setenv(nvq_trial_option,narrow_g2,1);
                        if(nvq_cohort_pair_sweep)setenv("MFQ_GDN_PREFILL_PIPELINED",narrow_g2,1);
#endif
                        std::cout<<(nvq_cohort_pair_sweep?"prefill_nvq_cohort_mode_m=":"prefill_nvq_k_mode_m=")
                            <<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<(nvq_cohort_pair_sweep?" cohorts=":" narrow_g2=")<<narrow_g2<<std::endl;
                        if(nvq_cohort_pair_sweep)std::cout<<"prefill_gdn_pipeline_mode_m="<<m<<" chunk="<<chunk
                            <<" repeat="<<repeat<<" pipelined="<<narrow_g2<<std::endl;
                    }
                    if(gr_mix_pair_sweep) {
                        const auto* mix_option=std::getenv("MFQ_PREFILL_MIX_KERNEL");
                        const auto* mix_baseline=std::getenv("MFQ_PREFILL_MIX_BASELINE");
                        const auto* mixed=repeat%2?(mix_option?mix_option:"1"):(mix_baseline?mix_baseline:"0");
#ifdef _WIN32
                        _putenv_s("MFQ_GR_PREFILL_FUSED_MIX",mixed);
#else
                        setenv("MFQ_GR_PREFILL_FUSED_MIX",mixed,1);
#endif
                        std::cout<<"prefill_mix_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<" mode="<<mixed<<std::endl;
                    }
                    std::cout<<"prefill_kernel_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" gr_kernel="<<gr<<" expert_batch="<<batch<<std::endl;
                    std::cout<<"prefill_gdn_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" columns="<<columns<<std::endl;
                }
                if(layer_major_sweep || layer_major_requested)
                    std::cout<<"prefill_layer_major_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" layer_major="<<int(layer_major)<<std::endl;
                if(phase_sweep) {
                    const auto* enabled=(repeat&1)?"1":"0";
#ifdef _WIN32
                    _putenv_s("MFQ_MOE_PREFILL_LAYER_PHASED",enabled);
#else
                    setenv("MFQ_MOE_PREFILL_LAYER_PHASED",enabled,1);
#endif
                    std::cout<<"prefill_transfer_phase_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" phased="<<enabled<<std::endl;
                }
                if(async_sweep) {
                    const auto* enabled=(repeat&1)?"1":"0";
#ifdef _WIN32
                    _putenv_s("MFQ_MOE_PREFILL_LAYER_ASYNC",enabled);
#else
                    setenv("MFQ_MOE_PREFILL_LAYER_ASYNC",enabled,1);
#endif
                    std::cout<<"prefill_transfer_async_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" async="<<enabled<<std::endl;
                }
                if(flush_sweep) {
                    const auto* enabled=(repeat&1)?"1":"0";
#ifdef _WIN32
                    _putenv_s("MFQ_MOE_PREFILL_LAYER_FLUSH",enabled);
#else
                    setenv("MFQ_MOE_PREFILL_LAYER_FLUSH",enabled,1);
#endif
                    std::cout<<"prefill_transfer_flush_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" flush="<<enabled<<std::endl;
                }
                model.reset(1);mfq_cuda_synchronize();
                execution.profiler.reset();
                const auto started=std::chrono::steady_clock::now();
                mfq_tensor_backend::Tensor logits;
                if(layer_major) {
                    if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                        execution.profiler.enabled=profile;
                        auto hidden=qwen4_layer_major_prefill_hidden(model,all_ids.narrow(1,0,m),chunk,true);
                        logits=model.apply_final_logit_softcap(model.adapter_last_logits(
                            model.lm_head,model.last_hidden(std::move(hidden))));
                        mfq_cuda_synchronize();
                        if(profile) {
                            execution.profiler.report("prefill_layer_major_m="+std::to_string(m)+
                                " chunk="+std::to_string(chunk)+" repeat="+std::to_string(repeat));
                            execution.profiler.reset();
                        }
                    } else throw std::runtime_error("layer-major sweep requires Qwen4");
                } else for(int64_t offset=0;offset<m;offset+=chunk) {
                    execution.profiler.enabled=profile && offset==0;
                    const auto count=std::min(chunk,m-offset);
                    auto part=all_ids.narrow(1,offset,count);
                    if(offset+count==m)logits=model.last_logits(part);
                    else (void)model.hidden_forward(part);
                    mfq_cuda_synchronize();
                    if(execution.profiler.enabled) {
                        execution.profiler.report("prefill_m="+std::to_string(m)+" chunk="+std::to_string(chunk)+
                            " repeat="+std::to_string(repeat));
                        execution.profiler.reset();
                    }
                    std::cout<<"prefill_progress_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                             <<" completed="<<offset+count<<" elapsed_ms="
                             <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<std::endl;
                }
                execution.profiler.enabled=false;
                const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                // Validate outside the measured region. An argmax alone can
                // silently accept a projection which has produced NaNs.
                const auto checked=logits.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kFloat32).contiguous();
                for(int64_t i=0;i<checked.numel();++i)
                    if(!std::isfinite(checked.template data_ptr<float>()[i]))
                        throw std::runtime_error("nonfinite prefill logits at index "+std::to_string(i));
                if(layer_major_sweep || chunk_sweep_oracle) {
                    const bool reference=chunk_sweep_oracle ? chunk==chunks.front() && repeat==0 : exact_sweep?repeat%4==0:!layer_major;
                    if(reference) {
                        reference_logits=checked.clone();
                        if(chunk_sweep_oracle)chunk_reference_logits=checked.clone();
                    }
                    else {
                        if(chunk_sweep_oracle)reference_logits=chunk_reference_logits;
                        if(!reference_logits.defined() || reference_logits.sizes()!=checked.sizes())
                            throw std::runtime_error("layer-major logit reference missing");
                        double error=0,scale=0,maximum=0;
                        const auto* actual=checked.template data_ptr<float>();
                        const auto* expected=reference_logits.template data_ptr<float>();
                        for(int64_t i=0;i<checked.numel();++i) {
                            const double delta=double(actual[i])-expected[i];
                            error+=delta*delta;scale+=double(expected[i])*expected[i];
                            maximum=std::max(maximum,std::abs(delta));
                        }
                        const auto rms=std::sqrt(error/std::max(1e-30,scale));
                        std::cout<<"prefill_layer_major_oracle_m="<<m<<" chunk="<<chunk
                            <<" repeat="<<repeat<<" relative_rms="<<rms
                            <<" maximum_absolute="<<maximum<<std::endl;
                        const bool accepted=rms<=1e-4;
                        if(kernel_sweep || chunk_sweep_oracle) {
                            numerical_failure=numerical_failure || !accepted;
                            std::cout<<"prefill_numerical_validation_m="<<m<<" chunk="<<chunk
                                <<" repeat="<<repeat<<" accepted="<<int(accepted)<<std::endl;
                        } else MFQ_RUNTIME_CHECK(accepted,"layer-major full-prompt logits differ from chunk-major oracle");
                        MFQ_RUNTIME_CHECK(model.cache_pos==m,"layer-major logical cache did not advance");
                    }
                }
                if(phase_sweep && (repeat&1) && execution.moe_expert_cache) {
                    std::ostringstream stats;
                    print_moe_expert_cache_stats(execution.moe_expert_cache,stats);
                    const auto data=stats.str();
                    const std::string key="pipeline_phased_transfer_serves=";
                    const auto begin=data.find(key);
                    if(begin==std::string::npos || std::stoull(data.substr(begin+key.size()))==0)
                        throw std::runtime_error("phased prefill was requested but no split copy executed");
                }
                elapsed_ms.push_back(ms);
                const auto top=logits.argmax(-1).template item<int64_t>();
                if((layer_major_sweep && (!layer_major || exact_sweep && repeat%4==0) || chunk_sweep_oracle && repeat==0) && m==sizes.front() && chunk==chunks.front()) {
                    first_prompt_reference_top=top;
                    first_prompt_reference_logits=checked.clone();
                }
                std::cout<<"prefill_chunk_sample_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                         <<" elapsed_ms="<<ms<<" tok_per_s="<<1000.0*m/ms<<" top="<<top<<std::endl;
            }
            std::sort(elapsed_ms.begin(),elapsed_ms.end());
            std::cout<<((flush_sweep || async_sweep || phase_sweep || layer_major_sweep)?"prefill_mode_comparison_m=":"prefill_chunk_summary_m=")<<m<<" chunk="<<chunk<<" repeats="<<repeats
                     <<" median_ms="<<elapsed_ms[elapsed_ms.size()/2]
                     <<" min_ms="<<elapsed_ms.front()<<" max_ms="<<elapsed_ms.back()
                     <<" tok_per_s="<<1000.0*m/elapsed_ms[elapsed_ms.size()/2]<<std::endl;
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(layer_major_sweep || chunk_sweep_oracle) {
                // Validate the exact GDN improvement through the public path
                // even if a separate approximate GR experiment was rejected.
                // Chunk comparisons must time the requested kernels even when
                // a larger chunk fails the numerical oracle.
                if(numerical_failure && !chunk_sweep_oracle) {
#ifdef _WIN32
                    _putenv_s("MFQ_GR_PREFILL_MATMUL","1");
                    _putenv_s("MFQ_MOE_PREFILL_EXPERT_BATCH","0");
#else
                    setenv("MFQ_GR_PREFILL_MATMUL","1",1);
                    setenv("MFQ_MOE_PREFILL_EXPERT_BATCH","0",1);
#endif
                }
                const auto engine_chunks=chunk_sweep_oracle?chunks:std::vector<int64_t>{chunks.front()};
                const std::string engine_gr=std::getenv("MFQ_GR_PREFILL_MATMUL")?std::getenv("MFQ_GR_PREFILL_MATMUL"):"17";
                const std::string engine_batch=std::getenv("MFQ_MOE_PREFILL_EXPERT_BATCH")?std::getenv("MFQ_MOE_PREFILL_EXPERT_BATCH"):"1";
                const std::string engine_mix=std::getenv("MFQ_PREFILL_MIX_KERNEL")?std::getenv("MFQ_PREFILL_MIX_KERNEL"):"1";
                const std::string engine_mix_baseline=std::getenv("MFQ_PREFILL_MIX_BASELINE")?std::getenv("MFQ_PREFILL_MIX_BASELINE"):"0";
                for(const auto engine_chunk:engine_chunks) {
                std::vector<int64_t> engine_reference_tokens;
                const auto* public_repeat_option=std::getenv("MFQ_PREFILL_PUBLIC_PAIR_REPEATS");
                const int public_repeats=(exact_sweep || chunk_sweep_oracle) && public_repeat_option?std::clamp(std::atoi(public_repeat_option),1,8):1;
                for(int public_repeat=0;public_repeat<public_repeats;++public_repeat) {
                for(int engine_order=0;engine_order<(exact_sweep?2:1);++engine_order) {
                const int engine_mode=exact_sweep?(engine_order^(public_repeat&1)):0;
                std::cout<<"prefill_public_pair_repeat prompt="<<sizes.front()<<" chunk="<<engine_chunk
                    <<" repeat="<<public_repeat<<" mode="<<engine_mode<<std::endl;
                // End the previous long-prompt case before the public-path
                // check; unused asynchronous pool blocks are outside timing.
                model.reset(1);mfq_cuda_synchronize();
#ifdef MFQ_NATIVE_CUDA_RUNTIME
                cudaMemPool_t verification_pool=nullptr;
                MFQ_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&verification_pool,mfq_current_cuda_device()));
                MFQ_CUDA_CHECK(cudaMemPoolTrimTo(verification_pool,0));
#endif
                if(exact_sweep) {
#ifdef _WIN32
                    _putenv_s("MFQ_GR_PREFILL_MATMUL",gr_mix_pair_sweep || nvq_k_pair_sweep?"17":gr_tile_pair_sweep?(engine_mode?engine_gr.c_str():"6"):quant_pair_sweep?"6":engine_mode?engine_gr.c_str():"1");
                    _putenv_s("MFQ_MOE_PREFILL_EXPERT_BATCH",quant_pair_sweep || gr_tile_pair_sweep || gr_mix_pair_sweep || nvq_k_pair_sweep?"1":engine_mode?engine_batch.c_str():"0");
                    if(nvq_k_pair_sweep)_putenv_s(nvq_trial_option,engine_mode?"1":"0");
                    if(nvq_cohort_pair_sweep)_putenv_s("MFQ_GDN_PREFILL_PIPELINED",engine_mode?"1":"0");
                    if(quant_pair_sweep)_putenv_s("MFQ_NVQ_PREFILL_M256",engine_mode?"1":"0");
                    if(gr_mix_pair_sweep)_putenv_s("MFQ_GR_PREFILL_FUSED_MIX",engine_mode?engine_mix.c_str():engine_mix_baseline.c_str());
#else
                    setenv("MFQ_GR_PREFILL_MATMUL",gr_mix_pair_sweep || nvq_k_pair_sweep?"17":gr_tile_pair_sweep?(engine_mode?engine_gr.c_str():"6"):quant_pair_sweep?"6":engine_mode?engine_gr.c_str():"1",1);
                    setenv("MFQ_MOE_PREFILL_EXPERT_BATCH",quant_pair_sweep || gr_tile_pair_sweep || gr_mix_pair_sweep || nvq_k_pair_sweep?"1":engine_mode?engine_batch.c_str():"0",1);
                    if(nvq_k_pair_sweep)setenv(nvq_trial_option,engine_mode?"1":"0",1);
                    if(nvq_cohort_pair_sweep)setenv("MFQ_GDN_PREFILL_PIPELINED",engine_mode?"1":"0",1);
                    if(quant_pair_sweep)setenv("MFQ_NVQ_PREFILL_M256",engine_mode?"1":"0",1);
                    if(gr_mix_pair_sweep)setenv("MFQ_GR_PREFILL_FUSED_MIX",engine_mode?engine_mix.c_str():engine_mix_baseline.c_str(),1);
#endif
                }
                auto config=resolve_cuda_runtime_config({});
                config.generation.prefill_chunk_size=engine_chunk;
                DecodeGraphCache graph(model.max_position_embeddings());
                TextSessionCache cache(config.session_cache,config.prefix_cache);
                mfq::engine::InferenceRequest request;
                request.prompt.assign(token_ids.begin(),token_ids.begin()+sizes.front());
                request.sampling.max_tokens=exact_sweep?3:1;request.sampling.temperature=0.;
                request.sampling.top_k=1;request.sampling.top_p=1.;request.sampling.enable_mtp=false;
                mfq::engine::InferenceOutput output(request,nullptr,"layer-major-check");
                const auto engine_started=std::chrono::steady_clock::now();
                auto result=collect_generation(generate(model,graph,cache,config,request,output));
                mfq_cuda_synchronize();
                const auto engine_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-engine_started).count();
                std::cout<<"prefill_ordinary_generation_timing prompt="<<sizes.front()<<" chunk="<<engine_chunk
                    <<" mode="<<engine_mode<<" generated_tokens="<<result.tokens.size()
                    <<" elapsed_ms="<<engine_ms<<" tok_per_s="<<1000.0*sizes.front()/engine_ms<<std::endl;
                std::cout<<"prefill_ordinary_prefill_timing prompt="<<sizes.front()<<" chunk="<<engine_chunk
                    <<" mode="<<engine_mode<<" elapsed_ms="<<result.prefill.model_ms
                    <<" tok_per_s="<<1000.0*sizes.front()/result.prefill.model_ms<<std::endl;
                MFQ_RUNTIME_CHECK(!result.tokens.empty() && result.tokens.front()==first_prompt_reference_top,
                    "layer-major ordinary generation disagrees with chunk-major oracle");
                if(exact_sweep) {
                    if(engine_mode==0)engine_reference_tokens=result.tokens;
                    else MFQ_RUNTIME_CHECK(result.tokens==engine_reference_tokens,
                        "optimized prefill/decode continuation differs from original generation");
                    std::cout<<"prefill_decode_continuation_check prompt="<<sizes.front()
                        <<" mode="<<engine_mode<<" generated_tokens="<<result.tokens.size()<<" PASS"<<std::endl;
                } else MFQ_RUNTIME_CHECK(result.tokens.size()==1,"generation first-token check returned extra tokens");
                std::cout<<"prefill_layer_major_generation_check prompt="<<sizes.front()
                    <<" chunk="<<engine_chunk<<" top="<<first_prompt_reference_top<<" PASS"<<std::endl;
                }
                }
                }
                if(exact_sweep) {
#ifdef _WIN32
                    _putenv_s("MFQ_GR_PREFILL_MATMUL",engine_gr.c_str());
                    _putenv_s("MFQ_MOE_PREFILL_EXPERT_BATCH",engine_batch.c_str());
                    if(gr_mix_pair_sweep)_putenv_s("MFQ_GR_PREFILL_FUSED_MIX",engine_mix.c_str());
                    if(quant_pair_sweep)_putenv_s("MFQ_NVQ_PREFILL_M256","1");
                    if(nvq_k_pair_sweep)_putenv_s(nvq_trial_option,"1");
                    if(nvq_cohort_pair_sweep)_putenv_s("MFQ_GDN_PREFILL_PIPELINED","1");
#else
                    setenv("MFQ_GR_PREFILL_MATMUL",engine_gr.c_str(),1);
                    setenv("MFQ_MOE_PREFILL_EXPERT_BATCH",engine_batch.c_str(),1);
                    if(gr_mix_pair_sweep)setenv("MFQ_GR_PREFILL_FUSED_MIX",engine_mix.c_str(),1);
                    if(quant_pair_sweep)setenv("MFQ_NVQ_PREFILL_M256","1",1);
                    if(nvq_k_pair_sweep)setenv(nvq_trial_option,"1",1);
                    if(nvq_cohort_pair_sweep)setenv("MFQ_GDN_PREFILL_PIPELINED","1",1);
#endif
                }
                const auto* tail_profile=std::getenv("MFQ_PREFILL_POST_TIMING_PROFILE");
                if(tail_profile && std::atoi(tail_profile)!=0) {
                    model.reset(1);execution.profiler.reset();execution.profiler.enabled=true;
                    auto hidden=qwen4_layer_major_prefill_hidden(model,all_ids.narrow(1,0,sizes.front()),chunks.front(),true);
                    (void)model.apply_final_logit_softcap(model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden))));
                    mfq_cuda_synchronize();
                    execution.profiler.report("prefill_optimized_tail_profile_m="+std::to_string(sizes.front()));
                    execution.profiler.enabled=false;
                    execution.profiler.reset();
                }
                const auto* audit=std::getenv("MFQ_PREFILL_EXPERT_BATCH_AUDIT_RUN");
                if(audit && std::atoi(audit)!=0) {
#ifdef _WIN32
                    _putenv_s("MFQ_GR_PREFILL_MATMUL","1");
                    _putenv_s("MFQ_GDN_PREFILL_COLUMNS","0");
                    _putenv_s("MFQ_MOE_PREFILL_EXPERT_BATCH","1");
                    _putenv_s("MFQ_PREFILL_EXPERT_BATCH_AUDIT","1");
#else
                    setenv("MFQ_GR_PREFILL_MATMUL","1",1);
                    setenv("MFQ_GDN_PREFILL_COLUMNS","0",1);
                    setenv("MFQ_MOE_PREFILL_EXPERT_BATCH","1",1);
                    setenv("MFQ_PREFILL_EXPERT_BATCH_AUDIT","1",1);
#endif
                    model.reset(1);
                    auto hidden=qwen4_layer_major_prefill_hidden(model,all_ids.narrow(1,0,sizes.front()),chunks.front(),true);
                    auto result=model.apply_final_logit_softcap(model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden))))
                        .to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kFloat32).contiguous();
                    double error=0,scale=0;
                    for(int64_t i=0;i<result.numel();++i) {
                        const double expected=first_prompt_reference_logits.template data_ptr<float>()[i];
                        const double delta=double(result.template data_ptr<float>()[i])-expected;
                        error+=delta*delta;scale+=expected*expected;
                    }
                    const auto rms=std::sqrt(error/std::max(1e-30,scale));
                    std::cout<<"prefill_expert_batch_audit_reference_logits relative_rms="<<rms<<std::endl;
                    MFQ_RUNTIME_CHECK(std::isfinite(rms) && rms<=1e-4,"FFN batching audit changed non-FFN math");
                }
            }
        }
        return numerical_failure?2:0;
    }

    for (int64_t m : sizes) {
        auto ids = all_ids.narrow(1, 0, m);
        for (int warmup = 0; warmup < 2; ++warmup) {
            model.reset(1);
            (void)model.last_logits(ids);
            mfq_cuda_synchronize();
        }

        std::vector<double> elapsed_ms;
        elapsed_ms.reserve((size_t)repeats);
        int64_t top = -1;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            model.reset(1);
            auto started = std::chrono::steady_clock::now();
            auto logits = model.last_logits(ids);
            mfq_cuda_synchronize();
            auto ended = std::chrono::steady_clock::now();
            elapsed_ms.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
            if (repeat == 0) top = logits.argmax(-1).template item<int64_t>();
        }
        std::sort(elapsed_ms.begin(), elapsed_ms.end());
        const double median_ms = elapsed_ms[elapsed_ms.size() / 2];
        std::cout << "prefill_sweep_m=" << m
                  << " median_ms=" << median_ms
                  << " min_ms=" << elapsed_ms.front()
                  << " max_ms=" << elapsed_ms.back()
                  << " tok_per_s=" << (1000.0 * (double)m / median_ms)
                  << " top=" << top << "\n";
    }
    return 0;
}

template <typename Model>
static int run_block_trace_compare(
    Model& test,
    const std::string & reference_model_path,
    const std::string & config_path,
    int64_t context_size,
    mfq_tensor_backend::Tensor ids)
{
    auto& execution = *test.execution;
    Model reference = [&]() {
        if (execution.n_gpu_layers < 0) {
            return mfq::cuda::load_causal_lm<Model>(
                execution, reference_model_path, config_path, context_size);
        }
        const int saved_n_gpu_layers = execution.n_gpu_layers;
        const int saved_cpu_layers = execution.dense_cpu_layer_count;
        execution.n_gpu_layers = -1;
        try {
            auto loaded = mfq::cuda::load_causal_lm<Model>(
                execution, reference_model_path, config_path, context_size);
            execution.n_gpu_layers = saved_n_gpu_layers;
            execution.dense_cpu_layer_count = saved_cpu_layers;
            return loaded;
        } catch (...) {
            execution.n_gpu_layers = saved_n_gpu_layers;
            execution.dense_cpu_layer_count = saved_cpu_layers;
            throw;
        }
    }();
    std::vector<mfq_tensor_backend::Tensor> test_trace;
    std::vector<mfq_tensor_backend::Tensor> reference_trace;

    test.reset(ids.size(0));
    auto test_hidden = test.hidden_forward(ids, mfq_nullopt, mfq_nullopt, &test_trace);
    reference.reset(ids.size(0));
    auto reference_hidden = reference.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    mfq_cuda_synchronize();

    if (test_trace.size() != reference_trace.size()) {
        throw std::runtime_error("block trace stage count mismatch");
    }
    for (size_t i = 0; i < test_trace.size(); ++i) {
        auto ref = reference_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        auto got = test_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
        if (ref.numel() != got.numel()) {
            throw std::runtime_error("block trace tensor size mismatch");
        }
        auto ref_norm = ref.norm();
        auto got_norm = got.norm();
        const double ref_norm_value = ref_norm.template item<double>();
        const double denominator = std::max(ref_norm_value, 1.0e-30);
        const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
        const double cosine = mfq_tensor_backend::dot(ref, got).template item<double>() /
            std::max(ref_norm_value * got_norm.template item<double>(), 1.0e-30);
        const double norm_ratio = got_norm.template item<double>() / denominator;
        const double reference_rms = ref.square().mean().sqrt().template item<double>();
        const double test_rms = got.square().mean().sqrt().template item<double>();
        const std::string stage = i == 0
            ? "embedding"
            : "block_" + std::to_string(i - 1);
        std::cout << "block_trace stage=" << stage
                  << " relative_l2=" << relative_l2
                  << " cosine=" << cosine
                  << " norm_ratio=" << norm_ratio
                  << " reference_rms=" << reference_rms
                  << " test_rms=" << test_rms << "\n";
    }

    auto reference_logits = reference.lm_head.forward(
        execution, reference_hidden).to(mfq_tensor_backend::kFloat32);
    auto test_logits = test.lm_head.forward(
        execution, test_hidden).to(mfq_tensor_backend::kFloat32);
    double kl_sum = 0.0;
    int64_t same_top = 0;
    int64_t rows = 0;
    const int64_t tokens = reference_logits.numel() / reference_logits.size(-1);
    auto ref2 = reference_logits.reshape({tokens, -1});
    auto got2 = test_logits.reshape({tokens, -1});
    for (int64_t start = 0; start < tokens; start += 8) {
        const int64_t end = std::min(start + 8, tokens);
        auto ref_chunk = ref2.index({Slice(start, end)});
        auto got_chunk = got2.index({Slice(start, end)});
        auto ref_logp = mfq_tensor_backend::log_softmax(ref_chunk, -1);
        auto got_logp = mfq_tensor_backend::log_softmax(got_chunk, -1);
        kl_sum += (ref_logp.exp() * (ref_logp - got_logp)).sum(-1)
            .to(mfq_tensor_backend::kFloat64).sum().template item<double>();
        same_top += ref_chunk.argmax(-1).eq(got_chunk.argmax(-1)).sum().template item<int64_t>();
        rows += end - start;
    }
    const double logits_relative_l2 =
        (test_logits.to(mfq_tensor_backend::kFloat64) - reference_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
        std::max(reference_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
    std::cout << "block_trace_logits kld=" << (kl_sum / std::max<int64_t>(rows, 1))
              << " same_top=" << ((double)same_top / std::max<int64_t>(rows, 1))
              << " relative_l2=" << logits_relative_l2 << "\n";
    return 0;
}

template <typename Model>
static int run_block_trace_dump(
    Model& model,
    const std::string & output_dir,
    mfq_tensor_backend::Tensor ids,
    int64_t token_start,
    int64_t token_count)
{
    const int64_t total_tokens = ids.size(1);
    if (token_start < 0 || token_start >= total_tokens) {
        throw std::runtime_error("--dump-block-trace-start is outside the token range");
    }
    if (token_count <= 0) token_count = total_tokens - token_start;
    if (token_count > total_tokens - token_start) {
        throw std::runtime_error("--dump-block-trace-count exceeds the token range");
    }
    const std::filesystem::path root(output_dir);
    std::error_code error;
    if (!std::filesystem::create_directories(root, error) || error) {
        throw std::runtime_error(
            "block trace output directory must be new: " + output_dir);
    }

    model.reset(ids.size(0));
    std::vector<mfq_tensor_backend::Tensor> trace;
    auto final_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &trace);
    mfq_cuda_synchronize();

    auto ids_cpu = ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous();
    {
        std::ofstream output(root / "tokens.i32", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create block trace token file");
        output.write(
            reinterpret_cast<const char *>(ids_cpu.template data_ptr<int32_t>()),
            static_cast<std::streamsize>(ids_cpu.nbytes()));
        if (!output) throw std::runtime_error("failed to write block trace tokens");
    }

    std::ofstream metadata(root / "trace_meta.jsonl");
    if (!metadata) throw std::runtime_error("cannot create block trace metadata");
    auto token_slice = [&](mfq_tensor_backend::Tensor value) {
        if (value.dim() >= 3 && value.size(1) == total_tokens) {
            return value.narrow(1, token_start, token_count);
        }
        return value;
    };
    for (size_t index = 0; index < trace.size(); ++index) {
        auto value = token_slice(trace[index])
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    }
    auto dump_terminal = [&](const std::string & stage, mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
        const std::filesystem::path file = root / (stage + ".f32");
        std::ofstream output(file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot create block trace tensor: " + file.string());
        }
        output.write(
            reinterpret_cast<const char *>(value.template data_ptr<float>()),
            static_cast<std::streamsize>(value.nbytes()));
        if (!output) {
            throw std::runtime_error("failed to write block trace tensor: " + file.string());
        }
        metadata << "{\"stage\":\"" << stage << "\",\"file\":\""
                 << file.filename().string() << "\",\"shape\":[";
        for (int64_t dim = 0; dim < value.dim(); ++dim) {
            if (dim) metadata << ',';
            metadata << value.size(dim);
        }
        metadata << "],\"dtype\":\"float32\"}\n";
        metadata.flush();
        std::cout << "block_trace_dump stage=" << stage
                  << " values=" << value.numel() << "\n";
    };
    final_hidden = token_slice(final_hidden);
    dump_terminal("final_norm", final_hidden);
    auto logits = model.apply_final_logit_softcap(
        model.lm_head.forward(*model.execution, final_hidden));
    dump_terminal("logits", logits);
    metadata.flush();
    if (!metadata) throw std::runtime_error("failed to write block trace metadata");
    return 0;
}

template <typename Model>
static int run_dsv4_hc_model_compare(
    Model& model,
    mfq_tensor_backend::Tensor ids)
{
    std::vector<mfq_tensor_backend::Tensor> reference_trace;
    std::vector<mfq_tensor_backend::Tensor> candidate_trace;
    std::vector<mfq_tensor_backend::Tensor> repeat_trace;

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto reference_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &reference_trace);
    auto reference_logits = model.lm_head.forward(
        *model.execution, reference_hidden)
        .to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = true;
    model.reset(ids.size(0));
    auto candidate_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &candidate_trace);
    auto candidate_logits = model.lm_head.forward(
        *model.execution, candidate_hidden)
        .to(mfq_tensor_backend::kFloat32);

    g_dsv4_fused_hc = false;
    model.reset(ids.size(0));
    auto repeat_hidden = model.hidden_forward(
        ids, mfq_nullopt, mfq_nullopt, &repeat_trace);
    auto repeat_logits = model.lm_head.forward(
        *model.execution, repeat_hidden)
        .to(mfq_tensor_backend::kFloat32);
    g_dsv4_fused_hc = true;
    mfq_cuda_synchronize();

    if (reference_trace.size() != candidate_trace.size() ||
            reference_trace.size() != repeat_trace.size()) {
        throw std::runtime_error(
            "DeepSeek V4 HC trace stage count mismatch");
    }
    for (size_t index = 0; index < reference_trace.size(); ++index) {
        auto reference = reference_trace[index].reshape({-1});
        auto candidate = candidate_trace[index].reshape({-1});
        auto repeat = repeat_trace[index].reshape({-1});
        auto reference_f64 = reference.to(mfq_tensor_backend::kFloat64);
        auto candidate_f64 = candidate.to(mfq_tensor_backend::kFloat64);
        const double denominator = std::max(
            reference_f64.norm().template item<double>(), 1.0e-30);
        const std::string stage = index == 0
            ? "embedding"
            : "block_" + std::to_string(index - 1);
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_model_trace stage=" << stage
                  << " differing="
                  << candidate.ne(reference).sum().template item<int64_t>()
                  << " rel_l2="
                  << (candidate_f64 - reference_f64)
                         .norm().template item<double>() / denominator
                  << " mean_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().mean().template item<double>()
                  << " max_abs="
                  << (candidate_f64 - reference_f64)
                         .abs().max().template item<double>()
                  << " repeat_differing="
                  << repeat.ne(reference).sum().template item<int64_t>()
                  << "\n";
    }

    auto reference_logp = mfq_tensor_backend::log_softmax(reference_logits, -1);
    auto candidate_logp = mfq_tensor_backend::log_softmax(candidate_logits, -1);
    const double kld_candidate_reference = (
        candidate_logp.exp() *
        (candidate_logp - reference_logp))
        .sum(-1).mean().template item<double>();
    const double kld_reference_candidate = (
        reference_logp.exp() *
        (reference_logp - candidate_logp))
        .sum(-1).mean().template item<double>();
    auto logit_diff = (
        candidate_logits.to(mfq_tensor_backend::kFloat64) -
        reference_logits.to(mfq_tensor_backend::kFloat64));
    std::cout << std::scientific << std::setprecision(9)
              << "dsv4_hc_model_logits"
              << " mean_kld_candidate_reference="
              << kld_candidate_reference
              << " mean_kld_reference_candidate="
              << kld_reference_candidate
              << " relative_l2="
              << logit_diff.norm().template item<double>() /
                    std::max(
                        reference_logits.to(mfq_tensor_backend::kFloat64)
                            .norm().template item<double>(),
                        1.0e-30)
              << " mean_abs="
              << logit_diff.abs().mean().template item<double>()
              << " max_abs="
              << logit_diff.abs().max().template item<double>()
              << " same_top="
              << candidate_logits.argmax(-1)
                     .eq(reference_logits.argmax(-1))
                     .to(mfq_tensor_backend::kFloat32).mean().template item<double>()
              << " repeat_logits_equal="
              << (repeat_logits.equal(reference_logits) ? 1 : 0)
              << "\n";
    return 0;
}

// Real-weight correctness gate; does not require a tokenizer or start a runtime transport.
// It calls the same MTP generator used by the runtime, with synthetic token IDs.
static int run_qwen35_mtp_check(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp) {
    using Tensor = mfq_tensor_backend::Tensor;
    auto& execution = *model.execution;
    const MtpTarget target(model);
    // Identity projection isolates both dense gate modes and their dtype casts.
    const auto float_options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kFloat32);
    std::vector<float> identity(33 * 33, 0.f), input(6 * 33), gate(6 * 33);
    for (int i = 0; i < 33; ++i) identity[i * 33 + i] = 1.f;
    for (int i = 0; i < 6 * 33; ++i) {
        input[i] = static_cast<float>(i % 33 - 16) / 16.f;
        gate[i] = static_cast<float>(i % 31 - 15) / 4.f;
    }
    auto test_input = mfq_tensor_backend::tensor(input, float_options).reshape({1, 6, 33});
    auto test_gate = mfq_tensor_backend::tensor(gate, float_options).reshape({1, 6, 33});
    for (auto dtype : {mfq_tensor_backend::kFloat32, mfq_tensor_backend::kFloat16,
                       mfq_tensor_backend::kBFloat16}) {
        QuantLinear linear;
        linear.kind = QuantLinearKind::Dense;
        linear.dense = mfq_tensor_backend::tensor(identity, float_options)
            .reshape({33, 33}).to(dtype);
        const double tolerance = dtype == mfq_tensor_backend::kBFloat16 ? .008
            : dtype == mfq_tensor_backend::kFloat16 ? .001 : 2.e-6;
        for (int m = 1; m <= 6; ++m) for (int mode : {1, 2}) {
            auto actual = linear.forward_input_mul(
                execution, test_input.narrow(1, 0, m),
                test_gate.narrow(1, 0, m), mode);
            MFQ_RUNTIME_CHECK(actual.scalar_type() == dtype && actual.size(1) == m,
                "dense gate output dtype or shape mismatch");
            auto values = actual.to(mfq_tensor_backend::kFloat32).cpu();
            for (int i = 0; i < m * 33; ++i) {
                const double activated = (mode == 1 ? 1. : gate[i]) / (1. + std::exp(-double(gate[i])));
                const double reference = double(input[i]) * activated;
                const double value = values.template data_ptr<float>()[i];
                MFQ_RUNTIME_CHECK(std::isfinite(value) &&
                    std::abs(value - reference) <= tolerance * (1. + std::abs(reference)),
                    "dense gate differs from CPU double identity oracle");
            }
        }
    }
    std::cout << "mtp_check dense_gate_cases=36 PASS\n";
    // Nonidentity weights expose M-dependent cuBLAS accumulation; identity cannot.
    std::vector<float> projection_input(6 * 257), projection_weight(131 * 257);
    for (size_t i = 0; i < projection_input.size(); ++i)
        projection_input[i] = static_cast<float>(std::sin(double(i) * .37));
    for (size_t i = 0; i < projection_weight.size(); ++i)
        projection_weight[i] = static_cast<float>(std::cos(double(i) * .19) / 17.);
    auto projection = mfq_tensor_backend::tensor(projection_input, float_options)
        .reshape({1, 6, 257});
    for (auto dtype : {mfq_tensor_backend::kFloat32, mfq_tensor_backend::kFloat16,
                       mfq_tensor_backend::kBFloat16}) {
        QuantLinear linear;
        linear.kind = QuantLinearKind::Dense;
        linear.dense = mfq_tensor_backend::tensor(projection_weight, float_options)
            .reshape({131, 257}).to(dtype);
        std::vector<Tensor> serial;
        for (int row = 0; row < 6; ++row)
            serial.push_back(linear.forward(execution, projection.narrow(1, row, 1)));
        auto reference = mfq_tensor_backend::cat(serial, 1);
        for (int rows = 2; rows <= 6; ++rows) {
            auto actual = linear.forward(execution, projection.narrow(1, 0, rows));
            MFQ_RUNTIME_CHECK(actual.equal(reference.narrow(1, 0, rows)),
                "short dense projection differs from per-token decode");
        }
    }
    std::cout << "mtp_check dense_decode_geometry_cases=15 PASS\n";
    const auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
        .dtype(mfq_tensor_backend::kInt64);
    auto ids = [&](const std::vector<int64_t>& tokens) {
        return mfq_tensor_backend::tensor(tokens, options).reshape({1, -1});
    };
    bool numerical_mismatch = false;
    auto compare = [&](const Tensor& actual, const Tensor& reference, double tolerance,
                       const char* name) {
        MFQ_RUNTIME_CHECK(actual.sizes() == reference.sizes(), "MTP gate tensor shape mismatch");
        auto a = actual.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        auto r = reference.contiguous().to(mfq_tensor_backend::kFloat32).cpu();
        double squared = 0., norm = 0.;
        for (int64_t i = 0; i < a.numel(); ++i) {
            const double av = a.template data_ptr<float>()[i], rv = r.template data_ptr<float>()[i];
            MFQ_RUNTIME_CHECK(std::isfinite(av) && std::isfinite(rv), "nonfinite MTP gate value");
            squared += (av - rv) * (av - rv);
            norm += rv * rv;
        }
        const double rel = std::sqrt(squared / std::max(norm, 1.e-30));
        std::cout << "mtp_check " << name << " relative_l2=" << rel << '\n';
        numerical_mismatch = numerical_mismatch || rel > tolerance;
    };
    const std::vector<int64_t> prompt{100, 200, 300, 400, 500, 600, 700};
    // Explicitly test zero, partial, and full acceptance regardless of the
    // real predictor's acceptance rate.
    for (int accepted_drafts : {0, 1, 2}) {
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> verify_trace;
        std::vector<std::pair<std::string, Tensor>> verify_stages;
        execution.gemma_trace_layer = 3;
        execution.gemma_stage_trace = &verify_stages;
        (void)model.hidden_forward(ids({37, 41, 43}), mfq_nullopt, mfq_nullopt,
            &verify_trace, mfq_nullopt, nullptr, 1);
        execution.gemma_stage_trace = nullptr;
        if (accepted_drafts == 2) model.commit_speculative();
        else model.rollback_speculative(accepted_drafts);
        MFQ_RUNTIME_CHECK(
            model.cache_pos == static_cast<int64_t>(prompt.size()) +
                1 + accepted_drafts,
            "MTP resolution retained an incorrect logical cache length");
        auto actual = model.last_logits(ids({47})).clone();
        std::vector<std::pair<Tensor, Tensor>> states;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get()))
                states.emplace_back(linear->conv_state.clone(), linear->gdn_state.clone());
        }
        model.reset(1);
        (void)model.hidden_forward(ids(prompt));
        std::vector<Tensor> serial_trace;
        std::vector<std::pair<std::string, Tensor>> serial_stages;
        execution.gemma_stage_trace = &serial_stages;
        (void)model.hidden_forward(ids({37}), mfq_nullopt, mfq_nullopt,
            &serial_trace);
        execution.gemma_stage_trace = nullptr;
        MFQ_RUNTIME_CHECK(verify_stages.size() == serial_stages.size() && !verify_stages.empty(),
            "MTP full-attention stage trace mismatch");
        for (size_t stage = 0; stage < verify_stages.size(); ++stage) {
            MFQ_RUNTIME_CHECK(verify_stages[stage].first == serial_stages[stage].first,
                "MTP full-attention stage names differ");
            compare(verify_stages[stage].second.narrow(1, 0, 1), serial_stages[stage].second,
                .005, verify_stages[stage].first.c_str());
        }
        MFQ_RUNTIME_CHECK(verify_trace.size() == serial_trace.size(),
            "MTP diagnostic block trace size mismatch");
        for (size_t layer = 0; layer < verify_trace.size(); ++layer) {
            const auto label = "confirmed_prefix_block_" + std::to_string(layer);
            compare(verify_trace[layer].narrow(1, 0, 1), serial_trace[layer], .005,
                label.c_str());
        }
        if (accepted_drafts >= 1) (void)model.hidden_forward(ids({41}));
        if (accepted_drafts >= 2) (void)model.hidden_forward(ids({43}));
        auto reference = model.last_logits(ids({47}));
        const auto resolution_label =
            "resolution_" + std::to_string(accepted_drafts) + "_logits";
        compare(actual, reference, .005, resolution_label.c_str());
        size_t state = 0;
        for (auto& block : model.blocks) {
            if (auto* linear = dynamic_cast<
                    mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
                const auto conv_label = "conv_continuation_" + std::to_string(state);
                const auto gdn_label = "gdn_continuation_" + std::to_string(state);
                compare(states[state].first, linear->conv_state, .002, conv_label.c_str());
                compare(states[state].second, linear->gdn_state, .002, gdn_label.c_str());
                ++state;
            }
        }
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP gate numerical mismatch (see per-layer diagnostics)");
    model.reset(1);
    Tensor raw;
    (void)model.hidden_forward(ids(prompt), mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt, &raw);
    for (int tokens = 2; tokens <= 6; ++tokens) {
        auto next = ids(prompt).narrow(1, 1, tokens);
        std::vector<std::pair<std::string, Tensor>> batch_stages, row_stages;
        execution.gemma_trace_layer = 0;
        if (tokens == 2) execution.gemma_stage_trace = &batch_stages;
        mtp.reset();
        auto batched = mtp.forward(
            target, raw.narrow(1, 0, tokens), next).clone();
        mtp.reset();
        if (tokens == 2) execution.gemma_stage_trace = &row_stages;
        std::vector<Tensor> serial;
        for (int t = 0; t < tokens; ++t)
            serial.push_back(mtp.forward(
                target, raw.narrow(1, t, 1), next.narrow(1, t, 1)));
        execution.gemma_stage_trace = nullptr;
        if (tokens == 2) {
            MFQ_RUNTIME_CHECK(row_stages.size() == 2 * batch_stages.size(),
                "MTP predictor diagnostic stage count mismatch");
            for (size_t stage = 0; stage < batch_stages.size(); ++stage) {
                auto reference = mfq_tensor_backend::cat({row_stages[stage].second,
                    row_stages[stage + batch_stages.size()].second}, 1);
                const auto label = "predictor_stage_" + batch_stages[stage].first;
                compare(batch_stages[stage].second, reference,
                    std::numeric_limits<double>::infinity(), label.c_str());
            }
        }
        compare(batched, mfq_tensor_backend::cat(serial, 1), .005, "predictor_batched_vs_serial");
    }
    MFQ_RUNTIME_CHECK(!numerical_mismatch, "MTP predictor gate numerical mismatch");
    uint64_t total_cycles = 0;
    for (const auto& input : std::vector<std::vector<int64_t>>{{1, 2, 3}, prompt, std::vector<int64_t>(17, 10)}) {
        MfqSamplingParams params;
        params.max_tokens = 32;
        params.temperature = 0.;
        params.top_k = 1;
        params.seed = 20260907;
        model.reset(1);
        auto current = ids(input);
        std::vector<int64_t> expected;
        for (int step = 0; step < params.max_tokens; ++step) {
            auto next = model.next_token(current);
            expected.push_back(next.template item<int64_t>());
            current = next.reshape({1, 1});
        }
        std::vector<int64_t> got;
        got = check_mtp_steps(model, mtp, input, params);
        const int produced = static_cast<int>(got.size());
        total_cycles += mtp.last_cycles;
        std::cout << "mtp_check greedy_prompt_tokens=" << input.size() << " produced=" << produced
            << " exact=" << (got == expected) << " accepted=" << mtp.last_accepted
            << " rejected=" << mtp.last_rejected << '\n';
        MFQ_RUNTIME_CHECK(produced == params.max_tokens && got == expected,
            "MTP greedy generation differs from ordinary incremental decode");
        // Output limit then a fresh request exercises state reset after an
        // early return, including stopping before a computed bonus is emitted.
        got.clear();
        got = check_mtp_steps(model, mtp, input, params, 3);
        const int stopped = static_cast<int>(got.size());
        MFQ_RUNTIME_CHECK(stopped == 3 && got == std::vector<int64_t>(expected.begin(), expected.begin() + 3),
            "MTP step emitted extra or incorrect tokens");
    }
    MfqSamplingParams stochastic;
    stochastic.max_tokens = 8;
    stochastic.temperature = .8;
    stochastic.top_k = 100;
    stochastic.top_p = .95;
    stochastic.presence_penalty = .2;
    stochastic.frequency_penalty = .1;
    stochastic.repetition_penalty = 1.05;
    stochastic.seed = 20260907;
    const int produced = static_cast<int>(check_mtp_steps(model, mtp, prompt, stochastic).size());
    MFQ_RUNTIME_CHECK(produced == 8 && mtp.last_cycles > 0 && total_cycles > 0,
        "MTP runtime gate did not execute speculative cycles");
    int64_t linear_layers = 0, ffn_batches = 0, projection_batches = 0;
    for (const auto& block : model.blocks) {
        if (auto* linear = dynamic_cast<const
                mfq::cuda::qwen35::LinearAttentionBlock*>(block.get())) {
            MFQ_RUNTIME_CHECK(
                linear->speculative_ffn_batches > 0 &&
                    linear->speculative_projection_batches > 0,
                "MTP gate did not exercise full-window recurrent batching");
            ++linear_layers;
            ffn_batches += linear->speculative_ffn_batches;
            projection_batches += linear->speculative_projection_batches;
        }
    }
    MFQ_RUNTIME_CHECK(
        linear_layers > 0, "MTP batching gate found no recurrent layers");
    std::cout << "mtp_check batched_linear_layers=" << linear_layers
        << " ffn_calls=" << ffn_batches
        << " projection_calls=" << projection_batches << '\n';
    std::cout << "mtp_check PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8\n";
    return 0;
}

template <typename Model>
static int run_flash_next_check(Model& model) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(
        model.metadata.flash_next && model.vocab_size() >= 8 &&
            model.max_position_embeddings() >= 16,
        "Flash-Next diagnostic requires a Flash-Next text graph, vocab>=8 and context>=16");
    auto ids=tb::tensor(std::vector<int64_t>{1,2,3,4,5,6,7},
        tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64)).reshape({1,7});
    const auto json_tensor=[](const tb::Tensor& value) {
        auto host=value.to(tb::kFloat32).contiguous().cpu();
        return nlohmann::json{{"shape",host.sizes().vec()},
            {"data",std::vector<float>(host.template data_ptr<float>(),host.template data_ptr<float>()+host.numel())}};
    };
    nlohmann::json result;
    model.reset(1);
    result["full"]=json_tensor(model.forward(ids));
    model.reset(1);
    std::vector<tb::Tensor> pieces;
    for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
        pieces.push_back(model.forward(ids.narrow(1,begin,count)));
    result["chunked"]=json_tensor(tb::cat(pieces,1));
    for (bool accept : {false,true}) {
        const std::string name=accept ? "committed" : "rejected";
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        auto hidden=model.hidden_forward(ids.narrow(1,2,2),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,nullptr,1);
        result[name+"_verify"]=json_tensor(model.logits_from_hidden(hidden));
        if (accept) model.commit_speculative(); else model.rollback_speculative();
        MFQ_RUNTIME_CHECK(model.cache_pos==(accept?4:3),"Flash-Next transaction cache position mismatch");
        result[name]=json_tensor(model.forward(ids.narrow(1,5,1)));
        model.reset(1);
        model.forward(ids.narrow(1,0,2));
        model.forward(ids.narrow(1,2,accept?2:1));
        result[name+"_reference"]=json_tensor(model.forward(ids.narrow(1,5,1)));
    }
    model.reset(1);
    result["reset"]=json_tensor(model.forward(ids));
    if (model.metadata.multi_axis_positions) {
        auto positions=tb::stack({ids.reshape({7})-1,ids.reshape({7})+1,ids.reshape({7})+3},0);
        model.reset(1);
        result["axis_full"]=json_tensor(model.logits_from_hidden(model.hidden_forward(ids,positions)));
        model.reset(1);pieces.clear();
        auto positions4=tb::cat({positions.narrow(0,0,1)+11,positions},0);
        for (auto [begin,count] : std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
            pieces.push_back(model.logits_from_hidden(model.hidden_forward(ids.narrow(1,begin,count),positions4.narrow(-1,begin,count))));
        result["axis_chunked"]=json_tensor(tb::cat(pieces,1));
        model.reset(1);
        result["batch"]=json_tensor(model.forward(ids.repeat({2,1})));
        result["batch_reset"]=json_tensor(model.forward(ids));
        model.reset(1);result["last"]=json_tensor(model.last_logits(ids));
    }
    result["architecture"] = model.graph.backbone;
    std::cout << "flash_next_check " << result.dump() << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
