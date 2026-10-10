#include "decode_window.h"
#include "execution_options.h"
#include "mfq_cuda_context.h"
#include "mfq/cpu_expert_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <iostream>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace mfq::cuda {
namespace {
thread_local cudaStream_t flushed_window_stream=nullptr;
thread_local bool has_flushed_window=false;
struct WindowSubmission {
    cudaStream_t previous=flushed_window_stream;
    bool previously_flushed=has_flushed_window;
    ~WindowSubmission(){flushed_window_stream=previous;has_flushed_window=previously_flushed;}
};
void report_window_error(const char* operation,const char* phase) noexcept {
    try {
        std::rethrow_exception(std::current_exception());
    } catch(const std::exception& error) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=%s\n",operation,phase,error.what());
    } catch(...) {
        std::fprintf(stderr,"decode_window_error operation=%s phase=%s error=unknown\n",operation,phase);
    }
    std::fflush(stderr);
}
}
void check_stream_progress(cudaStream_t stream) {
    const auto status=cudaStreamQuery(stream);
    if(status!=cudaSuccess && status!=cudaErrorNotReady)MFQ_NATIVE_CUDA_CHECK(status);
}
// Adapted from Strata's session_run_token (src/core/session.cpp).
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// MIT license: ../../../third_party/strata.LICENSE
// One WDDM flush submits the complete token graph. Layer waits keep their
// bounded fallback polling; standalone publications still flush immediately.
void check_route_submission(cudaStream_t stream) {
    if(!has_flushed_window || stream!=flushed_window_stream)check_stream_progress(stream);
}
void publish_mapped_flag(uint32_t* flag) noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *static_cast<volatile uint32_t*>(flag)=1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms) {
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();auto flush=start;
    const auto poll_interval=std::chrono::microseconds(runtime_options::route_poll_interval_us());
    const auto* option=std::getenv("MFQ_MOE_ROUTE_SPIN");
    const bool spin=!option || option[0]!='0';
    uint32_t spins=0;
    if(*static_cast<const volatile uint32_t*>(flag)<1)check_route_submission(stream);
    while(*static_cast<const volatile uint32_t*>(flag)<1) {
        if(spin) {
            // Strata checks its mapped doorbell after pause and consults the
            // clock every 1024 spins (src/core/session.cpp, MIT license above).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if((++spins&1023u)!=0)continue;
        }
        const auto now=Clock::now();
        if(now-flush>=poll_interval) {
            flush=now;const auto q=cudaStreamQuery(stream);
            if(q!=cudaSuccess && q!=cudaErrorNotReady)MFQ_NATIVE_CUDA_CHECK(q);
            if(q==cudaSuccess && *static_cast<const volatile uint32_t*>(flag)<1)
                throw std::runtime_error("MFQ graph finished before publishing its next layer");
        }
        if(now-start>std::chrono::milliseconds(timeout_ms))
            throw std::runtime_error("MFQ graph route publication stalled");
        if(!spin)std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}
namespace { thread_local DecodeWindow* active=nullptr; }
struct DecodeWindow::LayerPoint {
    int layer;
    std::string phase;
    cudaEvent_t event=nullptr;
    uint64_t samples=0;
    int64_t total_ns=0,min_ns=0,max_ns=0;
    LayerPoint(int l,const char* p):layer(l),phase(p) {MFQ_NATIVE_CUDA_CHECK(cudaEventCreate(&event));}
    ~LayerPoint(){if(event)cudaEventDestroy(event);}
};
DecodeWindow* DecodeWindow::recording() noexcept { return active; }
DecodeWindow::DecodeWindow(cudaStream_t stream,bool layer_profile):stream_(stream),graphs_(stream),layer_profile_(layer_profile) {}
void DecodeWindow::mark(int layer,const char* phase) {
    if(!layer_profile_)return;
    // Scope 2 keeps the MFE phase boundaries. Timing every small operation
    // measurably perturbs Windows graph replay; ordinary execution adds none.
    if(const auto* scope=std::getenv("MFQ_TRACE_LAYER_TIMINGS");scope &&
            (std::strcmp(scope,"2")==0 || std::strcmp(scope,"3")==0)) {
        constexpr const char* phases[]={"window_begin","window_end","hot_begin","hot_end",
            "dma_end","transfer_gate_up_end","cpu_wait_begin","transfer_end"};
        // Scope 3 separates the attention/residual work, route publication and
        // input preparation (including the mapped host-plan wait). Scope 2
        // retains its existing event count for historical comparisons.
        constexpr const char* boundaries[]={"attention_core_begin","attention_core_end",
            "ffn_pre_end","expert_begin","shared_end","ple_begin","ple_end"};
        if(std::none_of(std::begin(phases),std::end(phases),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;}) &&
            (std::strcmp(scope,"3")!=0 || std::none_of(std::begin(boundaries),std::end(boundaries),
                [&](const char* allowed){return std::strcmp(phase,allowed)==0;})))return;
    }
    if(active!=this)throw std::logic_error("MFQ layer timing outside window recording");
    const auto key=std::make_pair(layer,std::string(phase));
    auto found=layer_point_index_.find(key);
    if(found==layer_point_index_.end()) {
        auto point=std::make_unique<LayerPoint>(layer,phase);
        found=layer_point_index_.emplace(key,point.get()).first;layer_points_.push_back(std::move(point));
    }
    cudaStreamCaptureStatus capture;
    MFQ_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
    MFQ_NATIVE_CUDA_CHECK(cudaEventRecordWithFlags(found->second->event,stream_,
        capture==cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
}
void DecodeWindow::observe_layers() {
    for(std::size_t i=1;i<layer_points_.size();++i) {
        auto& current=*layer_points_[i];float ms=0;
        MFQ_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,layer_points_[i-1]->event,current.event));
        const auto ns=static_cast<int64_t>(ms*1e6);
        current.total_ns+=ns;current.min_ns=current.samples ? std::min(current.min_ns,ns) : ns;
        current.max_ns=std::max(current.max_ns,ns);++current.samples;
    }
}
std::vector<DecodeWindow::GpuPhase> DecodeWindow::gpu_phases() const {
    std::vector<GpuPhase> phases;
    for(std::size_t i=1;i<layer_points_.size();++i) {
        const auto& previous=*layer_points_[i-1];const auto& current=*layer_points_[i];
        phases.push_back({previous.layer,current.layer,previous.phase,current.phase,
            current.samples,current.total_ns,current.min_ns,current.max_ns});
    }
    return phases;
}
DecodeWindow::~DecodeWindow() {
    // Mapped flags and task owners must survive the last consumer, including
    // a graph released by a failed CPU or I/O phase.
    for(auto& task:tasks_)if(task.cancel)task.cancel();
    (void)cudaStreamSynchronize(stream_);
    graphs_.clear();
    for(auto& task:tasks_)if(task.cleanup)task.cleanup();
    if(replays_ && std::getenv("MFQ_TRACE_NATIVE_CUDA_GRAPH"))
        std::cerr<<"native_cuda_window tasks="<<tasks_.size()<<" replays="<<replays_
            <<" capture_ns="<<capture_ns_<<" replay_ns="<<replay_ns_
            <<" reset_ns="<<reset_ns_<<" launch_ns="<<launch_ns_<<" serve_ns="<<serve_ns_
            <<" wait_ns="<<wait_ns_<<" finish_ns="<<finish_ns_<<" profile_ns="<<profile_ns_<<std::endl;
    for(const auto& phase:gpu_phases())if(phase.samples)
        std::cerr<<"native_layer_gpu from_layer="<<phase.from_layer<<" layer="<<phase.layer
            <<" from="<<phase.from<<" to="<<phase.to<<" samples="<<phase.samples
            <<" total_ns="<<phase.total_ns<<" min_ns="<<phase.min_ns<<" max_ns="<<phase.max_ns<<std::endl;
}
void DecodeWindow::enroll(Task task) {
    if(active!=this)throw std::logic_error("MFQ task outside window capture");
    if(std::any_of(tasks_.begin(),tasks_.end(),[&](const auto& t){return t.identity==task.identity;}))
        throw std::logic_error("one MFQ operator cell was recorded twice in a window");
    tasks_.push_back(std::move(task));
}
void DecodeWindow::capture(const std::function<void()>& body,const std::function<void()>& restore,
        const std::function<void()>& release_output) {
    if(valid())return;
    if(active)throw std::logic_error("nested MFQ window capture");
    struct Recording {
        explicit Recording(DecodeWindow* window){active=window;}
        ~Recording(){active=nullptr;}
    } recording(this);
    const auto capture_started=std::chrono::steady_clock::now();
    const auto* trace_option=std::getenv("MFQ_TRACE_MOE_SERVE");
    const bool trace=trace_option && trace_option[0] && trace_option[0]!='0';
    const char* phase="prepare";
    const auto checkpoint=[&](const char* next) {
        phase=next;
        if(trace) {
            std::fprintf(stderr,"decode_window operation=capture phase=%s tasks=%zu\n",phase,tasks_.size());
            std::fflush(stderr);
        }
    };
    const auto release_warm_outputs=[&] {
        if(release_output)release_output();
        for(auto& task:tasks_)if(task.release_warm_output)task.release_warm_output();
    };
    // Preparation allocates every persistent cell and warms CUDA workspaces.
    // Warmup/capture may overwrite recurrent state, so both restore it.
    try {
        checkpoint("prepare");
        body();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));restore();
        release_warm_outputs();
        tasks_.clear();
        checkpoint("warmup");
        {int device=0;MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
            GraphWarmupScope warmup(default_context(device),stream_);body();}
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));restore();
        release_warm_outputs();
        const auto warm_tasks=tasks_.size();tasks_.clear();
        checkpoint("record");
        std::string error;
        if(!graphs_.record(0,1,body,error))throw std::runtime_error(error);
        if(tasks_.size()!=warm_tasks)throw std::logic_error("MFQ window task order changed during capture");
        checkpoint("restore");
        restore();MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream_));
        capture_ns_=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-capture_started).count();
        checkpoint("complete");
    } catch(...) {
        report_window_error("capture",phase);
        for(auto& task:tasks_)if(task.cancel)task.cancel();
        (void)cudaStreamSynchronize(stream_);restore();failed_=true;throw;
    }
}
void DecodeWindow::run() {
    if(failed_ || !valid())throw std::runtime_error("MFQ window is unavailable after a failed execution");
    const mfq::cpu::ScopedHostAffinity placement(runtime_options::decode_host_pin()
        ?mfq::cpu::planned_host_core():-1);
    const auto* trace_option=std::getenv("MFQ_TRACE_MOE_SERVE");
    const bool trace=trace_option && trace_option[0] && trace_option[0]!='0' && (replays_<2 || replays_%64==0);
    const char* phase="reset";
    const auto checkpoint=[&](const char* next) {
        phase=next;
        if(trace) {
            std::fprintf(stderr,"decode_window operation=replay replay=%llu phase=%s tasks=%zu\n",
                static_cast<unsigned long long>(replays_),phase,tasks_.size());
            std::fflush(stderr);
        }
    };
    try {
        WindowSubmission submission;
        using Clock=std::chrono::steady_clock;
        auto previous=Clock::now();const auto started=previous;
        const auto sample=[&](int64_t& total) {
            const auto now=Clock::now();
            total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-previous).count();previous=now;
        };
        checkpoint("reset");
        for(auto& task:tasks_)task.reset();
        sample(reset_ns_);
        std::string error;
        const auto* graph=graphs_.find(0,1);
        checkpoint("launch");
        if(!graph->launch(stream_,error))throw std::runtime_error(error);
        checkpoint("submit");
        if(runtime_options::single_window_flush()) {
            check_stream_progress(stream_);
            flushed_window_stream=stream_;
            has_flushed_window=true;
        }
        sample(launch_ns_);
        checkpoint("serve");
        for(auto& task:tasks_)task.serve();
        sample(serve_ns_);
        checkpoint("wait");
        if(!graph->wait_ms(60000))throw std::runtime_error("MFQ model window did not finish");
        sample(wait_ns_);
        // Only after all GPU consumers finish may adaptation overwrite L1.
        checkpoint("finish");
        for(auto& task:tasks_)if(task.finish)task.finish();
        sample(finish_ns_);
        observe_layers();sample(profile_ns_);
        replay_ns_+=std::chrono::duration_cast<std::chrono::nanoseconds>(previous-started).count();++replays_;
        checkpoint("complete");
    } catch(...) {
        report_window_error("replay",phase);
        failed_=true;
        for(auto& task:tasks_)if(task.cancel)task.cancel();
        (void)cudaStreamSynchronize(stream_);throw;
    }
}
}
