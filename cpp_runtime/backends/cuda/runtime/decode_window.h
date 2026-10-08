#pragma once
#include "graph_registry.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace mfq::cuda {
void wait_route_publication(const uint32_t* flag,cudaStream_t stream,int timeout_ms=60000);
void publish_mapped_flag(uint32_t* flag) noexcept;
void check_stream_progress(cudaStream_t stream);
// A captured window submits the model once. Each expert operator records its
// GPU body and supplies the corresponding host task; replay serves those tasks
// in graph order without launching a graph or synchronizing at every layer.
class DecodeWindow {
public:
    struct Task {
        const void* identity;
        std::function<void()> reset, serve, finish, cancel;
        std::function<void()> cleanup;
        std::function<void()> release_warm_output;
    };
    struct GpuPhase {
        int from_layer,layer;
        std::string from,to;
        uint64_t samples;
        int64_t total_ns,min_ns,max_ns;
    };
    explicit DecodeWindow(cudaStream_t stream,bool layer_profile=false);
    ~DecodeWindow();
    DecodeWindow(const DecodeWindow&)=delete;
    DecodeWindow& operator=(const DecodeWindow&)=delete;
    void capture(const std::function<void()>& body,
                 const std::function<void()>& restore,
                 const std::function<void()>& release_output = {});
    void enroll(Task task);
    void run();
    void mark(int layer,const char* phase);
    std::vector<GpuPhase> gpu_phases() const;
    std::size_t nodes() const { return graphs_.find(0,1)->nodes(); }
    bool valid() const { return graphs_.find(0,1)!=nullptr; }
    static DecodeWindow* recording() noexcept;
private:
    cudaStream_t stream_;
    GraphRegistry graphs_;
    struct LayerPoint;
    std::vector<std::unique_ptr<LayerPoint>> layer_points_;
    std::map<std::pair<int,std::string>,LayerPoint*> layer_point_index_;
    bool layer_profile_;
    void observe_layers();
    std::vector<Task> tasks_;
    bool failed_=false;
    int64_t capture_ns_=0,replay_ns_=0,reset_ns_=0,launch_ns_=0,serve_ns_=0,wait_ns_=0,finish_ns_=0,profile_ns_=0;
    uint64_t replays_=0;
};
}
