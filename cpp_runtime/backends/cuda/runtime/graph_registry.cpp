#include "graph_registry.h"
#include "mfq_cuda_context.h"
#include <chrono>
#include <thread>

namespace mfq::cuda {
struct CapturedGraph::Impl {
    Graph graph;
    Event completion;
    StreamHandle stream;
    Impl(void* native, const std::function<void()>& body) {
        int device = 0;
        MFQ_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
        stream = StreamHandle(device, static_cast<cudaStream_t>(native));
        StreamGuard selected(stream);
        try {
            graph.capture_begin_shared();
            body();
            MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
            graph.capture_end();
            if (!graph.nodes()) throw Error("CUDA graph body produced no work");
        } catch (...) {
            graph.reset();
            throw;
        }
    }
};
CapturedGraph::CapturedGraph(void* stream, const std::function<void()>& body)
    : impl_(std::make_unique<Impl>(stream, body)) {}
CapturedGraph::~CapturedGraph() = default;
CapturedGraph::CapturedGraph(CapturedGraph&&) noexcept = default;
CapturedGraph& CapturedGraph::operator=(CapturedGraph&&) noexcept = default;
bool CapturedGraph::valid() const { return impl_ && impl_->graph.valid(); }
std::size_t CapturedGraph::nodes() const { return impl_ ? impl_->graph.nodes() : 0; }
bool CapturedGraph::launch(void* stream, std::string& error) const {
    try {
        if (!valid() || impl_->stream.stream() != static_cast<cudaStream_t>(stream))
            throw Error("CUDA graph replay has no matching capture stream");
        StreamGuard selected(impl_->stream);
        impl_->graph.replay();
        impl_->completion.record(impl_->stream.stream());
        return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!valid() || timeout_ms < 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        if (impl_->completion.ready()) return true;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
bool GraphRegistry::record(std::uint64_t identity, int tokens,
        const std::function<void()>& body, std::string& error) {
    if (find(identity, tokens)) return true;
    try {
        CapturedGraph graph(stream_, body);
        graphs_.emplace(std::make_pair(identity, tokens), std::move(graph));
        ++captures_;
        return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
const CapturedGraph* GraphRegistry::find(std::uint64_t identity, int tokens) const {
    const auto found = graphs_.find({identity, tokens});
    return found == graphs_.end() ? nullptr : &found->second;
}
bool GraphRegistry::launch(std::uint64_t identity, int tokens, int timeout_ms, std::string& error) const {
    const auto* graph = find(identity, tokens);
    if (!graph) { error = "CUDA graph was not recorded for this operator and shape"; return false; }
    if (!graph->launch(stream_, error)) return false;
    if (!graph->wait_ms(timeout_ms)) { error = "CUDA graph completion exceeded its deadline"; return false; }
    return true;
}
}
