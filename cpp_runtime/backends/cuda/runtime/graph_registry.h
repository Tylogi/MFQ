#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace mfq::cuda {
class CapturedGraph {
public:
    ~CapturedGraph();
    CapturedGraph(CapturedGraph&&) noexcept;
    CapturedGraph& operator=(CapturedGraph&&) noexcept;
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    bool launch(void* stream, std::string& error) const;
    bool wait_ms(int timeout_ms) const;
    std::size_t nodes() const;
    bool valid() const;
private:
    friend class GraphRegistry;
    CapturedGraph(void* stream, const std::function<void()>& body);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class GraphRegistry {
public:
    explicit GraphRegistry(void* stream) : stream_(stream) {}
    bool record(std::uint64_t identity, int tokens,
                const std::function<void()>& body, std::string& error);
    const CapturedGraph* find(std::uint64_t identity, int tokens) const;
    bool launch(std::uint64_t identity, int tokens, int timeout_ms, std::string& error) const;
    std::size_t size() const { return graphs_.size(); }
    std::size_t captures() const { return captures_; }
    void clear() { graphs_.clear(); }
private:
    void* stream_;
    std::map<std::pair<std::uint64_t, int>, CapturedGraph> graphs_;
    std::size_t captures_ = 0;
};
}
