#pragma once

#include <cstddef>
#include <utility>
#include <string>
#include <vector>

namespace mfq::engine {

// Model output assembly: stop sequences and UTF-8 boundaries are independent
// of the wire protocol and of the device executing the model.
class TextEmitter {
public:
    explicit TextEmitter(std::vector<std::string> stops);
    std::string take() { return std::exchange(output_, {}); }
    bool append(const std::string& piece);
    bool flush();
    bool stopped() const noexcept { return stopped_; }

private:
    bool emit_prefix(std::size_t limit);
    bool emit_bytes(std::size_t count);

    std::vector<std::string> stops_;
    std::string output_;
    std::string pending_;
    bool stopped_ = false;
};

} // namespace mfq::engine
