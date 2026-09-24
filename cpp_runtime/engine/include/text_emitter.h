#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace mfq::engine {

// Model output assembly: stop sequences and UTF-8 boundaries are independent
// of the wire protocol and of the device executing the model.
class TextEmitter {
public:
    using Emit = std::function<bool(const std::string&)>;

    TextEmitter(std::vector<std::string> stops, Emit emit);
    bool append(const std::string& piece);
    bool flush();
    bool stopped() const noexcept { return stopped_; }

private:
    bool emit_prefix(std::size_t limit);
    bool emit_bytes(std::size_t count);

    std::vector<std::string> stops_;
    Emit emit_;
    std::string pending_;
    bool stopped_ = false;
};

} // namespace mfq::engine
