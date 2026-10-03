#include "text_emitter.h"

#include <algorithm>
#include <utility>

namespace mfq::engine {
namespace {

std::size_t complete_utf8_prefix(const std::string& value, std::size_t limit) {
    std::size_t i = 0;
    std::size_t complete = 0;
    limit = std::min(limit, value.size());
    while (i < limit) {
        const unsigned char lead = static_cast<unsigned char>(value[i]);
        std::size_t width = 1;
        if ((lead & 0x80u) == 0) width = 1;
        else if ((lead & 0xE0u) == 0xC0u) width = 2;
        else if ((lead & 0xF0u) == 0xE0u) width = 3;
        else if ((lead & 0xF8u) == 0xF0u) width = 4;
        else break;
        if (i + width > limit) break;
        bool valid = true;
        for (std::size_t j = 1; j < width; ++j) {
            if ((static_cast<unsigned char>(value[i + j]) & 0xC0u) != 0x80u) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        i += width;
        complete = i;
    }
    return complete;
}

} // namespace

TextEmitter::TextEmitter(std::vector<std::string> stops)
    : stops_(std::move(stops)) {
    stops_.erase(std::remove(stops_.begin(), stops_.end(), std::string{}), stops_.end());
}

bool TextEmitter::append(const std::string& piece) {
    if (stopped_) return false;
    pending_ += piece;
    std::size_t stop_pos = std::string::npos;
    for (const auto& stop : stops_) {
        const std::size_t pos = pending_.find(stop);
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos)) {
            stop_pos = pos;
        }
    }
    if (stop_pos != std::string::npos) {
        if (!emit_prefix(stop_pos)) return false;
        pending_.clear();
        stopped_ = true;
        return false;
    }

    std::size_t retain = 0;
    for (const auto& stop : stops_) {
        const std::size_t max_prefix = std::min(stop.size() - 1, pending_.size());
        for (std::size_t n = 1; n <= max_prefix; ++n) {
            if (pending_.compare(pending_.size() - n, n, stop, 0, n) == 0) {
                retain = std::max(retain, n);
            }
        }
    }
    return emit_prefix(pending_.size() - retain);
}

bool TextEmitter::flush() {
    const std::size_t complete = complete_utf8_prefix(pending_, pending_.size());
    if (complete > 0 && !emit_bytes(complete)) return false;
    if (!pending_.empty()) {
        pending_.clear();
        output_ += "\xEF\xBF\xBD";
        return true;
    }
    return true;
}

bool TextEmitter::emit_prefix(std::size_t limit) {
    const std::size_t complete = complete_utf8_prefix(pending_, limit);
    return complete == 0 || emit_bytes(complete);
}

bool TextEmitter::emit_bytes(std::size_t count) {
    std::string text = pending_.substr(0, count);
    pending_.erase(0, count);
    output_ += text;
    return true;
}

} // namespace mfq::engine
