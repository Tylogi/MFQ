#include "text_emitter.h"

#include <algorithm>
#include <utility>

namespace mfq::engine {
namespace {

constexpr char kReplacement[] = "\xEF\xBF\xBD";

struct Utf8Step {
    std::size_t width;
    bool valid;
    bool truncated;
};

// Decodes one code point at value[i] with the well-formed byte ranges of
// Unicode Table 3-7, so overlong forms, surrogates and values past U+10FFFF
// are rejected. An invalid step covers one maximal subpart.
Utf8Step utf8_step(const std::string& value, std::size_t i, std::size_t limit) {
    const unsigned char lead = static_cast<unsigned char>(value[i]);
    if (lead < 0x80u) return {1, true, false};
    std::size_t width = 0;
    unsigned char low = 0x80u;
    unsigned char high = 0xBFu;
    if (lead >= 0xC2u && lead <= 0xDFu) {
        width = 2;
    } else if (lead >= 0xE0u && lead <= 0xEFu) {
        width = 3;
        if (lead == 0xE0u) low = 0xA0u;
        if (lead == 0xEDu) high = 0x9Fu;
    } else if (lead >= 0xF0u && lead <= 0xF4u) {
        width = 4;
        if (lead == 0xF0u) low = 0x90u;
        if (lead == 0xF4u) high = 0x8Fu;
    } else {
        return {1, false, false};
    }
    for (std::size_t j = 1; j < width; ++j) {
        if (i + j >= limit) return {j, false, true};
        const unsigned char next = static_cast<unsigned char>(value[i + j]);
        if (next < (j == 1 ? low : 0x80u) || next > (j == 1 ? high : 0xBFu)) {
            return {j, false, false};
        }
    }
    return {width, true, false};
}

// Appends value[0, limit) to out as well-formed UTF-8 and returns the bytes
// consumed. A sequence cut off at limit waits for the next piece unless at_end.
std::size_t append_utf8_prefix(const std::string& value, std::size_t limit, bool at_end, std::string& out) {
    limit = std::min(limit, value.size());
    std::size_t i = 0;
    while (i < limit) {
        const Utf8Step step = utf8_step(value, i, limit);
        if (step.truncated && !at_end) break;
        if (step.valid) out.append(value, i, step.width);
        else out += kReplacement;
        i += step.width;
    }
    return i;
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
        if (!emit_prefix(stop_pos, true)) return false;
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
    return emit_prefix(pending_.size(), true);
}

bool TextEmitter::emit_prefix(std::size_t limit, bool at_end) {
    pending_.erase(0, append_utf8_prefix(pending_, limit, at_end, output_));
    return true;
}

} // namespace mfq::engine
