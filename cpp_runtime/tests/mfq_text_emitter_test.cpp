#include "text_emitter.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    std::string output;
    mfq::engine::TextEmitter emitter({"STOP"}, [&](const std::string& chunk) {
        output += chunk;
        return true;
    });
    assert(emitter.append("hello ST"));
    assert(output == "hello ");
    assert(!emitter.append("OP ignored"));
    assert(emitter.stopped());
    assert(emitter.flush());
    assert(output == "hello ");

    std::string utf8;
    mfq::engine::TextEmitter split_utf8({}, [&](const std::string& chunk) {
        utf8 += chunk;
        return true;
    });
    assert(split_utf8.append("\xE4\xB8"));
    assert(utf8.empty());
    assert(split_utf8.append("\xAD"));
    assert(split_utf8.flush());
    assert(utf8 == "\xE4\xB8\xAD");

    std::string truncated;
    mfq::engine::TextEmitter incomplete({}, [&](const std::string& chunk) {
        truncated += chunk;
        return true;
    });
    assert(incomplete.append("\xE4"));
    assert(incomplete.flush());
    assert(truncated == "\xEF\xBF\xBD");
}
