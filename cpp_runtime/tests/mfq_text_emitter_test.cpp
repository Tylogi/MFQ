#include "text_emitter.h"
#include <cassert>
#include <string>
namespace {
std::string emit_all(const std::string& text) {
    mfq::engine::TextEmitter emitter({});
    assert(emitter.append(text)); assert(emitter.flush());
    return emitter.take();
}
}
int main() {
    const std::string bad = "\xEF\xBF\xBD";
    mfq::engine::TextEmitter stop({"STOP", ""});
    assert(stop.append("hello ST")); assert(stop.take() == "hello ");
    assert(!stop.append("OP ignored")); assert(stop.stopped());
    stop.flush(); assert(stop.take().empty());
    assert(!stop.append("more")); assert(stop.take().empty());
    mfq::engine::TextEmitter utf8({});
    assert(utf8.append("\xE4\xB8")); assert(utf8.take().empty());
    assert(utf8.append("\xAD")); assert(utf8.take() == "\xE4\xB8\xAD");
    assert(utf8.append("\xE4")); assert(utf8.flush()); assert(utf8.take() == bad);

    mfq::engine::TextEmitter invalid({});
    assert(invalid.append("prefix \x80")); assert(invalid.take() == "prefix " + bad);
    assert(invalid.append(" ASCII suffix")); assert(invalid.take() == " ASCII suffix");
    assert(invalid.append("\xE4\xB8")); assert(invalid.take().empty());
    assert(invalid.append("A\xF0\x9F")); assert(invalid.take() == bad + "A");
    assert(invalid.append("\x98\x80")); assert(invalid.take() == "\xF0\x9F\x98\x80");
    assert(invalid.flush()); assert(invalid.take().empty());

    assert(emit_all("a\xC0\xAF" "b") == "a" + bad + bad + "b");
    assert(emit_all("\xE0\x80\xAF") == bad + bad + bad);
    assert(emit_all("\xED\xA0\x80") == bad + bad + bad);
    assert(emit_all("\xF4\x90\x80\x80") == bad + bad + bad + bad);
    assert(emit_all("\xF5\xFF") == bad + bad);
    assert(emit_all("\xF0\x9F\x98") == bad);
    assert(emit_all("\xF4\x8F\xBF\xBF" "\xED\x9F\xBF") == "\xF4\x8F\xBF\xBF" "\xED\x9F\xBF");

    mfq::engine::TextEmitter stopped({"STOP"});
    assert(!stopped.append("a\x80" "b\xE4STOP")); assert(stopped.take() == "a" + bad + "b" + bad);
}
