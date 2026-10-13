#include "text_emitter.h"
#include <cassert>
int main() {
    mfq::engine::TextEmitter stop({"STOP", ""});
    assert(stop.append("hello ST")); assert(stop.take() == "hello ");
    assert(!stop.append("OP ignored")); assert(stop.stopped());
    stop.flush(); assert(stop.take().empty());
    assert(!stop.append("more")); assert(stop.take().empty());
    mfq::engine::TextEmitter utf8({});
    assert(utf8.append("\xE4\xB8")); assert(utf8.take().empty());
    assert(utf8.append("\xAD")); assert(utf8.take() == "\xE4\xB8\xAD");
    assert(utf8.append("\xE4")); assert(utf8.flush()); assert(utf8.take() == "\xEF\xBF\xBD");
}
