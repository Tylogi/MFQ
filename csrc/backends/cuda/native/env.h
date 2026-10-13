#pragma once

#include <string>
#include <vector>

namespace mfq::cuda {

// Single seam for every MFQ_* runtime knob in the CUDA backend.  Two things sit
// behind it: provenance (every name the process looked up is recorded, so the
// effective knob set is dumpable) and stability (after env_freeze() a name is
// memoized, so a knob cannot change value between a CUDA graph capture and the
// replays of it).
//
// The returned pointer stays valid until that name is env_clear()ed.
const char* env_lookup(const char* name);

// Strict readers: a value that is set but not parseable throws std::runtime_error.
bool env_flag(const char* name, bool when_unset);
long long env_number(const char* name, long long when_unset);
std::string env_text(const char* name, std::string when_unset);

// Frozen mode is entered once per process by the engine after it has read its
// configuration.  Unit tests that steer knobs with env_set/env_clear keep
// working in frozen mode; the memo entry for that name is dropped.
void env_freeze();
bool env_frozen() noexcept;
void env_set(const char* name, const char* value);   // throws if the process environment rejects it
void env_clear(const char* name);

struct EnvKnob {
    std::string name;
    std::string value;  // empty when the name is not set
};

// Knobs in first-lookup order.  env_dump() lists only the ones that are set.
std::vector<EnvKnob> env_snapshot();
std::string env_dump();

}  // namespace mfq::cuda
