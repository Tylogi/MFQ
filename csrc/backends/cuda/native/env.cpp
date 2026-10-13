#include "env.h"

#include <charconv>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mfq::cuda {
namespace {

// Retired knob names that still steer their renamed successor.  Consulted only
// when the current name is unset, so an existing launch script keeps working.
const char* retired_alias(const char* name) {
    static const std::unordered_map<std::string, std::string> table = {
        {"MFQ_DISABLE_NVQ_FUSION", "MFQ_DISABLE_NIQ_FUSION"},
        {"MFQ_DISABLE_NVQ2_EXEC", "MFQ_DISABLE_NIQ2_EXEC"},
    };
    const auto entry = table.find(name);
    return entry == table.end() ? nullptr : entry->second.c_str();
}

struct State {
    std::mutex mutex;
    bool frozen = false;
    std::vector<std::string> order;
    std::unordered_set<std::string> seen;
    // Present only while frozen: nullopt means "looked up, not set".
    std::unordered_map<std::string, std::optional<std::string>> memo;
};

State& state() {
    static State instance;
    return instance;
}

void record(State& current, const std::string& name) {
    if (current.seen.insert(name).second) current.order.push_back(name);
}

}  // namespace

const char* env_lookup(const char* name) {
    State& current = state();
    const std::lock_guard lock(current.mutex);
    if (current.frozen) {
        const auto entry = current.memo.find(name);
        if (entry != current.memo.end())
            return entry->second ? entry->second->c_str() : nullptr;
    }
    const char* value = std::getenv(name);
    record(current, name);
    if (!value) {
        if (const char* alias = retired_alias(name)) {
            value = std::getenv(alias);
            record(current, alias);
        }
    }
    if (current.frozen)
        current.memo.emplace(name, value ? std::optional<std::string>(value) : std::nullopt);
    return value;
}

namespace {

std::string_view require_value(const char* name, const char* value) {
    if (value == nullptr || value[0] == '\0') return {};
    return std::string_view(value);
}

template <typename Number>
Number parse(const char* name, std::string_view text) {
    Number parsed = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::runtime_error(std::string("invalid ") + name);
    return parsed;
}

}  // namespace

bool env_flag(const char* name, bool when_unset) {
    const auto text = require_value(name, env_lookup(name));
    return text.empty() ? when_unset : parse<int>(name, text) != 0;
}

long long env_number(const char* name, long long when_unset) {
    const auto text = require_value(name, env_lookup(name));
    return text.empty() ? when_unset : parse<long long>(name, text);
}

std::string env_text(const char* name, std::string when_unset) {
    const auto text = require_value(name, env_lookup(name));
    return text.empty() ? when_unset : std::string(text);
}

void env_freeze() {
    State& current = state();
    const std::lock_guard lock(current.mutex);
    current.frozen = true;
}

bool env_frozen() noexcept {
    State& current = state();
    const std::lock_guard lock(current.mutex);
    return current.frozen;
}

void env_set(const char* name, const char* value) {
#ifdef _MSC_VER
    const int status = _putenv_s(name, value);
#else
    const int status = ::setenv(name, value, 1);
#endif
    if (status != 0) throw std::runtime_error(std::string("cannot set ") + name);
    State& current = state();
    const std::lock_guard lock(current.mutex);
    current.memo.erase(name);
}

void env_clear(const char* name) {
#ifdef _MSC_VER
    const int status = _putenv_s(name, "");
#else
    const int status = ::unsetenv(name);
#endif
    if (status != 0) throw std::runtime_error(std::string("cannot clear ") + name);
    State& current = state();
    const std::lock_guard lock(current.mutex);
    current.memo.erase(name);
}

std::vector<EnvKnob> env_snapshot() {
    State& current = state();
    std::vector<EnvKnob> knobs;
    {
        const std::lock_guard lock(current.mutex);
        knobs.reserve(current.order.size());
        for (const auto& name : current.order) knobs.push_back({name, {}});
    }
    for (auto& knob : knobs) {
        if (const char* value = env_lookup(knob.name.c_str())) knob.value = value;
    }
    return knobs;
}

std::string env_dump() {
    std::string text;
    for (const auto& knob : env_snapshot()) {
        if (knob.value.empty()) continue;
        text += knob.name;
        text += '=';
        text += knob.value;
        text += '\n';
    }
    return text;
}

}  // namespace mfq::cuda
