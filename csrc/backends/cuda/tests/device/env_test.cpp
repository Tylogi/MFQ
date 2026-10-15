// Interface test for the single MFQ_* knob seam: presence, strict parsing, the
// retired-name alias, frozen memoization and provenance recording.
#include "device/env.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "failed: " << what << '\n';
    }
}

}  // namespace

int main() {
    mfq::cuda::env_clear("MFQ_TEST_KNOB");
    check(mfq::cuda::env_lookup("MFQ_TEST_KNOB") == nullptr, "an unset knob reads as unset");
    check(mfq::cuda::env_flag("MFQ_TEST_KNOB", true), "an unset flag falls back");
    check(mfq::cuda::env_number("MFQ_TEST_KNOB", 7) == 7, "an unset number falls back");
    check(mfq::cuda::env_text("MFQ_TEST_KNOB", "d") == "d", "an unset text falls back");

    mfq::cuda::env_set("MFQ_TEST_KNOB", "0");
    check(!mfq::cuda::env_flag("MFQ_TEST_KNOB", true), "a set flag overrides the fallback");
    check(mfq::cuda::env_number("MFQ_TEST_KNOB", 7) == 0, "a set number is parsed");

    bool rejected = false;
    mfq::cuda::env_set("MFQ_TEST_KNOB", "on");
    try {
        mfq::cuda::env_flag("MFQ_TEST_KNOB", false);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "a non-numeric value is rejected instead of guessed");

    mfq::cuda::env_clear("MFQ_DISABLE_NIQ_FUSION");
    check(mfq::cuda::env_lookup("MFQ_DISABLE_NVQ_FUSION") == nullptr,
          "the NVQ knob is unset when neither name is set");
    mfq::cuda::env_set("MFQ_DISABLE_NIQ_FUSION", "1");
    const auto* aliased = mfq::cuda::env_lookup("MFQ_DISABLE_NVQ_FUSION");
    check(aliased && std::string(aliased) == "1", "the retired NIQ name still steers NVQ");
    mfq::cuda::env_clear("MFQ_DISABLE_NIQ_FUSION");

    mfq::cuda::env_freeze();
    check(mfq::cuda::env_frozen(), "freeze is observable");
    mfq::cuda::env_set("MFQ_TEST_KNOB", "5");
    check(mfq::cuda::env_number("MFQ_TEST_KNOB", 0) == 5, "env_set overrides a frozen memo");
    mfq::cuda::env_clear("MFQ_TEST_KNOB");
    check(mfq::cuda::env_lookup("MFQ_TEST_KNOB") == nullptr, "env_clear drops the memo entry");

    const auto snapshot = mfq::cuda::env_snapshot();
    check(!snapshot.empty(), "every looked-up knob is recorded");
    mfq::cuda::env_set("MFQ_TEST_KNOB", "9");
    check(mfq::cuda::env_dump().find("MFQ_TEST_KNOB=9") != std::string::npos,
          "the dump lists the knobs that are set");
    mfq::cuda::env_clear("MFQ_TEST_KNOB");
    check(mfq::cuda::env_dump().find("MFQ_TEST_KNOB") == std::string::npos,
          "the dump omits knobs that are not set");

    if (failures) return 1;
    std::cout << "cuda env knobs passed\n";
    return 0;
}
