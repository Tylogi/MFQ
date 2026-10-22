#include "cuda_format_comparison/cases.h"
#include "numerical_check.h"

#include <array>
#include <cassert>
#include <limits>
#include <sstream>
#include <string>

template<class Function>
void expect_failure(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find(message) != std::string::npos);
        return;
    }
    assert(false && "expected validation failure");
}

void check_manifest_keys() {
    const std::string first = "NINT4\tmodel-a.mfq\tgate\t0\t640\t2560\t4.5\n";
    for (const auto& second : {
             "NINT4\tmodel-b.mfq\tgate\t0\t640\t2560\t4.5\n",
             "NINT4\tmodel-a.mfq\tup\t0\t640\t2560\t4.5\n",
             "NINT4\tmodel-a.mfq\tgate\t1\t640\t2560\t4.5\n",
             "NINT4\tmodel-a.mfq\tgate\t0\t640\t2560\t4.5\n"}) {
        std::istringstream manifest(first + second);
        int exports = 0;
        expect_failure([&] {
            for (const auto& entry : format_bench::read_cases(manifest)) {
                (void)entry;
                ++exports;
            }
        }, "duplicate benchmark case key: NINT4_640x2560");
        assert(exports == 0);
    }
    std::istringstream distinct(first +
        "NINT4-b\tmodel-b.mfq\tgate\t0\t640\t2560\t4.5\n"
        "NINT4\tmodel-a.mfq\tdown\t0\t2560\t640\t4.5\n");
    const auto cases = format_bench::read_cases(distinct);
    assert(cases.size() == 3);
    assert(cases[0].name() == "NINT4_640x2560");
    assert(cases[1].name() == "NINT4-b_640x2560");
    assert(cases[2].name() == "NINT4_2560x640");
}

void check_numerical_values() {
    using mfq::bench::max_absolute_difference;
    const std::array<float, 3> reference{1.0f, -2.0f, 0.0f};
    const std::array<float, 3> candidate{1.25f, -2.5f, 0.0f};
    assert(max_absolute_difference(reference.data(), reference.data(), reference.size()) == 0.0f);
    assert(max_absolute_difference(reference.data(), candidate.data(), reference.size()) == 0.5f);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity()}) {
        for (std::size_t index = 0; index < candidate.size(); ++index) {
            auto invalid = candidate;
            invalid[index] = value;
            expect_failure([&] {
                max_absolute_difference(reference.data(), invalid.data(), reference.size());
            }, "non-finite");
            expect_failure([&] {
                max_absolute_difference(invalid.data(), reference.data(), reference.size());
            }, "non-finite");
        }
    }
    const float positive = std::numeric_limits<float>::max(), negative = -positive;
    expect_failure([&] {
        max_absolute_difference(&positive, &negative, 1);
    }, "non-finite");
}

int main() {
    check_manifest_keys();
    check_numerical_values();
}
