#include "models/common/axial_patch_vision.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using mfq::models::axial_patch_vision::aligned_grid;
using mfq::models::axial_patch_vision::Config;
using mfq::models::axial_patch_vision::gather_indices;
using mfq::models::axial_patch_vision::span_token_types;
using mfq::models::axial_patch_vision::validate_span;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        Config config{
            1024, 24, 16, 2816, 14, 3, 5120, 10'000.0, 1e-6};
        config.validate();
        require(
            config.patch_width() == 3 * 14 * 14 &&
                config.head_dim() == 64,
            "axial patch vision derived geometry mismatch");

        bool rejected = false;
        try {
            Config bad = config;
            bad.downsample_ratio = 0;
            bad.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid axial patch vision config was accepted");

        const auto non_square = aligned_grid(7, 5, 3);
        require(
            non_square.height == 3 && non_square.width == 2 &&
                non_square.count() == 6,
            "aligner ceil geometry mismatch");
        const auto indices = gather_indices(7, 5, 3);
        require(
            indices.size() == 54 && indices.front() == 0 &&
                indices.at(1) == 1 && indices.at(2) == 2 &&
                indices.at(3) == 5 && indices.back() == 35,
            "aligner gather geometry mismatch");

        const auto types = span_token_types(2, 3);
        require(
            types == std::vector<std::int64_t>{
                0, 1, 1, 1, 2, 1, 1, 1, 2, 3},
            "V4.1 row-major image span mismatch");
        require(
            types.size() ==
                static_cast<std::size_t>(2 * (3 + 1) + 2),
            "V4.1 image span token count mismatch");

        validate_span(2, 13, 7, 5, 3, 2, 13, 3, span_token_types(3, 2));
        rejected = false;
        try {
            validate_span(2, 13, 7, 5, 3, 2, 13, 3, span_token_types(2, 3));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "image span accepted an inconsistent token layout");

        rejected = false;
        try {
            validate_span(0, 10, 7, 5, 3, 2, 13, 3, span_token_types(3, 2));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "image span accepted a truncated placeholder count");

        rejected = false;
        try {
            validate_span(2, 13, 6, 5, 3, 2, 13, 3, span_token_types(3, 2));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "image span accepted a nonintegral aligner grid");

        const auto first_types = span_token_types(3, 2);
        const auto second_types = span_token_types(1, 3);
        validate_span(0, 11, 7, 5, 3, 2, 19, 3, first_types);
        validate_span(15, 21, 3, 9, 1, 3, 21, 3, second_types);

        std::cout << "DeepSeek-V4.1 vision layout tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 vision layout test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
