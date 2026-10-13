#include "grid_vision.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    try {
        require(
            mfq::grid_vision_canonical_name(
                mfq::GridVisionTensorRole::block_mlp_up_weight, 12) ==
                "vision.block.12.mlp.up.weight",
            "canonical vision name mismatch");
        const std::vector<mfq::GridShape> grids{{1, 4, 4}};
        const auto layout = mfq::make_grid_vision_layout(grids, 2);
        require(layout.patch_count == 16 &&
                layout.segment_lengths == std::vector<int32_t>{16} &&
                layout.positions == std::vector<int32_t>{
                    0,0, 0,1, 1,0, 1,1, 0,2, 0,3, 1,2, 1,3,
                    2,0, 2,1, 3,0, 3,1, 2,2, 2,3, 3,2, 3,3},
            "grid-ViT block-major layout mismatch");
        const auto interpolation =
            mfq::make_learned_position_interpolation(grids, 2, 2);
        require(interpolation.patch_count == 16 &&
                interpolation.indices.size() == 64 &&
                interpolation.weights.size() == 64,
            "learned-position interpolation mismatch");
        const auto positions = mfq::build_grid_mrope_positions(
            {7, 100, 100, 100, 100, 8}, 100, 101, 2, {{1, 4, 4}}, {});
        require(positions.token_count == 6 && positions.decode_delta == -2 &&
                positions.values == std::vector<int32_t>{
                    0,1,1,1,1,3, 0,1,1,2,2,3, 0,1,2,1,2,3},
            "grid-MRoPE positions mismatch");
        bool rejected = false;
        try {
            (void)mfq::build_grid_mrope_positions(
                {100}, 100, 101, 2, {{2, 4, 4}}, {});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "multi-frame image grid was accepted");
        std::cout << "grid-Vision core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grid-Vision core test failed: " << error.what() << '\n';
        return 1;
    }
}
