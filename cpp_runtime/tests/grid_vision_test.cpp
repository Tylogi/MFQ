#include "grid_vision.h"
#include "../models/common/vision_prompt.h"

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
        mfq::GridVisionConfig config;
        config.hidden_size = 8; config.intermediate_size = 16; config.depth = 1;
        config.num_heads = 2; config.patch_size = 2; config.temporal_patch_size = 2;
        config.spatial_merge_size = 2; config.out_hidden_size = 32;
        config.validate(false);
        rejected = false;
        try { config.validate(); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "learned-position tower accepted a missing position table");
        const auto embed = [](const auto& ids) { return std::vector<float>(ids.size(), -1.f); };
        const auto scatter = [](auto& output, const auto& merged, const auto& indices) {
            for (size_t i = 0; i < indices.size(); ++i) output[indices[i]] = merged[i];
        };
        const auto injected = mfq::models::inject_vision_embeddings(
            std::vector<int64_t>{7, 100, 101, 100, 8}, std::vector<float>{1.f, 2.f, 3.f},
            3, 100, 101, true, embed, scatter);
        require(injected == std::vector<float>{-1.f, 1.f, 2.f, 3.f, -1.f},
            "mixed image/video embeddings lost prompt order");
        for (bool accept_video : {false, true}) {
            rejected = false;
            try {
                (void)mfq::models::inject_vision_embeddings(std::vector<int64_t>{100, 101},
                    std::vector<float>{1.f}, 1, 100, 101, accept_video, embed, scatter);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid video/count prompt contract was accepted");
        }
        std::cout << "grid-Vision core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "grid-Vision core test failed: " << error.what() << '\n';
        return 1;
    }
}
