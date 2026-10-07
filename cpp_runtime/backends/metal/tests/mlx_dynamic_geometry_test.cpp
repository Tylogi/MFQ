#include "mlx_deepseek_v4_hc.h"
#include "mlx_deepseek_v41_mhc.h"
#include "mlx_dsa.h"
#include "mlx_mx.h"
#include "mlx_nint8_zero.h"
#include "mlx_sampling.h"
#include "mlx_sparse_attention.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mlx::core;

extern "C" __attribute__((noinline)) void mfq_dynamic_geometry_phase(int phase) {
    std::cout << "dynamic_geometry_phase=" << phase << std::endl;
}

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void finite(array value) {
    value = astype(value, float32);
    value.eval();
    for (std::size_t index = 0; index < value.size(); ++index) {
        require(std::isfinite(value.data<float>()[index]), "nonfinite dynamic geometry output");
    }
}

template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

mfq::metal::MlxNint8ZeroWeight nint_weight() {
    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, 64);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, 2);
    append<std::int64_t>(blob, 64);
    append<std::uint32_t>(blob, 2);
    append<std::uint32_t>(blob, 2);
    for (int block = 0; block < 4; ++block) {
        append<std::uint16_t>(blob, 0x3c00);
        blob.insert(blob.end(), 32, 1);
    }
    return mfq::metal::MlxNint8ZeroWeight::from_blob(blob);
}

void exercise(int tokens, const mfq::metal::MlxNint8ZeroWeight& nint,
              const mfq::metal::MlxMxWeight& mx) {
    const std::vector<std::int32_t> ids(tokens, 1);
    const array token_ids(ids.begin(), Shape{tokens});
    auto counts = mfq::metal::sample_token_counts_add(zeros({3}, int32), token_ids);
    counts.eval();
    require(counts.data<std::int32_t>()[0] == 0 &&
            counts.data<std::int32_t>()[1] == tokens &&
            counts.data<std::int32_t>()[2] == 0, "dynamic token histogram mismatch");
    finite(mfq::metal::sample_apply_penalties(ones({tokens, 3}), counts, 0.1, 0.1, 1.1));
    auto input = ones({tokens, 64}, float16);
    auto output = astype(nint.matmul(input), float32);
    output.eval();
    for (std::size_t index = 0; index < output.size(); ++index) {
        require(output.data<float>()[index] == 64.0f, "dynamic NINT8-0 matmul mismatch");
    }
    finite(nint.embedding(token_ids));
    finite(nint.grouped_row_matmul(ones({tokens, 2, 64}, float16), 2));
    finite(nint.grouped_row_matmul_inverse_rope(
        ones({tokens, 2, 64}, float16), 2,
        ones({tokens, 32}), zeros({tokens, 32}), 64, 64));
    finite(mx.matmul(input));
    finite(mx.embedding(token_ids));
    finite(mfq::metal::mlx_weighted_rms_rope_mxfp8_sim(
        ones({1, tokens, 512}, float16), ones({512}), 1e-6f, 64,
        ones({tokens, 32}), zeros({tokens, 32})));

    const auto residual = ones({1, tokens, 4, 64}, float16);
    const auto mixes = zeros({1, tokens, 24});
    const auto scale = ones({3});
    const auto base = zeros({24});
    const auto pre = mfq::metal::deepseek_v4_hc_pre(residual, mixes, scale, base);
    finite(pre.reduced);
    finite(mfq::metal::deepseek_v4_hc_pre_norm(
        residual, mixes, scale, base, ones({64})).reduced);
    finite(mfq::metal::deepseek_v4_hc_post(
        pre.reduced, residual, pre.post, pre.combination));
    const auto metadata = mfq::metal::deepseek_v41_hc_metadata_exact(mixes, scale, base);
    finite(metadata.combination);
    const auto large_residual = ones({1, tokens, 4, 5120}, float16);
    finite(mfq::metal::deepseek_v41_hc_collapse_norm(
        large_residual, metadata.pre, ones({5120})));
    finite(mfq::metal::deepseek_v41_hc_post(
        ones({1, tokens, 5120}, float16), large_residual,
        metadata.post, metadata.combination));

    const int keys = 129 + tokens;
    auto scores = astype(mfq::metal::mlx_dsa_indexer_scores(
        ones({1, tokens, 64, 128}, float16),
        ones({1, keys, 128}, float16), ones({1, tokens, 64}, float16), keys, 4), float32);
    scores.eval();
    require(std::isfinite(scores.data<float>()[0]), "dynamic DSA first visible score invalid");
    const auto topk = zeros({1, tokens, 8}, int32);
    const auto plan = mfq::metal::mlx_dsa_build_prefill_plan(topk, keys, 0, keys / 4, 4, 128);
    require(plan.first.shape(1) == tokens, "dynamic DSA plan shape mismatch");
    eval(plan.first, plan.second);
    const auto visible = mfq::metal::mlx_dsa_build_prefill_plan_visible(
        topk, zeros({1, tokens}, int32), zeros({1, tokens}, int32),
        keys, 0, keys / 4, 4, 128, 16);
    eval(visible.first, visible.second);
    const array length({keys}, int32);
    const auto decode_plan = mfq::metal::mlx_dsa_build_decode_plan(
        zeros({1, 1, 8}, int32), length, keys / 4, 4, 128);
    eval(decode_plan.first, decode_plan.second);
    finite(mfq::metal::mlx_sparse_selected_mla_attention(
        ones({1, 64, tokens, 512}), ones({1, keys, 512}, float16),
        zeros({1, tokens, 32}, int32), zeros({1, tokens, 32}, float16), zeros({64})));
    finite(mfq::metal::mlx_sparse_circular_mla_decode_attention(
        ones({1, 64, 1, 512}), ones({1, 128, 512}, float16),
        ones({1, keys, 512}, float16), keys / 4,
        zeros({1, 1, 8}, int32), zeros({64}), keys, 4, 128));
}

}

int main() {
    try {
        set_default_device(Device::gpu);
        const auto nint = nint_weight();
        const auto mx = mfq::metal::MlxMxWeight::from_arrays(
            "MXFP4", full({2, 32}, 0x22, uint8), full({2, 2}, 127, uint8), 64, 2);
        exercise(7, nint, mx);
        exercise(8, nint, mx);
        exercise(32, nint, mx);
        mfq_dynamic_geometry_phase(1);
        for (int tokens : {9, 13, 17, 21, 29, 47}) exercise(tokens, nint, mx);
        mfq_dynamic_geometry_phase(2);
        std::cout << "MFQ dynamic Metal geometry numerical/reuse tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
