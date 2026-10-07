#include "mlx_kernel_prepare.h"
#include "mlx_nint.h"
#include "mlx_sampling.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mlx::core;

extern "C" __attribute__((noinline)) void mfq_kernel_prepare_phase(int phase) {
    std::cout << "kernel_prepare_phase=" << phase << std::endl;
}

namespace {
template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

mfq::metal::MlxNintWeight weight() {
    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, 64);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, 64);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, 2);
    append<std::int64_t>(blob, 64);
    append<std::uint32_t>(blob, 2);
    append<std::uint32_t>(blob, 1);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0);
    append<std::uint16_t>(blob, 0);
    append<std::uint8_t>(blob, 3);
    append<std::uint8_t>(blob, 0);
    blob.insert(blob.end(), 64, 0x11);
    return mfq::metal::MlxNintWeight::from_blob(blob);
}
}

int main() {
    try {
#ifdef MFQ_MLX_METALLIB_DEFAULT
        metal::set_metallib_path(MFQ_MLX_METALLIB_DEFAULT);
#endif
        set_default_device(Device::gpu);
        const auto packed = weight();
        mfq::metal::MlxKernelPreparation preparation(2048, 2048);
        const auto memory = get_active_memory();
        mfq_kernel_prepare_phase(1);
        std::vector<array> descriptions;
        for (const auto rows : preparation.row_buckets()) {
            descriptions.push_back(packed.matmul(zeros({rows, 64}, float16)));
            preparation.add(descriptions.back());
        }
        const auto logits = zeros({1, 13}, float32);
        auto counts = mfq::metal::sample_token_counts_add(zeros({13}, int32), zeros({8}, int32));
        auto adjusted = mfq::metal::sample_apply_penalties(logits, counts, 0.1, 0.1, 1.05);
        descriptions.push_back(mfq::metal::sample_greedy(adjusted));
        preparation.add(descriptions.back());
        require(preparation.size() >= 3, "no compilation descriptions collected");
        require(preparation.unsupported().empty(), "native compilation interface missing");
        const auto description_memory = get_active_memory();
        preparation.finish();
        require(mfq::metal::MlxKernelPreparation::current() == nullptr, "preparation still active after load");
        std::cout << "metadata_memory=" << description_memory - memory
                  << " compilation_memory=" << get_active_memory() - description_memory << std::endl;
        require(get_active_memory() == description_memory, "pure compilation allocated tensor buffers");
        for (const auto& description : descriptions)
            require(!description.is_available(), "pure compilation evaluated an output");
        mfq_kernel_prepare_phase(2);
        for (const int rows : {9, 17, 29}) {
            auto output = astype(packed.matmul(ones({rows, 64}, float16)), float32);
            output.eval();
            for (std::size_t index = 0; index < output.size(); ++index)
                require(output.data<float>()[index] == 64.0f, "prepared projection is incorrect");
        }
        descriptions.back().eval();
        require(descriptions.back().item<int>() == 1, "prepared sampling is incorrect");
        mfq_kernel_prepare_phase(3);
        std::cout << "Metal pure kernel preparation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
